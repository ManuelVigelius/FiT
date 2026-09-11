"""
Sampling for the quadtree FiT variants, across the four ablation conditions.

    baseline   BASELINE model (pyramid_free), tree from the VariancePredictor
               applied to the current noisy state at every step. Tokens are
               NN-upsampled to fill their leaf footprint (no pyramid decoder).
    learned    LEARNED model, same per-step VariancePredictor tree, tokens
               decoded through the trained PyramidDecoder.
    gt         GT-variance model, tree recomputed per step from the TRUE
               variance of a real training latent (oracle structure).
    prior      GT-variance model, tree sampled ONCE per image from the trained
               class-conditional QuadtreePrior, then coarsened per step.

Why the sampler is written here rather than reusing noise_field_sampler
----------------------------------------------------------------------
Two properties of the quadtree models rule the existing sampler out:

  * These models predict CLEAN x1, not velocity (see Transport._loss_quadtree
    and loss_quadtree_dense). The Euler update therefore uses
    v = (x1_hat - x_t) / (1 - t) rather than the model output directly.
  * The state's REPRESENTATION changes every step, because the tree changes
    every step. The dense latent is the only thing that persists; the token
    layout is rebuilt from scratch each iteration.

The dense state is what we integrate, and the tree is a per-step encoding of
it. That keeps all four conditions on one code path: they differ only in where
the plan comes from, and in how tokens are turned back into a dense latent.

Noise and compression must agree
--------------------------------
For the baseline there is no pyramid decoder, so a leaf of side N reconstructs
to a single 2x2 latent patch that is NN-upsampled over its N x N footprint —
the output is piecewise-constant within a leaf. Injecting full-resolution i.i.d.
noise there would feed the model detail it structurally cannot cancel, and the
residual would accumulate over the trajectory. So the noise is area-averaged
within each leaf (`quantize_to_plan`), which is the same idea as sigma_inj in
the noise-field sampler: noise is drawn at the resolution the model can act on.

Timestep convention
-------------------
The dataset stores t_ds with 0 = clean and 1 = noise; the transport uses
t = 1 - t_ds. The VariancePredictor and the max_res_schedule were both trained
against t_ds, so DATASET convention is used for those and converted at the
model boundary. `_TDS` in the loop is always dataset convention.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL

from fit.model.quadtree_model import QuadtreeFiT
from fit.quadtree_compression.predictive_compressor import (
    PredictiveVarianceCompressor)
from fit.quadtree_compression.quadtree_compression import (
    plan_from_variance, REGION_SIZES)
from fit.quadtree_compression.variance_prediction import (
    VariancePredictor, predict_variance, load_latent)
from fit.data.in1k_gt_quadtree_latent_dataset import (
    gt_variance_grids, plan_from_gt_variance, normalize_max_res_schedule,
    min_leaf_size_at)
from fit.utils.utils import positions_to_grid
import quadtree_prior.structure as S


# ──────────────────────────────── constants ─────────────────────────────────
# Fixed by the trained checkpoints; changing one means retraining.
LATENT_SIZE = 32          # latent H == W for 256x256 images
PATCH_SIZE = 2
C_IN = 4
HIDDEN_SIZE = 1152
PAD_TO_MULTIPLE = 128
NUM_CLASSES = 1000
VAE_PATH = "stabilityai/sd-vae-ft-ema"
VAE_SCALE = 0.18215

WORKDIR = "/visinf/projects_students/mb_mvigel/workdir"
CKPT_ROOT = "/visinf/projects_students/mb_mvigel/checkpoints"
DATASET_ROOT = "/visinf/projects_students/mb_mvigel/datasets/imagenet1k_latents_256_sd_vae_ft_ema"

QUADTREE_THRESHOLD = 0.5
MAX_RES_SCHEDULE = [[0.0, 1], [0.7, 2], [0.85, 4], [0.95, 8]]

# Shared trunk config. Only token_input_dim / use_pyramid_decoder differ by arm,
# mirroring configs/quadtree/config_quadtree_xl_*.yaml.
_TRUNK = dict(
    context_size=256, patch_size=PATCH_SIZE, in_channels=C_IN,
    hidden_size=HIDDEN_SIZE, depth=36, num_heads=16, mlp_ratio=4.0,
    class_dropout_prob=0.1, num_classes=NUM_CLASSES,
    use_swiglu=True, use_swiglu_large=False,
    q_norm="layernorm", k_norm="layernorm", qk_norm_weight=False,
    rel_pos_embed="rope", online_rope=True,
    max_pe_len_h=16, max_pe_len_w=16, decouple=True, ori_max_pe_len=16,
    adaln_type="lora", adaln_lora_dim=288, use_size_cond=True,
)

CONDITIONS = {
    # name: (checkpoint project, uses pyramid, tree source)
    "baseline": dict(project="quadtree_xl_cluster_baseline",
                     pyramid=False, tree="vp"),
    "learned":  dict(project="quadtree_xl_cluster_learned",
                     pyramid=True,  tree="vp"),
    "gt":       dict(project="quadtree_xl_cluster_gt",
                     pyramid=True,  tree="gt"),
    "prior":    dict(project="quadtree_xl_cluster_gt",
                     pyramid=True,  tree="prior"),
}


# ─────────────────────────────── model loading ──────────────────────────────
def _resolve(project, step, ema):
    d = os.path.join(WORKDIR, project, "checkpoints", f"save-checkpoint-{step}")
    if not os.path.isdir(d):
        d = os.path.join(WORKDIR, project, "checkpoints", f"checkpoint-{step}")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no checkpoint-{step} under {project}")
    return d


def load_arm(cond, step, ema, device):
    """Build model + compressor for one condition and load trained weights.

    Accelerate numbers the files by prepare_model call order (train_quadtree.py):
    model.safetensors, model_1.safetensors (EMA), model_2.safetensors
    (compressor). The baseline's compressor file is a 40-byte empty header —
    pyramid_free builds zero parameters — which loads as an empty dict.
    """
    spec = CONDITIONS[cond]
    ckpt_dir = _resolve(spec["project"], step, ema)

    cfg = dict(_TRUNK)
    cfg["token_input_dim"] = HIDDEN_SIZE if spec["pyramid"] else 4 * C_IN
    cfg["use_pyramid_decoder"] = spec["pyramid"]

    model = QuadtreeFiT(**cfg).to(device).eval()
    sd = load_file(os.path.join(
        ckpt_dir, "model_1.safetensors" if ema else "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{cond}: state_dict mismatch\n  missing={missing[:6]}\n"
            f"  unexpected={unexpected[:6]}")

    comp = PredictiveVarianceCompressor(
        latent_channels=C_IN,
        c=256,
        d=HIDDEN_SIZE if spec["pyramid"] else 4 * C_IN,
        crop=LATENT_SIZE, pad_to_multiple=PAD_TO_MULTIPLE,
        with_decoder=spec["pyramid"], share_weights=True,
        residual_encoder=spec["pyramid"], residual_decoder=spec["pyramid"],
        pyramid_free=not spec["pyramid"],
        patch_size=PATCH_SIZE, load_base_head=False,
    ).to(device).eval()

    comp_sd = load_file(os.path.join(ckpt_dir, "model_2.safetensors"))
    if spec["pyramid"]:
        comp.load_state_dict(comp_sd, strict=True)
    elif comp_sd:
        raise RuntimeError(
            f"{cond}: expected an empty compressor (pyramid_free), got "
            f"{len(comp_sd)} tensors")

    for p in list(model.parameters()) + list(comp.parameters()):
        p.requires_grad_(False)
    return model, comp


def load_variance_predictor(device):
    path = os.path.join(CKPT_ROOT, "variance_predictor.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — the baseline/learned arms need it. Train it "
            "with:\n  python -m fit.quadtree_compression.variance_prediction "
            f"--train --save {path}")
    vp = VariancePredictor(latent_channels=C_IN, region_sizes=REGION_SIZES)
    vp.load_state_dict(torch.load(path, map_location="cpu"))
    return vp.to(device).eval().requires_grad_(False)


def load_prior(step, ema, device):
    from quadtree_prior.model import QuadtreePrior
    ckpt_dir = _resolve("quadtree_prior_s_cluster", step, ema)
    prior = QuadtreePrior(
        hidden_size=384, depth=6, num_heads=6, mlp_ratio=4.0,
        num_classes=NUM_CLASSES, class_dropout_prob=0.1,
        norm_layer="layernorm").to(device).eval()
    prior.load_state_dict(load_file(os.path.join(
        ckpt_dir, "model_1.safetensors" if ema else "model.safetensors")))
    return prior.requires_grad_(False)


# ──────────────────────────────── tree sources ──────────────────────────────
def coarsen_plan(plan, min_leaf):
    """Raise every leaf in `plan` to at least `min_leaf` on a side.

    Round-trips through the (16, 16) leaf-size grid: set each cell to
    max(size, min_leaf), snap the min_leaf-aligned blocks to a uniform size, and
    rebuild. Merging is always legal — four siblings collapsing into their
    parent is exactly the quadtree's own coarsening step.
    """
    if min_leaf <= 1:
        return plan
    g = S.sizes_grid_from_plan(plan, patch_grid=LATENT_SIZE // 2)
    g = g.clamp(min=min_leaf)
    # A block of side min_leaf must be uniform for plan_from_sizes_grid to
    # accept it; take the block max so a straddling coarse leaf wins.
    n = g.shape[0]
    for y in range(0, n, min_leaf):
        for x in range(0, n, min_leaf):
            blk = g[y:y + min_leaf, x:x + min_leaf]
            g[y:y + min_leaf, x:x + min_leaf] = int(blk.max())
    return S.plan_from_sizes_grid(g)


def plans_from_vp(vp, x_t, t_ds):
    """Per-step trees from the VariancePredictor, as in the training loader.

    Uncapped: the predictor path has no max_res_schedule (the threshold alone
    controls the token count), which is how config_quadtree_xl_learned.yaml and
    _baseline.yaml were trained.
    """
    var = predict_variance(vp, x_t, t_ds)
    out = []
    for i in range(x_t.shape[0]):
        levels, positions, sizes = plan_from_variance(
            [v[i] for v in var], QUADTREE_THRESHOLD, LATENT_SIZE)
        out.append(dict(levels=levels, positions=positions, sizes=sizes))
    return out


def plans_from_gt(latents, min_leaf):
    """Oracle trees: TRUE variance of a real latent, capped at `min_leaf`."""
    out = []
    for i in range(latents.shape[0]):
        levels, positions, sizes = plan_from_gt_variance(
            gt_variance_grids(latents[i]), QUADTREE_THRESHOLD, LATENT_SIZE,
            min_leaf_size=min_leaf)
        out.append(dict(levels=levels, positions=positions, sizes=sizes))
    return out


# ───────────────────────────── packing / decode ─────────────────────────────
def pack(comp, x_t, plans, labels, t_ds, device):
    """Encode + pack, then attach the FlexAttention block mask.

    The compressor's forward builds every packed field except block_mask, which
    the trainer builds outside the compiled model (train_quadtree.py:533).
    """
    from torch.nn.attention.flex_attention import create_block_mask
    packed = comp(x_t, plans, labels, t_ds, mode="encode")
    di = packed["doc_ids"].long()

    def doc_mask_mod(b, h, q_idx, kv_idx):
        return di[b, q_idx] == di[b, kv_idx]

    N = di.shape[1]
    packed["block_mask"] = create_block_mask(
        doc_mask_mod, 1, None, N, N, device=device)
    return packed


def scatter_baseline(tokens, plans, counts, device):
    """Dense latent from packed tokens WITHOUT a pyramid decoder.

    Inverse of `PyramidEncoder.base_features`, which is what pyramid_free hands
    the model. A size-N leaf's token is NOT a raw 2x2 patch: it is the 2x2
    QUADRANT MEANS of the leaf's 2N x 2N latent footprint, in channel order
    (c, p1, p2) — see _patchify_2x2 and base_features in
    adaptive_patch_pyramid.py. So each of the 4 quadrant values expands to fill
    an N x N latent block, giving a 2x2 arrangement of constant blocks.

    `sizes` is in PATCH CELLS, so a size-N leaf centered at (cy, cx) covers
    2N x 2N latent pixels starting at (cy - N, cx - N) — see
    S.sizes_grid_from_plan.

    This is `PyramidDecoder.base_predict`'s base term with no residual added —
    exactly what "no learned decompression" means, and it is why the baseline's
    output is piecewise-constant inside coarse leaves.
    """
    B = len(plans)
    out = torch.zeros(B, C_IN, LATENT_SIZE, LATENT_SIZE, device=device,
                      dtype=tokens.dtype)
    off = 0
    for i, plan in enumerate(plans):
        n = counts[i]
        tok = tokens[0, off:off + n]                       # (n, 16)
        off += n
        # (c, p1, p2) channel order — matches _patchify_2x2, NOT (p1, p2, c).
        patch = tok.reshape(n, C_IN, PATCH_SIZE, PATCH_SIZE)
        pos = plan["positions"].to(device).float()
        sizes = plan["sizes"].to(device).long()
        for k in range(n):
            N = int(sizes[k])
            y0 = int(round(float(pos[k, 0]) - N))
            x0 = int(round(float(pos[k, 1]) - N))
            p = patch[k]                                    # (C, 2, 2) quadrants
            if N != 1:
                # Each quadrant mean fills an N x N latent block. This is
                # repeat_interleave (expand each value), not repeat (tile).
                p = p.repeat_interleave(N, dim=1).repeat_interleave(N, dim=2)
            out[i, :, y0:y0 + 2 * N, x0:x0 + 2 * N] = p
    return out


def quantize_to_plan(x, plans):
    """Area-average `x` within each leaf, so it is constant over the footprint.

    Applied to the injected noise for the baseline arm: the model can only
    express a piecewise-constant signal inside a coarse leaf, so noise carrying
    finer detail than that is signal it can never cancel. Averaging puts the
    noise on the same resolution the model acts at — the same correction
    sigma_inj makes in the noise-field sampler.

    A size-N leaf covers 2N x 2N latent pixels. `base_features` reduces that to
    2x2 QUADRANT means, so the finest structure the model can carry there is
    four N x N constant blocks. This applies exactly that reduction, making it
    the projection onto what `scatter_baseline` can reproduce.
    """
    out = x.clone()
    for i, plan in enumerate(plans):
        pos = plan["positions"].float()
        sizes = plan["sizes"].long()
        for k in range(sizes.shape[0]):
            N = int(sizes[k])
            if N == 1:
                continue
            y0 = int(round(float(pos[k, 0]) - N))
            x0 = int(round(float(pos[k, 1]) - N))
            blk = x[i, :, y0:y0 + 2 * N, x0:x0 + 2 * N]
            # (C, 2, N, 2, N): quadrant index is the SLOW axis of each spatial
            # dim, so mean over the two N axes leaves the 2x2 quadrant means.
            m = blk.reshape(C_IN, PATCH_SIZE, N, PATCH_SIZE, N).mean(
                dim=(2, 4))                                  # (C, 2, 2)
            out[i, :, y0:y0 + 2 * N, x0:x0 + 2 * N] = \
                m.repeat_interleave(N, dim=1).repeat_interleave(N, dim=2)
    return out


# ──────────────────────────────── the sampler ───────────────────────────────
@torch.no_grad()
def sample_batch(cond, model, comp, labels, *, vp=None, gt_latents=None,
                 prior_plans=None, num_steps=50, cfg_scale=4.0,
                 device="cuda", dtype=torch.float32, generator=None,
                 log=None):
    """Euler flow-matching sampler over a per-step quadtree.

    The dense latent is the integration state. At each step the tree is chosen
    for the CURRENT state, the state is encoded onto that tree, the model
    predicts clean x1 in token space, and the result is decoded back to dense.

    The models predict x1, so the velocity is recovered as
    v = (x1_hat - x_t) / (1 - t) with t in transport convention.
    """
    spec = CONDITIONS[cond]
    B = labels.shape[0]
    schedule = normalize_max_res_schedule(MAX_RES_SCHEDULE)

    x = torch.randn(B, C_IN, LATENT_SIZE, LATENT_SIZE, device=device,
                    dtype=dtype, generator=generator)

    # t_ds: 1 = pure noise, 0 = clean. Integrate downward.
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

    for step in range(num_steps):
        t_ds = ts[step]
        t_next = ts[step + 1]
        dt = t_ds - t_next                       # positive
        t_batch_ds = torch.full((B,), float(t_ds), device=device, dtype=dtype)

        # --- 1. choose the tree for the current state -----------------------
        if spec["tree"] == "vp":
            plans = plans_from_vp(vp, x, t_batch_ds)
        else:
            min_leaf = min_leaf_size_at(float(t_ds), schedule)
            if spec["tree"] == "gt":
                plans = plans_from_gt(gt_latents, min_leaf)
            else:                                 # prior: sampled once, coarsened
                plans = [coarsen_plan(p, min_leaf) for p in prior_plans]

        if log is not None:
            log.append(dict(step=step, t=float(t_ds),
                            tokens=[int(p["sizes"].shape[0]) for p in plans]))

        # --- 2. align the state with the tree (baseline only) ---------------
        # Without a pyramid decoder the model's output is piecewise-constant
        # inside a leaf, so the input it conditions on should be too.
        x_in = quantize_to_plan(x, plans) if not spec["pyramid"] else x

        # --- 3. encode, run the model with CFG, decode ----------------------
        # CFG doubles the batch: conditional first, then the null class.
        x_cfg = torch.cat([x_in, x_in], 0)
        y_cfg = torch.cat([labels, torch.full_like(labels, NUM_CLASSES)], 0)
        t_cfg = torch.cat([t_batch_ds, t_batch_ds], 0)
        plans_cfg = plans + plans

        packed = pack(comp, x_cfg, plans_cfg, y_cfg, t_cfg, device)
        fwd = dict(y=packed["label"].to(torch.int).clamp(min=0),
                   grid=packed["grid"], mask=packed["mask"].float(),
                   size=packed["size"], tsize=packed["tsize"],
                   doc_ids=packed["doc_ids"].long(),
                   block_mask=packed["block_mask"])
        # The model takes transport convention (0 = noise, 1 = clean).
        tokens = model(packed["feature"], 1.0 - packed["t"], **fwd)

        if spec["pyramid"]:
            x1_all = comp(tokens, plans_cfg, packed["counts"], x_cfg,
                          mode="decode")
        else:
            x1_all = scatter_baseline(tokens, plans_cfg, packed["counts"],
                                      device)

        x1_cond, x1_null = x1_all[:B], x1_all[B:]
        x1 = x1_null + cfg_scale * (x1_cond - x1_null)

        # --- 4. Euler step on the dense state -------------------------------
        # x1-prediction -> velocity. t here is transport convention, so
        # (1 - t) == t_ds; floor it to keep the last step finite.
        v = (x1 - x) / t_ds.clamp(min=1e-3)
        x = x + dt * v

    return x


# ──────────────────────────────────── main ──────────────────────────────────
def _discover_latents(root):
    import glob
    for sub in ("greater_than_256_crop", "greater_than_512_crop"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            f = sorted(glob.glob(os.path.join(d, "*.safetensors")))
            if f:
                return f
    return sorted(glob.glob(os.path.join(root, "**", "*.safetensors"),
                            recursive=True))


def _center_crop(latent, c=LATENT_SIZE):
    _, Hs, Ws = latent.shape
    ph, pw = max(0, c - Hs), max(0, c - Ws)
    if ph or pw:
        latent = F.pad(latent[None], (0, pw, 0, ph), mode="reflect")[0]
        _, Hs, Ws = latent.shape
    return latent[:, (Hs - c) // 2:(Hs - c) // 2 + c,
                  (Ws - c) // 2:(Ws - c) // 2 + c]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True,
                    choices=list(CONDITIONS) + ["all"])
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--step", type=int, default=8000,
                    help="diffusion checkpoint step")
    ap.add_argument("--prior-step", type=int, default=50000)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/visinf/projects_students/mb_mvigel/samples")
    ap.add_argument("--dataset", default=DATASET_ROOT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema = not args.no_ema
    conds = list(CONDITIONS) if args.condition == "all" else [args.condition]

    print(f"Device: {device}  EMA: {ema}  steps: {args.num_steps}  "
          f"CFG: {args.cfg_scale}")

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()

    # Shared across conditions so every arm generates the SAME class labels —
    # FID is computed per condition, but matched labels make the sample grids
    # directly comparable by eye.
    g_lab = torch.Generator(device="cpu").manual_seed(args.seed)
    all_labels = torch.randint(0, NUM_CLASSES, (args.n_images,),
                               generator=g_lab)

    for cond in conds:
        spec = CONDITIONS[cond]
        print(f"\n{'=' * 60}\nCondition: {cond}  "
              f"({spec['project']}, tree={spec['tree']})")

        model, comp = load_arm(cond, args.step, ema, device)
        vp = load_variance_predictor(device) if spec["tree"] == "vp" else None
        prior = load_prior(args.prior_step, ema, device) \
            if spec["tree"] == "prior" else None

        latent_files = None
        if spec["tree"] == "gt":
            latent_files = _discover_latents(args.dataset)
            if not latent_files:
                raise RuntimeError(f"no latents under {args.dataset}")
            print(f"  GT trees from {len(latent_files)} train latents")

        out_dir = os.path.join(args.out, cond)
        os.makedirs(out_dir, exist_ok=True)
        token_log = []

        done = 0
        while done < args.n_images:
            bs = min(args.batch_size, args.n_images - done)
            labels = all_labels[done:done + bs].to(device)
            gen = torch.Generator(device=device).manual_seed(args.seed + done)

            gt_latents = prior_plans = None
            if spec["tree"] == "gt":
                idx = torch.randint(0, len(latent_files), (bs,),
                                    generator=g_lab).tolist()
                gt_latents = torch.stack([
                    _center_crop(load_latent(latent_files[j])).float()
                    for j in idx]).to(device)
            elif spec["tree"] == "prior":
                # Sampled ONCE per image, before the loop. The schedule only
                # coarsens this fixed tree per step; it never re-samples.
                from quadtree_prior.sample import sample_plans
                prior_plans, _ = sample_plans(prior, labels, device=device)

            x = sample_batch(
                cond, model, comp, labels, vp=vp, gt_latents=gt_latents,
                prior_plans=prior_plans, num_steps=args.num_steps,
                cfg_scale=args.cfg_scale, device=device, generator=gen,
                log=token_log if done == 0 else None)

            imgs = vae.decode(x / VAE_SCALE).sample
            imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            for k in range(bs):
                Image.fromarray(imgs[k]).save(
                    os.path.join(out_dir, f"{done + k:06d}.png"))

            done += bs
            print(f"  {done}/{args.n_images}", flush=True)

        # Token counts per step: the cheapest way to see the tree behaving —
        # a baseline run that ratchets monotonically coarser is the VP-on-a-
        # quantized-state feedback loop, not a model failure.
        with open(os.path.join(out_dir, "token_counts.json"), "w") as f:
            json.dump(token_log, f, indent=2)
        if token_log:
            first = sum(token_log[0]["tokens"]) / len(token_log[0]["tokens"])
            last = sum(token_log[-1]["tokens"]) / len(token_log[-1]["tokens"])
            print(f"  mean tokens/image: {first:.1f} (t=1) -> {last:.1f} (t=0)")

        del model, comp, vp, prior
        torch.cuda.empty_cache()

    print(f"\nDone. Samples under {args.out}")


if __name__ == "__main__":
    main()
