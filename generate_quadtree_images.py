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
    gt_struct  GT-STRUCT model (also conditioned on the full tree's 84 split
               bits), trees as in `gt`.
    prior_struct  GT-STRUCT model, trees as in `prior`.

The *_struct models take the structure vector of the FULL tree: computed once
per image before the loop, from the uncapped GT tree or the uncoarsened prior
sample, and never from the per-step (schedule-pruned) plans.

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
DATASET_ROOT = "/visinf/projects_students/mb_mvigel/datasets"

# A fully dense tree is one size-1 leaf per 2x2 patch cell: (32/2)^2 = 256.
DENSE_TOKENS = (LATENT_SIZE // 2) ** 2
# Coarsest legal tree: one size-8 leaf per quadrant of the 32x32 latent.
MIN_TOKENS = (LATENT_SIZE // 2 // 8) ** 2

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
    "gt_struct":    dict(project="quadtree_xl_cluster_gt_struct",
                         pyramid=True, tree="gt", struct=True),
    "prior_struct": dict(project="quadtree_xl_cluster_gt_struct",
                         pyramid=True, tree="prior", struct=True),
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
    cfg["struct_cond_dim"] = S.STRUCT_DIM if spec.get("struct") else 0

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
        # share_weights=True makes merges[0..2] (and splits[0..2]) the SAME
        # module object, so safetensors refuses to write the aliases and
        # accelerate drops them: the checkpoint holds only merges.0/splits.0.
        # Loading non-strictly restores the real weights into that one shared
        # object, and every alias sees them because it IS that object. Guard
        # anyway, so a genuinely truncated checkpoint still fails loudly.
        missing_c, unexpected_c = comp.load_state_dict(comp_sd, strict=False)
        aliased = [k for k in missing_c
                   if k.startswith(("encoder.merges.", "decoder.splits."))
                   and k.split(".")[2] != "0"]
        real = [k for k in missing_c if k not in aliased]
        if real or unexpected_c:
            raise RuntimeError(
                f"{cond}: compressor state_dict mismatch beyond the shared "
                f"merge/split aliases\n  missing={real[:6]}\n"
                f"  unexpected={unexpected_c[:6]}")
        # The aliases must genuinely be the same object, or those levels would
        # silently run on random init.
        enc_m, dec_s = comp.encoder.merges, comp.decoder.splits
        if any(m is not enc_m[0] for m in enc_m) or \
           any(s is not dec_s[0] for s in dec_s):
            raise RuntimeError(
                f"{cond}: checkpoint omits merge/split levels but this "
                f"compressor does not share them — weights would be random. "
                f"Was it trained with share_weights: false?")
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
    return S.plan_from_sizes_grid(g, device=plan['sizes'].device)


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


def plans_dense(B, min_leaf, device):
    """Uniform tree at the FINEST leaf the schedule allows.

    The dense control: no adaptive compression at all, every leaf the same
    size. `min_leaf` is 1 (fully dense, 256 tokens) except for the schedule-
    capped arms, where the cap still applies -- a size-1 tree at t > 0.7 is a
    resolution the GT/prior models were never trained on, so forcing it would
    measure an out-of-distribution failure rather than the cost of compression.
    """
    g = torch.full((LATENT_SIZE // 2, LATENT_SIZE // 2), int(min_leaf),
                   dtype=torch.long)
    plan = S.plan_from_sizes_grid(g, device=device)
    # Same structure for every image in the batch; the dicts are read-only here
    # so sharing one is safe and avoids B identical rebuilds.
    return [plan] * B


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
                 noise=None, log=None, return_last_plans=False, dense=False):
    """Euler flow-matching sampler over a per-step quadtree.

    The dense latent is the integration state. At each step the tree is chosen
    for the CURRENT state, the state is encoded onto that tree, the model
    predicts clean x1 in token space, and the result is decoded back to dense.

    The models predict x1, so the velocity is recovered as
    v = (x1_hat - x_t) / (1 - t) with t in transport convention.

    `noise` (B, C, H, W), if given, is the initial state; otherwise it is drawn
    from `generator`. Passing it in keeps an image's starting point independent
    of which batch or shard it lands in.
    """
    spec = CONDITIONS[cond]
    B = labels.shape[0]
    schedule = normalize_max_res_schedule(MAX_RES_SCHEDULE)

    # Structure conditioning: from the FULL tree, fixed for the whole
    # trajectory. The per-step `plans` below are schedule-pruned and must never
    # feed it. Both CFG branches get it (the null class sees the tree too).
    struct_cfg = None
    if spec.get("struct"):
        if spec["tree"] == "gt":
            full = plans_from_gt(gt_latents, 1)
        elif spec["tree"] == "prior":
            full = prior_plans                    # sampled uncoarsened
        else:
            raise ValueError(f"{cond}: structure conditioning needs a full "
                             f"tree, which tree={spec['tree']!r} lacks")
        struct = torch.stack([
            S.structure_vector(S.sizes_grid_from_plan(p)) for p in full])
        struct_cfg = torch.cat([struct, struct], 0)[None].to(device, dtype)

    if noise is not None:
        x = noise.to(device=device, dtype=dtype)
    else:
        x = torch.randn(B, C_IN, LATENT_SIZE, LATENT_SIZE, device=device,
                        dtype=dtype, generator=generator)

    last_plans = None          # num_steps=0 would otherwise leave this unbound

    # t_ds: 1 = pure noise, 0 = clean. Integrate downward.
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

    for step in range(num_steps):
        t_ds = ts[step]
        t_next = ts[step + 1]
        dt = t_ds - t_next                       # positive
        t_batch_ds = torch.full((B,), float(t_ds), device=device, dtype=dtype)

        # --- 1. choose the tree for the current state -----------------------
        if dense:
            # Dense control. The VP arms trained uncapped, so they go fully
            # dense (leaf 1); the schedule-capped arms keep their cap.
            min_leaf = 1 if spec["tree"] == "vp" \
                else min_leaf_size_at(float(t_ds), schedule)
            plans = plans_dense(B, min_leaf, device)
        elif spec["tree"] == "vp":
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
        # Cheap invariant: plan_to_masks derives mask device from the plan
        # tensors, so a stray CPU plan surfaces as a device mismatch deep inside
        # the decoder. Catch it here, where the source is obvious.
        for _p in plans:
            if _p['sizes'].device.type != torch.device(device).type:
                raise RuntimeError(
                    f"plan on {_p['sizes'].device}, expected {device}. "
                    "Plans must be built on the compute device.")

        plans_cfg = plans + plans

        packed = pack(comp, x_cfg, plans_cfg, y_cfg, t_cfg, device)
        fwd = dict(y=packed["label"].to(torch.int).clamp(min=0),
                   grid=packed["grid"], mask=packed["mask"].float(),
                   size=packed["size"], tsize=packed["tsize"],
                   doc_ids=packed["doc_ids"].long(),
                   block_mask=packed["block_mask"], struct=struct_cfg)
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
        last_plans = plans

    # The overlay must show the tree the FINAL prediction actually used, which
    # is the one chosen at the last step -- not a fresh tree for the clean
    # latent, which no model ever saw.
    return (x, last_plans) if return_last_plans else x


# ─────────────────────────────────── debug ──────────────────────────────────
def draw_quadtree(img, plan, colour=(255, 64, 64), width=1):
    """Overlay a plan's leaf boundaries on an HxWx3 uint8 image.

    Plan geometry (see `scatter_baseline`): `sizes` is in PATCH CELLS, and a
    size-N leaf centered at (cy, cx) covers 2N x 2N LATENT pixels starting at
    (cy - N, cx - N). The VAE upsamples latent -> image by a fixed factor, so
    one latent pixel is `scale` image pixels on a side.
    """
    out = img.copy()
    H = out.shape[0]
    scale = H // LATENT_SIZE            # 32 latent px -> 256 image px => 8
    pos = plan["positions"].detach().cpu()
    sizes = plan["sizes"].detach().cpu()

    for k in range(int(sizes.shape[0])):
        N = int(sizes[k])
        y0 = int(round(float(pos[k, 0]) - N)) * scale
        x0 = int(round(float(pos[k, 1]) - N)) * scale
        side = 2 * N * scale
        y1, x1 = min(y0 + side, H), min(x0 + side, out.shape[1])
        y0, x0 = max(y0, 0), max(x0, 0)
        if y1 <= y0 or x1 <= x0:
            continue
        # Draw inward so edge leaves keep their border inside the frame.
        out[y0:min(y0 + width, y1), x0:x1] = colour      # top
        out[max(y1 - width, y0):y1, x0:x1] = colour      # bottom
        out[y0:y1, x0:min(x0 + width, x1)] = colour      # left
        out[y0:y1, max(x1 - width, x0):x1] = colour      # right
    return out


def plot_token_counts(traj, step_ts, cond, path):
    """One line per image: token count over the sampling trajectory.

    x is the diffusion time in DATASET convention (t=1 noise -> t=0 clean), so
    the axis is drawn decreasing left-to-right: the run starts at the left and
    ends clean at the right. Returns False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing, skipping plot)")
        return False

    if not traj:
        return False
    n = len(traj)
    xs = step_ts if step_ts else list(range(len(traj[0])))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Many overlapping lines: keep each faint so density reads as shade, and
    # overlay the mean so the population trend stays legible.
    alpha = max(0.06, min(0.5, 12.0 / n))
    for t in traj:
        ax.plot(xs, t, color="#4a7ebb", alpha=alpha, lw=0.8)
    mean = [sum(c) / n for c in zip(*traj)]
    ax.plot(xs, mean, color="#d94", lw=2.0, label="mean")

    # Context: 256 = fully dense, 4 = coarsest legal tree. A trajectory pinned
    # to either bound is degenerate, not a model result.
    ax.axhline(DENSE_TOKENS, color="#888", ls=":", lw=1,
               label=f"dense {DENSE_TOKENS}")
    ax.axhline(MIN_TOKENS, color="#888", ls="--", lw=1,
               label=f"coarsest {MIN_TOKENS}")

    if step_ts:
        ax.set_xlabel("t  (1 = noise  ->  0 = clean)")
        ax.set_xlim(max(xs), min(xs))          # decreasing: noise on the left
    else:
        ax.set_xlabel("sampling step")
    ax.set_ylabel("tokens")
    ax.set_title(f"{cond}: tokens per image over sampling ({n} images)")
    ax.set_ylim(0, DENSE_TOKENS * 1.08)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# ──────────────────────────────────── main ──────────────────────────────────
def _discover_latents(root):
    """Locate the train-split crop folders under `root`.

    Mirrors GTQuadtreeLatentDataset._discover_files, with one deliberate
    difference: there is NO recursive fallback. `root` here may be the shared
    datasets directory holding ~1.1M files, and a recursive sweep of that
    either hangs or silently picks up latents from an unrelated dataset. A
    wrong GT latent is not a crash — it is a quietly invalid oracle tree that
    would corrupt the `gt` condition's FID. So: find the crop folders or fail.
    """
    import glob
    files, looked = [], []
    for sub in ("greater_than_256_crop", "greater_than_512_crop"):
        d = os.path.join(root, sub)
        looked.append(d)
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, "*.safetensors"))))
    if not files:
        raise RuntimeError(
            "no latent crop folders found. Looked for:\n  "
            + "\n  ".join(looked)
            + "\nPass the directory that CONTAINS greater_than_256_crop "
              "via --dataset."
        )
    return files


def _to_images(vae, x):
    """Latents -> HWC uint8 numpy, the standard VAE decode path."""
    imgs = vae.decode(x / VAE_SCALE).sample
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return imgs.permute(0, 2, 3, 1).cpu().numpy()


def _center_crop(latent, c=LATENT_SIZE):
    _, Hs, Ws = latent.shape
    ph, pw = max(0, c - Hs), max(0, c - Ws)
    if ph or pw:
        latent = F.pad(latent[None], (0, pw, 0, ph), mode="reflect")[0]
        _, Hs, Ws = latent.shape
    return latent[:, (Hs - c) // 2:(Hs - c) // 2 + c,
                  (Ws - c) // 2:(Ws - c) // 2 + c]


# ───────────────────────────── reproducible runs ────────────────────────────
# A run is split into a FIXED grid of batches: batch b always holds images
# [b * batch_size, (b + 1) * batch_size), and shard i of N takes batches
# i, i + N, i + 2N, ... Every image therefore sees the same inputs AND the same
# batch-mates no matter how many GPUs produce the run. Batch-mates matter: they
# change GPU reduction order, and through it the per-step VP trees, so images are
# only bit-reproducible for a fixed batch grid. That is why batch_size is part of
# the run record below.
#
# Each finished batch writes batches/<b>.json LAST. Its presence is what marks
# the batch done, so a killed run resumes where it stopped, and --merge can
# verify that a run is complete before anything is scored.


def _manifest(cond, args):
    """The arguments that determine this condition's images."""
    keys = ["n_images", "batch_size", "num_steps", "cfg_scale", "step",
            "no_ema", "seed", "debug"]
    tree = CONDITIONS[cond]["tree"]
    m = dict(condition=cond, **{k: getattr(args, k) for k in keys})
    if tree == "gt":
        m["dataset"] = os.path.abspath(args.dataset)
    if tree == "prior":
        m["prior_step"] = args.prior_step
    return m


def _check_manifest(out_dir, manifest):
    """Record `manifest` as out_dir/run.json, or verify it matches the record.

    Every shard and every resume goes through here, so one folder can never mix
    images from two configurations. Shards start concurrently: os.link fails if
    the target exists, which makes "first writer wins" atomic.
    """
    path = os.path.join(out_dir, "run.json")
    if not os.path.exists(path) and any(
            f.endswith(".png") for f in os.listdir(out_dir)):
        raise RuntimeError(
            f"{out_dir} already holds PNGs from a run without run.json "
            "(older script version). Use a fresh --out or clear the folder.")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    try:
        os.link(tmp, path)
    except FileExistsError:
        with open(path) as f:
            recorded = json.load(f)
        if recorded != manifest:
            diff = {k: (recorded.get(k), manifest.get(k))
                    for k in sorted(set(recorded) | set(manifest))
                    if recorded.get(k) != manifest.get(k)}
            raise RuntimeError(
                f"{out_dir} holds a run with different settings "
                f"(recorded, requested): {diff}. Use a fresh --out, or delete "
                "the folder to start over.")
    finally:
        os.remove(tmp)


def _fixed_inputs(seed, n_images, n_latents=None):
    """Labels, initial noise and (gt only) training-latent indices for ALL images.

    Drawn once on CPU, in this order, from one generator seeded with `seed`.
    Labels and noise use the same draws as measure_fid.py, so for equal seed and
    image count they match the FiT FID runs. Every shard draws the full set and
    slices its own images out, so an image's inputs depend only on its index.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.randint(0, NUM_CLASSES, (n_images,), generator=g)
    noise = torch.randn(n_images, C_IN, LATENT_SIZE, LATENT_SIZE, generator=g)
    gt_idx = (torch.randint(0, n_latents, (n_images,), generator=g)
              if n_latents else None)
    return labels, noise, gt_idx


def _batch_path(out_dir, b):
    return os.path.join(out_dir, "batches", f"{b:05d}.json")


@torch.no_grad()
def generate(cond, args, vae, device):
    """Generate this shard's batches of one condition, skipping finished ones."""
    spec = CONDITIONS[cond]
    out_dir = os.path.join(args.out, cond)
    os.makedirs(os.path.join(out_dir, "batches"), exist_ok=True)
    _check_manifest(out_dir, _manifest(cond, args))

    n_batches = math.ceil(args.n_images / args.batch_size)
    mine = range(args.shard_id, n_batches, args.num_shards)
    todo = [b for b in mine if not os.path.exists(_batch_path(out_dir, b))]
    tag = f"[{cond} shard {args.shard_id}/{args.num_shards}]"
    print(f"\n{'=' * 60}\n{tag} {spec['project']}, tree={spec['tree']}: "
          f"{len(todo)} of {len(mine)} batches to generate", flush=True)
    if not todo:
        return

    latent_files = None
    if spec["tree"] == "gt":
        latent_files = _discover_latents(args.dataset)
        print(f"  GT trees from {len(latent_files)} train latents")
    labels_all, noise_all, gt_idx_all = _fixed_inputs(
        args.seed, args.n_images,
        len(latent_files) if latent_files else None)

    model, comp = load_arm(cond, args.step, not args.no_ema, device)
    vp = load_variance_predictor(device) if spec["tree"] == "vp" else None
    prior = load_prior(args.prior_step, not args.no_ema, device) \
        if spec["tree"] == "prior" else None

    for i, b in enumerate(todo):
        s = b * args.batch_size
        e = min(s + args.batch_size, args.n_images)
        labels = labels_all[s:e].to(device)
        noise = noise_all[s:e]

        gt_latents = prior_plans = None
        if spec["tree"] == "gt":
            gt_latents = torch.stack([
                _center_crop(load_latent(latent_files[j])).float()
                for j in gt_idx_all[s:e].tolist()]).to(device)
        elif spec["tree"] == "prior":
            # Sampled ONCE per image, before the loop. The schedule only
            # coarsens this fixed tree per step; it never re-samples.
            from quadtree_prior.sample import sample_plans
            # plan_device matters: plan_to_masks builds its masks on the
            # device of the plan TENSORS (levels.device), ignoring the
            # device arg threaded through _batch_masks. CPU plans would
            # therefore hand the decoder CPU masks against CUDA tokens.
            # The generator is keyed to the batch start, which the fixed
            # batch grid makes shard-independent.
            g_prior = torch.Generator(device=device).manual_seed(args.seed + s)
            prior_plans, _ = sample_plans(prior, labels, device=device,
                                          plan_device=device,
                                          generator=g_prior)

        batch_log = []
        x, last_plans = sample_batch(
            cond, model, comp, labels, vp=vp, gt_latents=gt_latents,
            prior_plans=prior_plans, num_steps=args.num_steps,
            cfg_scale=args.cfg_scale, device=device, noise=noise,
            log=batch_log, return_last_plans=True)
        record = dict(start=s, log=batch_log,
                      final=[int(p["sizes"].shape[0]) for p in last_plans])

        imgs = _to_images(vae, x)
        for k in range(e - s):
            Image.fromarray(imgs[k]).save(
                os.path.join(out_dir, f"{s + k:06d}.png"))
            if args.debug:
                Image.fromarray(draw_quadtree(imgs[k], last_plans[k])).save(
                    os.path.join(out_dir, f"{s + k:06d}_tree.png"))

        # --- dense control -----------------------------------------------
        # Same noise, same labels, same GT latents / prior trees: the ONLY
        # difference is the tree. Pairs <id>.png and <id>_dense.png are
        # therefore directly comparable, and isolate what the adaptive
        # tree costs against no compression at all.
        if args.debug:
            x_d, dense_plans = sample_batch(
                cond, model, comp, labels, vp=vp, gt_latents=gt_latents,
                prior_plans=prior_plans, num_steps=args.num_steps,
                cfg_scale=args.cfg_scale, device=device, noise=noise,
                return_last_plans=True, dense=True)
            imgs_d = _to_images(vae, x_d)
            for k in range(e - s):
                Image.fromarray(imgs_d[k]).save(
                    os.path.join(out_dir, f"{s + k:06d}_dense.png"))
            record["dense_final"] = [int(p["sizes"].shape[0])
                                     for p in dense_plans]

        # Written last, atomically: this file is the batch's completion mark.
        path = _batch_path(out_dir, b)
        with open(path + ".tmp", "w") as f:
            json.dump(record, f)
        os.replace(path + ".tmp", path)
        print(f"{tag} batch {i + 1}/{len(todo)} (images {s}-{e - 1})",
              flush=True)

    del model, comp, vp, prior
    torch.cuda.empty_cache()


def _compute_fid(image_dir, n_images):
    """clean-fid against the same cached custom reference as measure_fid.py."""
    from cleanfid import fid as cleanfid
    name = os.environ.get("FID_REF_NAME", "fit_imagenet256")
    mode = os.environ.get("FID_MODE", "clean")
    if not cleanfid.test_stats_exists(name, mode=mode):
        raise RuntimeError(
            f"clean-fid reference '{name}' (mode={mode}) is not built yet. "
            "Build it once with tools/cluster_fid.sh (measure_fid.py).")
    return float(cleanfid.compute_fid(
        image_dir, dataset_name=name, dataset_split="custom", mode=mode,
        num_gen=n_images))


def merge(cond, args):
    """Verify a finished run, combine its per-batch stats, optionally score FID.

    Returns (fid or None, run record).
    """
    out_dir = os.path.join(args.out, cond)
    path = os.path.join(out_dir, "run.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing: nothing generated for "
                                f"{cond} under {args.out}")
    with open(path) as f:
        run = json.load(f)
    n_images, bs = run["n_images"], run["batch_size"]
    n_batches = math.ceil(n_images / bs)
    print(f"\n{'=' * 60}\n[{cond}] merging {n_batches} batches "
          f"({n_images} images)")

    # Refuse to summarise or score an incomplete run: clean-fid would happily
    # score whatever PNGs exist and return a confident but wrong FID.
    missing = [b for b in range(n_batches)
               if not os.path.exists(_batch_path(out_dir, b))]
    if missing:
        raise RuntimeError(
            f"[{cond}] {len(missing)}/{n_batches} batches unfinished (first: "
            f"{missing[:5]}). Rerun generation with the same settings to "
            "resume.")
    no_png = [i for i in range(n_images)
              if not os.path.isfile(os.path.join(out_dir, f"{i:06d}.png"))]
    if no_png:
        raise RuntimeError(f"[{cond}] {len(no_png)} PNGs missing "
                           f"(first: {no_png[:5]})")

    recs = []
    for b in range(n_batches):
        with open(_batch_path(out_dir, b)) as f:
            recs.append(json.load(f))

    # Token counts per step: the cheapest way to see the tree behaving —
    # a baseline run that ratchets monotonically coarser is the VP-on-a-
    # quantized-state feedback loop, not a model failure.
    token_log = recs[0]["log"]                     # per-step, first batch
    with open(os.path.join(out_dir, "token_counts.json"), "w") as f:
        json.dump(token_log, f, indent=2)
    if token_log:
        first = sum(token_log[0]["tokens"]) / len(token_log[0]["tokens"])
        last = sum(token_log[-1]["tokens"]) / len(token_log[-1]["tokens"])
        print(f"  mean tokens/image: {first:.1f} (t=1) -> {last:.1f} (t=0)")

    if run["debug"]:
        # Per-step trace of every batch -> one trajectory per image. The step
        # grid is identical across batches, so transposing each batch's log
        # and concatenating along the image axis is safe.
        step_ts = [e["t"] for e in token_log]
        traj = [[e["tokens"][k] for e in r["log"]]
                for r in recs for k in range(len(r["final"]))]
        final_tokens = [t for r in recs for t in r["final"]]
        dense_tokens = [t for r in recs for t in r.get("dense_final", [])]
        with open(os.path.join(out_dir, "token_traj.json"), "w") as f:
            json.dump(dict(t=step_ts, traj=traj, final=final_tokens), f)
        plot_path = os.path.join(out_dir, f"tokens_{cond}.png")
        if plot_token_counts(traj, step_ts, cond, plot_path):
            print(f"  wrote {plot_path}")
        print(f"  final tokens: min {min(final_tokens)} "
              f"max {max(final_tokens)} "
              f"mean {sum(final_tokens) / len(final_tokens):.1f} "
              f"(dense {DENSE_TOKENS})")
        if dense_tokens:
            mt = sum(final_tokens) / len(final_tokens)
            md = sum(dense_tokens) / len(dense_tokens)
            print(f"  dense control: mean {md:.1f} tokens "
                  f"({mt / md:.2f}x compression in the adaptive run)")

    fid = None
    if args.fid:
        if run["debug"]:
            raise RuntimeError(
                f"[{cond}] was generated with --debug; its folder also holds "
                "_tree/_dense PNGs, which clean-fid would score too.")
        if n_images < 2048:
            # Fewer samples than Inception feature dims -> singular covariance.
            raise RuntimeError(f"[{cond}] FID needs >= 2048 images, "
                               f"run has {n_images}")
        fid = _compute_fid(out_dir, n_images)
        print(f"  FID = {fid:.4f}")
    return fid, run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, nargs="+",
                    choices=list(CONDITIONS) + ["all"])
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="part of the run record: images are bit-"
                         "reproducible only for a fixed batch size")
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--step", type=int, default=8000,
                    help="diffusion checkpoint step")
    ap.add_argument("--prior-step", type=int, default=50000)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/visinf/home/mb_mvigel/samples")
    ap.add_argument("--dataset", default=DATASET_ROOT)
    ap.add_argument("--debug", action="store_true",
                    help="also write *_tree.png overlays of the final-step "
                         "quadtree, a dense control and a per-condition "
                         "token-count plot")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the run into this many independent processes")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="which shard this process generates (0-based)")
    ap.add_argument("--merge", action="store_true",
                    help="don't generate: verify finished runs and combine "
                         "their per-batch stats")
    ap.add_argument("--fid", action="store_true",
                    help="with --merge: score each run with clean-fid "
                         "(reference from FID_REF_NAME / FID_MODE)")
    args = ap.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        ap.error("need 0 <= --shard-id < --num-shards")
    if args.fid and not args.merge:
        ap.error("--fid only works together with --merge")

    conds = list(CONDITIONS) if "all" in args.condition else args.condition

    if args.merge:
        results = {}
        for cond in conds:
            fid, run = merge(cond, args)
            if fid is not None:
                results[cond] = dict(fid=fid, run=run)
        if results:
            path = os.path.join(args.out, "fid_results.json")
            existing = {}
            if os.path.isfile(path):
                with open(path) as f:
                    existing = json.load(f)
            existing.update(results)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"\n{'=' * 60}\nFID results:")
            for cond, r in results.items():
                print(f"  {cond:10s} FID = {r['fid']:.4f}")
            print(f"Saved {path}")
        return

    # Same inputs must give the same pixels across processes and reruns. With
    # benchmark mode on, cuDNN can pick a different conv algorithm per process
    # (VAE decode), which changes results in the last bits.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  EMA: {not args.no_ema}  steps: {args.num_steps}"
          f"  CFG: {args.cfg_scale}  shard: {args.shard_id}/{args.num_shards}")

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()

    for cond in conds:
        generate(cond, args, vae, device)

    print(f"\nDone. Samples under {args.out}")


if __name__ == "__main__":
    main()
