"""Checkpoint-compatibility test for :class:`~fit.model.quadtree_model.QuadtreeFiT`.

The question this answers
------------------------
`QuadtreeFiT` is a rewrite of `FiT`: packed-only forward, a learned per-leaf-size
embedding instead of the sinusoidal `SizeEmbedder`, tokens that arrive from a
learned compressor. Any of those rewrites could silently break the trunk — a
transposed grid, an off-by-one RoPE convention, conditioning attached to the wrong
token — and training would still produce a plausible-looking loss curve.

So we test against ground truth instead: load a PRETRAINED FiT checkpoint into
`QuadtreeFiT` and sample from it. Recognisable ImageNet images can only come out
if the trunk, RoPE, adaLN conditioning, packing and block mask are all wired
exactly as the checkpoint expects. Noise means something is wrong.

Why zero compression
--------------------
With ``QUADTREE_THRESHOLD = 0.0`` nothing is ever flat, so the quadtree walk
bottoms out at size-1 leaves everywhere: 256 tokens on a 16x16 grid, each one a
raw 2x2 latent patch. That is *exactly* the token set the pretrained FiT consumes
at 256x256 — same content, same count. The quadtree is then a pure relabelling of
the standard path, which is what makes the comparison valid. Two residual
differences remain, and they are the point of the test:

  * token order is the quadtree's recursion (Z) order, not row-major, and
  * grid coordinates are patch CENTERS (0.5 .. 15.5), not corners (0 .. 15).

RoPE is relative, so the constant +0.5 offset cancels between query and key and
the checkpoint should not care. If it does, this script shows it.

This also verifies the claim in the module docstring of
`adaptive_patch_pyramid` that the pyramid only ever supplies a *residual*: at
level 0 the tokens must already carry the full signal, since `MODE = "patchify"`
reconstructs images from raw patches alone.

The two modes
-------------
``MODE = "patchify"``  (the reference)
    Build the model with ``token_input_dim=16`` so ``x_embedder`` is the
    checkpoint's own ``PatchEmbedder``, and feed raw 2x2-patchified latents in
    quadtree order. Expect REAL IMAGES.

``MODE = "encoder"``   (the contrast)
    Keep ``token_input_dim=1152`` (``x_embedder`` is Identity) and take tokens
    from the level-0 output of `PyramidEncoder`. Those weights are randomly
    initialised and have never been aligned with the pretrained token space, so
    expect NOISE. This is not a failure of the trunk — it measures how far the
    untrained encoder sits from the embedding the checkpoint expects.

``MODE = "both"`` writes them side by side, reference on the left.

Prediction head
---------------
``LOAD_HEAD`` controls whether ``final_layer`` is taken from the checkpoint.

  True  — the pretrained head is loaded and its output is VELOCITY, matching how
          the checkpoint was trained. This is the setting that yields images.
  False — ``final_layer`` keeps `QuadtreeFiT`'s zero-init, so the model outputs
          identically zero and the sampler returns the pure noise it started
          from. Useful only to confirm the head is in fact what carries the
          prediction; it cannot produce images.

Note the training script (`train_quadtree.py`) instead trains a CLEAN-X1 head
with a 1/(1-t)^2 weight. A checkpoint from that run is not velocity and will not
generate under ``LOAD_HEAD = True``; this script targets the pretrained FiT.

Usage
-----
Edit the CONFIG block below, then::

    python test_quadtree_generate.py
"""

import os

# ---- GPU selection (must run BEFORE torch is imported) ----------------------
# Which physical GPU to use, as a string index ("0", "3", ...).
#
# Leave as None under a scheduler. SLURM (and friends) already pin the job to
# its allocated GPUs via CUDA_VISIBLE_DEVICES; overwriting that would move the
# run onto a device the job was never granted — which either fails outright or,
# worse, quietly contends with another user's job. Setting it only when unset
# means an explicit `CUDA_VISIBLE_DEVICES=2 python test_quadtree_generate.py`
# also wins over this file.
GPU_ID = None      # e.g. "3" to force a device on an unmanaged box

if GPU_ID is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import math

import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL

from fit.model.quadtree_model import QuadtreeFiT
from fit.quadtree_compression.quadtree_compression import (
    plan_from_variance, plan_to_masks, REGION_SIZES)
from fit.quadtree_compression.predictive_compressor import (
    PredictiveVarianceCompressor)
from fit.utils.utils import patchify, unpatchify


# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Pretrained FiT checkpoint (.safetensors). Either a plain file or an accelerate
# checkpoint directory, in which case CKPT_FILE picks the file inside it.
CKPT_PATH = "/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors"
CKPT_FILE = "model_1.safetensors"      # used only when CKPT_PATH is a directory

# Where the PNGs go.
OUTPUT_DIR = "./quadtree_ckpt_test"

# "patchify" | "encoder" | "both"  — see the module docstring.
MODE = "both"

# Load the checkpoint's final_layer (velocity head). False leaves it zero-init,
# which makes the model output exactly zero — no images by construction.
LOAD_HEAD = True

# Quadtree threshold. 0.0 = NO COMPRESSION (all size-1 leaves). This is what
# makes the run comparable to the pretrained checkpoint; raising it starts
# collapsing leaves and the pretrained weights no longer apply.
QUADTREE_THRESHOLD = 0.0

# Keep the learned per-leaf-size embedding active. It is zero-init, so at zero
# compression it contributes nothing — leaving it on proves that.
USE_SIZE_COND = True

N_IMAGES = 8
BATCH_SIZE = 8
NUM_STEPS = 50
CFG_SCALE = 4.0
NUM_CLASSES = 1000

# Fixed, recognisable ImageNet classes; easier to eyeball than random labels.
# 207 golden retriever, 88 macaw, 980 volcano, 933 cheeseburger,
# 417 balloon, 279 arctic fox, 973 coral reef, 291 lion.
CLASS_LABELS = [207, 88, 980, 933, 417, 279, 973, 291]

VAE_PATH = "stabilityai/sd-vae-ft-ema"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LATENT_SIZE = 32     # latent H == W for 256x256 images
PATCH_SIZE = 2
C_IN = 4
HIDDEN_SIZE = 1152
PAD_TO_MULTIPLE = 128

# Trunk config — must match the pretrained FiT XL exactly (see configs/fitv2).
_MODEL_CFG = dict(
    context_size=256,
    patch_size=PATCH_SIZE,
    in_channels=C_IN,
    hidden_size=HIDDEN_SIZE,
    depth=36,
    num_heads=16,
    mlp_ratio=4.0,
    class_dropout_prob=0.1,
    num_classes=NUM_CLASSES,
    use_swiglu=True,
    use_swiglu_large=False,
    q_norm="layernorm",
    k_norm="layernorm",
    qk_norm_weight=False,
    rel_pos_embed="rope",
    online_rope=True,
    max_pe_len_h=16,
    max_pe_len_w=16,
    decouple=True,
    ori_max_pe_len=16,
    adaln_type="lora",
    adaln_lora_dim=288,
)

# ─────────────────────────────────────────────────────────────────────────────


def _gpu_info() -> str:
    """Name the GPU actually in use, so a mis-set device is visible immediately.

    Indices here are *post*-CUDA_VISIBLE_DEVICES, so torch's device 0 is the
    first visible GPU, not necessarily physical GPU 0. Both are printed.
    """
    if not torch.cuda.is_available():
        return ""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    idx = torch.cuda.current_device()
    return (f":{idx} ({torch.cuda.get_device_name(idx)}), "
            f"CUDA_VISIBLE_DEVICES={visible}")


def resolve_ckpt(path: str) -> str:
    """Accept either a .safetensors file or an accelerate checkpoint directory."""
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, CKPT_FILE)
    if os.path.isfile(cand):
        return cand
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {path}")
    return os.path.join(path, files[0])


def build_uncompressed_plan(device):
    """The degenerate quadtree: every leaf at size 1, i.e. no compression at all.

    Driving `plan_from_variance` with an all-ones variance grid and threshold 0
    means no region ever tests flat, so the recursion splits all the way down.
    We go through the real planner rather than hand-building the plan so this
    test exercises the same code path training does.

    Returns the plan dict plus `order`, which maps a row-major (h*W + w) token
    sequence into the plan's recursion order.
    """
    var = [torch.ones(LATENT_SIZE // n, LATENT_SIZE // n, device=device)
           for n in REGION_SIZES]
    levels, positions, sizes = plan_from_variance(
        var, QUADTREE_THRESHOLD, LATENT_SIZE)

    g = LATENT_SIZE // PATCH_SIZE                       # 16
    n_tok = int(sizes.shape[0])
    assert n_tok == g * g, f"expected {g*g} tokens, got {n_tok}"
    assert bool((sizes == 1).all()), "threshold must yield size-1 leaves only"

    # Token k of the plan covers the 2x2 latent patch whose top-left is at
    # (cy - 1, cx - 1); its row-major index on the 16x16 patch grid is iy*g + ix.
    iy = ((positions[:, 0] - 1) / 2).round().long()
    ix = ((positions[:, 1] - 1) / 2).round().long()
    order = (iy * g + ix)                               # recursion pos -> row-major

    plan = dict(levels=levels, positions=positions, sizes=sizes)
    return plan, order, g


def check_premises(plan, order, g, device):
    """Assert the invariants this test leans on, before spending a GPU on it.

    Each of these is a claim the script's reasoning depends on. Checking them
    here means a violated premise reports itself, instead of quietly showing up
    as bad images that get blamed on the trunk.
    """
    # (1) The token reordering is a true permutation, so patchify -> reorder ->
    #     inverse -> unpatchify is the identity. If this failed, images would be
    #     spatially scrambled and the test would report a false failure.
    x = torch.randn(2, C_IN, LATENT_SIZE, LATENT_SIZE, device=device)
    tok = patchify(x, PATCH_SIZE)[:, order]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.shape[0], device=order.device)
    assert torch.equal(unpatchify(tok[:, inv], (LATENT_SIZE, LATENT_SIZE),
                                  PATCH_SIZE), x), "token reorder is not invertible"

    # (2) Our hand-built packing is byte-identical to what the real compressor
    #     emits, so "the test packs differently from training" is ruled out.
    comp = PredictiveVarianceCompressor(
        latent_channels=C_IN, c=8, d=HIDDEN_SIZE, crop=LATENT_SIZE,
        pad_to_multiple=PAD_TO_MULTIPLE, with_decoder=False,
        share_weights=True).to(device).eval()
    n = 3
    with torch.no_grad():
        ref = comp(torch.randn(n, C_IN, LATENT_SIZE, LATENT_SIZE, device=device),
                   [plan] * n, torch.zeros(n, dtype=torch.long, device=device),
                   torch.zeros(n, device=device))
    kw, N, _ = pack_inputs(plan, order, g, n, device, torch.float32)
    assert N == ref['feature'].shape[1]
    for key, got, want in (("grid", kw['grid'], ref['grid'].float()),
                           ("tsize", kw['tsize'], ref['tsize']),
                           ("mask", kw['mask'], ref['mask'].float()),
                           ("doc_ids", kw['doc_ids'], ref['doc_ids'].long())):
        assert torch.equal(got, want), f"packed '{key}' differs from compressor"

    print("Premises   : token order invertible, packing matches compressor  [ok]")


def pack_inputs(plan, order, g, n_pack, device, dtype):
    """Build the packed model_kwargs for `n_pack` images sharing one plan.

    Mirrors `PredictiveVarianceCompressor.forward`: B=1 sequence, per-token
    doc_ids, grid in patch units (centers), zero padding to PAD_TO_MULTIPLE.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    n_tok = int(plan['sizes'].shape[0])
    raw_len = n_pack * n_tok
    N = max(int(math.ceil(raw_len / PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE),
            PAD_TO_MULTIPLE)

    grid = torch.zeros(1, 2, N, device=device, dtype=dtype)
    tsize = torch.zeros(1, N, dtype=torch.long, device=device)
    mask = torch.zeros(1, N, device=device, dtype=dtype)
    doc = torch.full((1, N), -1, dtype=torch.int32, device=device)

    # Same convention as the compressor: patch centers in latent px, halved to
    # patch units, so neighbouring size-1 tokens sit 1 apart.
    g_one = plan['positions'].to(device).transpose(0, 1) / 2.0   # (2, n_tok)
    for i in range(n_pack):
        sl = slice(i * n_tok, (i + 1) * n_tok)
        grid[0, :, sl] = g_one.to(dtype)
        tsize[0, sl] = plan['sizes'].to(device)
        mask[0, sl] = 1
        doc[0, sl] = i

    # size is the per-image full latent grid size, used only to scale RoPE.
    size = torch.full((1, n_pack, 2), float(LATENT_SIZE // PATCH_SIZE),
                      device=device, dtype=torch.int32)

    _doc = doc
    def doc_mask_mod(b, h, q_idx, kv_idx):
        return _doc[b, q_idx] == _doc[b, kv_idx]
    block_mask = create_block_mask(doc_mask_mod, 1, None, N, N, device=device)

    return dict(grid=grid, mask=mask, size=size, tsize=tsize,
                doc_ids=doc.long(), block_mask=block_mask), N, raw_len


def tokens_from_patchify(x_sp, order, N, dtype):
    """Raw 2x2 latent patches, reordered into quadtree recursion order.

    `patchify` emits row-major tokens; `order[k]` is the row-major index of the
    plan's k-th token, so indexing with it puts them in the order the packed
    sequence (and hence `grid`/`doc_ids`) expects. Getting this wrong is exactly
    the kind of bug the test is built to catch — the images would scramble.
    """
    tok = patchify(x_sp, PATCH_SIZE)                    # (n_pack, n_tok, 16)
    tok = tok[:, order]                                 # -> recursion order
    n_pack, n_tok, D = tok.shape
    out = tok.new_zeros(1, N, D)
    out[0, :n_pack * n_tok] = tok.reshape(n_pack * n_tok, D)
    return out.to(dtype)


def tokens_from_encoder(compressor, x_sp, plan, N, dtype):
    """Level-0 tokens straight out of the (untrained) PyramidEncoder.

    Uses the compressor's own batched path so this reflects the real training
    pipeline, not a reimplementation of it.
    """
    plans = [plan] * x_sp.shape[0]
    tokens, counts = compressor.encode_batch(x_sp, plans)
    out = tokens.new_zeros(1, N, tokens.shape[-1])
    out[0, :sum(counts)] = tokens
    return out.to(dtype)


@torch.no_grad()
def sample_batch(model, compressor, plan, order, g, labels, mode, device, dtype):
    """Euler flow-matching sampler with CFG, integrating t: 0 (noise) -> 1 (clean).

    The pretrained head predicts velocity, so the update is the plain
    x <- x + v*dt. Conditional and unconditional passes are packed into ONE
    sequence (2*n_pack documents), which also exercises multi-document packing.
    """
    n_pack = labels.shape[0]
    kwargs, N, _ = pack_inputs(plan, order, g, 2 * n_pack, device, dtype)

    y_null = torch.full_like(labels, NUM_CLASSES)
    y = torch.cat([labels, y_null], 0).to(torch.int).view(1, 2 * n_pack)

    x = torch.randn(n_pack, C_IN, LATENT_SIZE, LATENT_SIZE,
                    device=device, dtype=dtype)
    ts = torch.linspace(0.0, 1.0, NUM_STEPS + 1, device=device, dtype=dtype)

    for i in range(NUM_STEPS):
        t, dt = ts[i], ts[i + 1] - ts[i]
        x2 = torch.cat([x, x], 0)                       # cond + uncond

        if mode == "patchify":
            feat = tokens_from_patchify(x2, order, N, dtype)
        else:
            feat = tokens_from_encoder(compressor, x2, plan, N, dtype)

        t_pack = t.repeat(1, 2 * n_pack)
        out = model(feat, t_pack, y, **kwargs)          # (1, N, ...)

        n_tok = int(plan['sizes'].shape[0])
        v_tok = out[0, :2 * n_pack * n_tok].reshape(2 * n_pack, n_tok, -1)
        # back to row-major before unpatchify
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.shape[0], device=order.device)
        v_tok = v_tok[:, inv]
        v = unpatchify(v_tok.float(), (LATENT_SIZE, LATENT_SIZE), PATCH_SIZE)

        v_c, v_u = v.chunk(2, dim=0)
        x = x + (v_u + CFG_SCALE * (v_c - v_u)).to(dtype) * dt

    return x.float()


def decode_and_save(vae, x, paths):
    with torch.no_grad():
        imgs = vae.decode(x / vae.config.scaling_factor).sample
    imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
    imgs = imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    for arr, p in zip(imgs, paths):
        Image.fromarray(arr).save(p)


def save_side_by_side(a_dir, b_dir, out_dir, n):
    """Reference (patchify) left, encoder right, one file per image."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        a = Image.open(os.path.join(a_dir, f"{i:03d}.png"))
        b = Image.open(os.path.join(b_dir, f"{i:03d}.png"))
        canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        canvas.save(os.path.join(out_dir, f"{i:03d}.png"))


def load_model(ckpt_path, token_input_dim, device):
    """Instantiate QuadtreeFiT and load the pretrained FiT weights into it.

    Reports missing/unexpected keys explicitly: they are the first place a
    structural mismatch between the two models shows up.
    """
    cfg = dict(_MODEL_CFG, token_input_dim=token_input_dim,
               use_size_cond=USE_SIZE_COND, use_pyramid_decoder=False)
    model = QuadtreeFiT(**cfg)

    state = load_file(ckpt_path, device="cpu")
    if not LOAD_HEAD:
        state = {k: v for k, v in state.items() if not k.startswith("final_layer.")}

    missing, unexpected = model.load_state_dict(state, strict=False)
    # size_embedder is new in QuadtreeFiT and intentionally absent from a FiT
    # checkpoint; it is zero-init, so at zero compression it is a no-op.
    expected_missing = {"size_embedder.weight"}
    real_missing = [k for k in missing if k not in expected_missing]
    print(f"  loaded {len(state)} tensors")
    if real_missing:
        print(f"  [warn] {len(real_missing)} missing keys "
              f"(first 5: {real_missing[:5]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys "
              f"(first 5: {unexpected[:5]})")
    if not real_missing and not unexpected:
        print("  key match: exact")
    return model.to(device).eval()


def run_mode(mode, ckpt_path, plan, order, g, vae, labels):
    """Generate N_IMAGES for one token-source mode; returns its output dir."""
    print(f"\n{'='*64}\nMODE = {mode}")
    out_dir = os.path.join(OUTPUT_DIR, mode)
    os.makedirs(out_dir, exist_ok=True)

    # patchify -> pretrained PatchEmbedder (16 -> 1152).
    # encoder  -> Identity, tokens already at hidden_size.
    token_input_dim = C_IN * PATCH_SIZE ** 2 if mode == "patchify" else HIDDEN_SIZE
    model = load_model(ckpt_path, token_input_dim, DEVICE)
    dtype = next(model.parameters()).dtype

    compressor = None
    if mode == "encoder":
        compressor = PredictiveVarianceCompressor(
            latent_channels=C_IN, c=256, d=HIDDEN_SIZE, crop=LATENT_SIZE,
            pad_to_multiple=PAD_TO_MULTIPLE, with_decoder=False,
            share_weights=True).to(DEVICE).eval()
        print("  [note] PyramidEncoder is randomly initialised — expect noise.")

    if not LOAD_HEAD:
        print("  [note] LOAD_HEAD=False: zero-init head outputs 0; "
              "the sampler will return its initial noise.")

    done = 0
    while done < N_IMAGES:
        bs = min(BATCH_SIZE, N_IMAGES - done)
        torch.manual_seed(SEED + done)
        y = labels[done:done + bs]
        x = sample_batch(model, compressor, plan, order, g, y, mode, DEVICE, dtype)
        decode_and_save(vae, x, [os.path.join(out_dir, f"{done+i:03d}.png")
                                 for i in range(bs)])
        done += bs
        print(f"    {done}/{N_IMAGES}", end="\r", flush=True)
    print(f"    {N_IMAGES}/{N_IMAGES}  ->  {out_dir}")

    del model, compressor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    ckpt_path = resolve_ckpt(CKPT_PATH)
    print(f"Device     : {DEVICE}{_gpu_info()}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Head       : {'pretrained (velocity)' if LOAD_HEAD else 'zero-init'}")

    plan, order, g = build_uncompressed_plan(DEVICE)
    print(f"Quadtree   : threshold={QUADTREE_THRESHOLD} -> "
          f"{int(plan['sizes'].shape[0])} tokens, all size-1 "
          f"(= uncompressed {g}x{g} grid)")
    check_premises(plan, order, g, DEVICE)

    labels = torch.tensor(
        (CLASS_LABELS * (N_IMAGES // len(CLASS_LABELS) + 1))[:N_IMAGES],
        device=DEVICE)

    print(f"\nLoading VAE from {VAE_PATH} …")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE).eval()

    modes = ["patchify", "encoder"] if MODE == "both" else [MODE]
    dirs = {m: run_mode(m, ckpt_path, plan, order, g, vae, labels) for m in modes}

    if MODE == "both":
        sbs = os.path.join(OUTPUT_DIR, "side_by_side")
        save_side_by_side(dirs["patchify"], dirs["encoder"], sbs, N_IMAGES)
        print(f"\nSide-by-side (patchify | encoder) -> {sbs}")

    print(f"\nDone. Inspect {OUTPUT_DIR}.")
    print("PASS  = 'patchify' images are recognisable objects.")
    print("FAIL  = 'patchify' images are noise -> QuadtreeFiT diverges from the "
          "checkpoint (trunk, RoPE grid, conditioning, packing or token order).")


if __name__ == "__main__":
    main()
