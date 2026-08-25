# quadtree_prior

Class-conditional autoregressive prior over **quadtree structures**.

The GT-variance path (`fit/data/in1k_gt_quadtree_latent_dataset.py`) decides each
image's tree from the clean latent `x0`. That is an oracle: at sampling time
there is no `x0`. This package trains a small transformer to generate a plausible
tree **from the class label alone**, run once before image generation, whose
output feeds the quadtree compressor in place of the oracle plan.

## The 17-token encoding

The compressor patchifies 2x2 neighbouring leaves into one token, so all four
leaves of a patch share a size and the tree lives on the **16x16 patch grid**. A
leaf of side `N` occupies `N x N` patch cells, so the grid is exactly a 2x2
arrangement of size-8 patches and only two levels of decisions exist:

| position | predicts | classes |
|---|---|---|
| `0` (root) | which of the four 8x8 quadrants are **not** compressed at 8x8 | 16 (4-bit mask) |
| `1..16` | per 4x4 region: stop at size 4, or a 4-bit mask over its 2x2 quadrants | 17 (16 masks + `STOP_4`) |

Bit set at the root means "descend"; bit set in a region means "this quadrant
goes lossless (size 1)", bit clear means "one size-2 patch". Bit order is
row-major over the 2x2.

A 4x4 region inside an already-compressed 8x8 quadrant is fully determined: it is
**neither predicted nor supervised** (`IGNORE_INDEX` in the targets), and its
input slot gets the dedicated `COVERED` embedding so the model sees that it was
skipped. `sample.py` reproduces that feeding rule exactly, so there is no
train/inference input-distribution gap.

Token counts range from 4 (all size-8) to 256 (all lossless).

## Conditioning

The class label enters **only through adaptive layer norm** (adaLN-Zero, as in
`fit/model/modules.py:FiTBlock`) — no conditioning tokens occupy sequence
positions, and classifier-free guidance is a single vector swap.

There is **no timestep**: the prior runs once before the diffusion process
starts, so no `t` exists to condition on, and the targets are the uncapped
GT-variance tree of the clean latent (no `max_res_schedule` — that schedule
belongs to the compressor's path, where it caps resolution as a function of `t`).

Two output heads keep the root's 16-way alphabet separate from the regions'
17-way one.

## Vocabularies

The two mask alphabets are the **same 16 symbols** — a mask means "these
quadrants split" at whatever level the position is, and the learned positional
embedding already tells the model which level that is.

*Output*, per head (they differ, which is why there are two):

| head | classes |
|---|---|
| `root_head` (position 0) | **16** — masks `0..15` |
| `region_head` (positions 1..16) | **17** — masks `0..15` + `STOP_4` |

`STOP_4` is masked to `-inf` at position 0, so it is structurally unreachable
there rather than something the root must learn to avoid.

*Input* embedding table: `INPUT_VOCAB = 19` — the 16 shared masks + `STOP_4` +
`BOS` + `COVERED`.

## Files

| file | role |
|---|---|
| `structure.py` | the encoding, and exact conversion to/from the compressor's `levels/positions/sizes` plan layout. Run it directly for its smoke test. |
| `dataset.py` | GT trees from clean latents, tokenized. Reuses the oracle planner, so targets match what the compressor is trained against. Deterministic — no RNG. |
| `model.py` | `QuadtreePrior` — decoder-only transformer + `structure_loss`. |
| `sample.py` | batched ancestral sampling, with temperature / top-k / CFG. |
| `train.py` | Accelerate training entry point. |

## Training

```bash
accelerate launch -m quadtree_prior.train \
    --project_name quadtree_prior_s \
    --cfgdir configs/quadtree_prior/config_prior_s.yaml
```

`quadtree_threshold` in the config **must match the compressor's config**
(`configs/quadtree/config_quadtree_xl_gt_variance.yaml`) — otherwise the prior
learns a distribution of trees the compressor never saw. It is the only knob on
the target trees; an uncapped GT tree can reach 256 tokens for a busy image, so
lower it if generated trees need to be smaller.

Logged metrics: `root_acc`, `region_acc`, and `exact` (fraction of sequences
where *every* decision is right — the one that matters, since a tree is only
usable if fully correct).

## Sampling

```python
from quadtree_prior.model import QuadtreePrior
from quadtree_prior.sample import sample_plans
from fit.utils.utils import init_from_ckpt

prior = QuadtreePrior(hidden_size=384, depth=6, num_heads=6)
init_from_ckpt(prior, "workdir/quadtree_prior_s/checkpoints/checkpoint-50000/model.safetensors")
prior.eval()

plans, n_tokens = sample_plans(prior, labels, top_k=8, cfg_scale=1.5)
packed = compressor(x_t, plans, labels, t)     # drop-in for oracle plans
```

`plans` are the same dicts `plan_from_gt_variance` produces — verified to match
the oracle planner tensor-for-tensor, including emission order and positions —
so they are a drop-in. Use `n_tokens` to batch images against a token budget.
