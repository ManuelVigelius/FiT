"""Gradient-flow test for the residual gates on the quadtree compressor.

The question this answers
------------------------
Both pyramid paths are residual and zero-gated:

    encode   token = base_proj(mean-pooled patch) + res_gamma * pyramid(...)
    decode   x_hat = per-token readout          + res_gamma * pyramid(...)

with ``res_gamma`` zero-init, so at step 0 the compressor is exactly the
mean-pooled baseline and the pyramid only earns its way in. The consequence,
which is easy to miss, is that the pyramid's own weights receive gradient
*scaled by the gate*: while ``res_gamma == 0`` they are gradient-starved by
construction.

That is fine ONLY because the gate itself is not starved —
``dL/dres_gamma = dL/dt . t`` does not contain a factor of gamma, so the gate
moves on the very first optimizer step and opens the path. The thing actually
worth testing is therefore the *conditional* claim, not the dynamics:

    gamma == 0  ->  pyramid weights get NO gradient   (starved, by design)
    gamma != 0  ->  pyramid weights DO get gradient   (the path is live)

So this sets the gates directly rather than training to reach a non-zero value.
If the second arm ever fails, a warmup would spend its whole budget training
nothing while the loss still fell — the base path alone keeps improving, so the
failure is invisible on the loss curve. That is what makes it worth a test.

Why the leaf-size distribution matters
--------------------------------------
A pyramid level only receives gradient if the tree actually USES that level.
Random latents have high variance everywhere, so the planner never merges and
emits an all-size-1 tree, leaving every coarse level unused and apparently
"broken". The test therefore runs three regimes — detailed, smooth, and mixed —
and asserts against what each tree can support.

Run directly:  python test_res_gamma_gradient.py
"""

import collections

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from fit.utils.utils import instantiate_from_config
from fit.data.in1k_gt_quadtree_latent_dataset import (
    plan_from_gt_variance, gt_variance_grids,
)

CONFIG = 'configs/quadtree/config_quadtree_xl_learned.yaml'
CROP, LATENT_CH, BATCH = 32, 4, 2
QUADTREE_THRESHOLD = 0.5


def resolve_tuple(*args):
    return tuple(args)


if not OmegaConf.has_resolver('tuple'):
    OmegaConf.register_new_resolver('tuple', resolve_tuple)


def build_compressor():
    """The real compressor from the real config, only narrowed to run on CPU.

    c/d are shrunk and the checkpoint seeding dropped; the wiring under test --
    residual encoder, residual decoder, shared Merge/Split -- is untouched.
    """
    cfg = OmegaConf.load(CONFIG)
    p = cfg.diffusion.compressor_config.params
    p.c, p.d, p.base_proj_ckpt = 16, 128, None
    return instantiate_from_config(cfg.diffusion.compressor_config)


def make_plans(x0):
    """Oracle quadtree plans, the same call the GT loader makes."""
    plans = []
    for i in range(x0.shape[0]):
        levels, positions, sizes = plan_from_gt_variance(
            gt_variance_grids(x0[i]), QUADTREE_THRESHOLD, CROP)
        plans.append(dict(levels=levels, positions=positions, sizes=sizes))
    return plans


def _grad_mag(param):
    """max|grad|, with a detached/unused parameter (grad is None) read as 0."""
    return 0.0 if param.grad is None else param.grad.abs().max().item()


def gated_params(compressor):
    """Pyramid weights that sit BEHIND a gate.

    Excludes the gates themselves, and the base path (base_proj / base_head),
    which is added ungated and so always carries gradient.
    """
    return {n: p for n, p in compressor.named_parameters()
            if not n.endswith('res_gamma')
            and not n.startswith('encoder.base_proj')
            and not n.startswith('decoder.base_head')}


def backward_once(compressor, x0, x_t, plans, gamma):
    """Set both gates to `gamma`, run compress -> decode -> full-res loss."""
    with torch.no_grad():
        compressor.encoder.res_gamma.fill_(gamma)
        compressor.decoder.res_gamma.fill_(gamma)
    compressor.zero_grad(set_to_none=True)

    labels = torch.zeros(x0.shape[0], dtype=torch.long)
    t = torch.rand(x0.shape[0])
    packed = compressor(x_t, plans, labels, t, x0=x0)
    # Stand-in for the transformer: hand the tokens straight back to the
    # decoder. Same graph shape as Transport.loss_quadtree_dense, without
    # needing to build and run the full model.
    x_hat = compressor.decode_packed(
        packed['feature'], plans, packed['counts'], x_t)
    loss = F.mse_loss(x_hat, x0)
    loss.backward()

    live = {n: p.grad.abs().max().item() for n, p in gated_params(compressor).items()
            if p.grad is not None and p.grad.abs().max() > 0}
    return loss.item(), live


def latents(kind):
    """Three content regimes, chosen for the trees they produce."""
    if kind == 'detailed':
        # High variance everywhere -> nothing merges -> all size-1 leaves.
        x0 = torch.randn(BATCH, LATENT_CH, CROP, CROP)
    elif kind == 'smooth':
        # Low variance everywhere -> maximal merging -> coarse leaves.
        x0 = F.interpolate(torch.randn(BATCH, LATENT_CH, 4, 4),
                           size=(CROP, CROP), mode='bilinear')
    elif kind == 'mixed':
        # Smooth on the left, noisy on the right -> a genuinely multi-level tree.
        x0 = F.interpolate(torch.randn(BATCH, LATENT_CH, 4, 4),
                           size=(CROP, CROP), mode='bilinear')
        x0[:, :, :, CROP // 2:] += 3.0 * torch.randn(
            BATCH, LATENT_CH, CROP, CROP // 2)
    else:
        raise ValueError(kind)
    return x0, x0 + 0.1 * torch.randn_like(x0)


def main():
    torch.manual_seed(0)
    failures = []

    for kind in ('detailed', 'smooth', 'mixed'):
        torch.manual_seed(0)
        compressor = build_compressor()
        x0, x_t = latents(kind)
        plans = make_plans(x0)
        total = len(gated_params(compressor))

        sizes = dict(collections.Counter(plans[0]['sizes'].tolist()))
        print(f'\n=== {kind}: leaf sizes {sizes} ===')

        # ---- arm 1: closed gate, nothing behind it may train ----------------
        loss0, live0 = backward_once(compressor, x0, x_t, plans, 0.0)
        # grad is None, not zero, when the path is detached outright -- which is
        # one of the very failures this test exists to catch, so treat it as a
        # magnitude of zero and report rather than crashing on the attribute.
        gate_enc = _grad_mag(compressor.encoder.res_gamma)
        gate_dec = _grad_mag(compressor.decoder.res_gamma)
        print(f'  gamma=0.0   loss={loss0:.6f}  pyramid tensors with grad: '
              f'{len(live0)}/{total}')
        print(f'              gate grad: encoder={gate_enc:.3e} '
              f'decoder={gate_dec:.3e}')

        if live0:
            failures.append(
                f'{kind}: {len(live0)} pyramid tensors got gradient at gamma=0, '
                f'expected none (e.g. {sorted(live0)[:3]})')
        # The gates must NOT be starved -- this is what lets them open at all.
        if gate_enc == 0.0 or gate_dec == 0.0:
            detached = [w for w, m in (('encoder', compressor.encoder),
                                       ('decoder', compressor.decoder))
                        if m.res_gamma.grad is None]
            how = (f'grad is None on {detached} -- the residual path is detached '
                   f'from the loss') if detached else 'grad is exactly zero'
            failures.append(
                f'{kind}: a res_gamma got no gradient at gamma=0 '
                f'(encoder={gate_enc:.3e}, decoder={gate_dec:.3e}); {how}. The '
                f'gate can never leave zero, so the pyramid can never train')

        # ---- arm 2: open gate, the path must be live ------------------------
        loss1, live1 = backward_once(compressor, x0, x_t, plans, 0.1)
        print(f'  gamma=0.1   loss={loss1:.6f}  pyramid tensors with grad: '
              f'{len(live1)}/{total}')
        if not live1:
            failures.append(
                f'{kind}: NO pyramid tensor got gradient at gamma=0.1; the '
                f'residual path is dead and a warmup would train nothing')

        dead = sorted(set(gated_params(compressor)) - set(live1))
        if dead:
            print(f'              no grad ({len(dead)}): {dead[:4]}'
                  f'{"..." if len(dead) > 4 else ""}')

    # A level only gets gradient when the tree uses it, so the strong claim --
    # essentially every gated tensor trains -- is only meaningful on a tree that
    # spans levels. `mixed` is that case; require it explicitly.
    torch.manual_seed(0)
    compressor = build_compressor()
    x0, x_t = latents('mixed')
    plans = make_plans(x0)
    _, live = backward_once(compressor, x0, x_t, plans, 0.1)
    total = len(gated_params(compressor))
    dead = sorted(set(gated_params(compressor)) - set(live))
    # mask_proj.N.weight is Conv2d(1, c, 1) over level N's occupancy mask: on a
    # level the tree leaves empty the input is all-zeros, so dL/dW vanishes while
    # the bias still trains. Expected, and it resolves over a real dataset where
    # some batch occupies every level.
    unexpected = [n for n in dead if 'mask_proj' not in n]
    print(f'\n=== multi-level coverage (mixed) ===')
    print(f'  {len(live)}/{total} gated tensors training; '
          f'{len(dead)} idle, {len(unexpected)} of them unexplained')
    if unexpected:
        failures.append(
            f'gated tensors with no gradient on a multi-level tree: {unexpected[:6]}')

    print()
    if failures:
        for f in failures:
            print(f'FAIL: {f}')
        raise SystemExit(1)
    print('PASS: residual gates receive gradient at zero, and the pyramid '
          'behind them trains once they open.')


if __name__ == '__main__':
    main()
