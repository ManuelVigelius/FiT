"""CPU tests for global quadtree-structure conditioning (struct_cond_dim).

    CUDA_VISIBLE_DEVICES= python test_struct_cond.py [--dataset DIR]

Checks the three places the structure vector must come from the FULL tree, and
the model wiring:

  1. dataset  -- `struct` ignores max_res_schedule: at t >= 0.95 the plan is
                 pruned to 4 size-8 leaves, but `struct` is the uncapped tree's
                 vector, identical to the one at small t. The packer stacks it.
  2. model    -- zero-init: a struct-conditioned model with copied trunk weights
                 matches the unconditioned one exactly; once the MLP output layer
                 is non-zero the struct changes the output; it gets gradient;
                 a missing struct is an error.
  3. sampler  -- sample_batch hands the model the SAME full-tree vector at every
                 step (both CFG branches), while the per-step plans get pruned.

Steps (2) and (3) need no data; (1) and (3) read a few real latents.
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

import fit.data.in1k_gt_quadtree_latent_dataset as D
import quadtree_prior.structure as S
from fit.model.quadtree_model import QuadtreeFiT

DATASET = "/visinf/projects_students/mb_mvigel/datasets/greater_than_256_crop"
THRESHOLD = 0.5
SCHEDULE = [[0.0, 1], [0.7, 2], [0.85, 4], [0.95, 8]]


def _some_files(root, n):
    out = []
    with os.scandir(root) as it:              # lazy: the dir holds ~1M files
        for e in it:
            if e.name.endswith(".safetensors"):
                out.append(e.path)
                if len(out) == n:
                    break
    return sorted(out)


def _full_struct(x0):
    """Reference: the uncapped GT tree's vector, computed independently."""
    _, pos, sizes = D.plan_from_gt_variance(
        D.gt_variance_grids(x0), THRESHOLD, 32, min_leaf_size=1)
    return S.structure_vector(S.sizes_grid_from_plan(
        dict(positions=pos, sizes=sizes)))


# ──────────────────────────────── 1. dataset ────────────────────────────────
def test_dataset(files):
    ds = D.GTQuadtreeLatentDataset.__new__(D.GTQuadtreeLatentDataset)
    ds.root_dir, ds.files, ds.crop = None, files, 32
    ds.snr_type, ds.t0, ds.t1 = "lognorm", 0.0, 1.0
    ds.threshold, ds._base_seed = THRESHOLD, 0
    ds.max_res_schedule = D.normalize_max_res_schedule(SCHEDULE)

    orig = D._sample_t
    try:
        items = {}
        for t in (0.3, 0.97):
            D._sample_t = lambda *a, _t=t, **k: torch.tensor(_t)
            items[t] = [ds[i] for i in range(len(files))]
    finally:
        D._sample_t = orig

    n_split = 0
    for lo, hi in zip(items[0.3], items[0.97]):
        assert hi["plan"]["sizes"].tolist() == [8, 8, 8, 8], "t=0.97 not pruned"
        assert lo["struct"].shape == (S.STRUCT_DIM,)
        assert torch.equal(lo["struct"], hi["struct"]), "struct depends on t"
        assert torch.equal(hi["struct"], _full_struct(hi["x0"]))
        n_split += int(hi["struct"].any())
    # The pruned t=0.97 tree has NO splits; a vector built from it would be all
    # zero. Real images do split, so a non-zero vector proves it is the full one.
    assert n_split > 0, "every struct is zero -- looks like the pruned tree"

    class _Loader:                            # one raw batch, then exhausted
        def __iter__(self):
            return iter([items[0.3]])
    batch = next(D.GTQuadtreePacker(_Loader(), max_seq_len=10_000,
                                    device="cpu"))
    assert batch["struct"].shape == (len(files), S.STRUCT_DIM)
    assert torch.equal(batch["struct"][0], items[0.3][0]["struct"])
    print(f"[dataset] ok: {len(files)} images, {n_split} with splits; struct "
          "identical at t=0.3 and t=0.97 (pruned plan), packer stacks it")


# ──────────────────────────────── 2. model ──────────────────────────────────
_TINY = dict(context_size=256, patch_size=2, in_channels=4, hidden_size=64,
             depth=2, num_heads=4, mlp_ratio=2.0, num_classes=10,
             use_swiglu=True, q_norm="layernorm", k_norm="layernorm",
             rel_pos_embed="rope", online_rope=True, max_pe_len_h=16,
             max_pe_len_w=16, decouple=True, ori_max_pe_len=16,
             adaln_type="lora", adaln_lora_dim=16, use_size_cond=True,
             token_input_dim=64, use_pyramid_decoder=True)


def _tiny_inputs(P=3, n_per=(4, 7, 16)):
    N = sum(n_per)
    g = torch.Generator().manual_seed(0)
    doc = torch.repeat_interleave(torch.arange(P), torch.tensor(n_per))
    return dict(
        x=torch.randn(1, N, 64, generator=g),
        t=torch.rand(1, P, generator=g),
        y=torch.randint(0, 10, (1, P), generator=g),
        grid=torch.rand(1, 2, N, generator=g) * 16,
        mask=torch.ones(1, N),
        size=torch.full((1, P, 2), 16.0),
        tsize=torch.tensor([8, 4, 2, 1])[torch.randint(0, 4, (1, N), generator=g)],
        doc_ids=doc[None],
        struct=(torch.rand(1, P, S.STRUCT_DIM, generator=g) < 0.5).float(),
    )


def test_model():
    torch.manual_seed(0)
    m0 = QuadtreeFiT(**_TINY).eval()
    # A fresh model zero-inits every adaLN output, so nothing reads c yet and
    # no conditioning could get gradient. Perturb all weights to stand in for
    # the pretrained trunk the real run starts from.
    with torch.no_grad():
        for p in m0.parameters():
            p.add_(0.02 * torch.randn_like(p))
    m1 = QuadtreeFiT(**_TINY, struct_cond_dim=S.STRUCT_DIM).eval()
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("struct_embedder.")
                                  for k in missing), (missing, unexpected)
    # load_state_dict leaves m1's own (zero-init) struct_embedder untouched.
    assert not m1.struct_embedder[-1].weight.any()

    inp = _tiny_inputs()
    struct = inp.pop("struct")
    x = inp.pop("x")
    with torch.no_grad():
        out0 = m0(x, **inp)
        out1 = m1(x, struct=struct, **inp)
    assert torch.equal(out0, out1), "zero-init struct MLP is not a no-op"

    try:
        m1(x, **inp)
        raise AssertionError("missing struct was accepted")
    except ValueError:
        pass

    # Gradient reaches the (zero) output layer, so training can open it.
    m1.train()
    m1(x, struct=struct, **inp).sum().backward()
    assert m1.struct_embedder[-1].weight.grad.abs().sum() > 0

    # Once non-zero, different structures give different outputs.
    m1.eval()
    with torch.no_grad():
        torch.nn.init.normal_(m1.struct_embedder[-1].weight, std=0.02)
        a = m1(x, struct=struct, **inp)
        b = m1(x, struct=1.0 - struct, **inp)
    assert not torch.allclose(a, b), "struct has no effect on the output"
    print("[model]   ok: zero-init no-op, gradient flows, struct changes "
          "output once trained, missing struct rejected")


# ──────────────────────────────── 3. sampler ────────────────────────────────
def test_sampler(files):
    import generate_quadtree_images as G

    lat = torch.stack([G._center_crop(G.load_latent(f)).float()
                       for f in files[:3]])
    expected = torch.stack([_full_struct(x) for x in lat])
    assert expected.any()

    seen, tokens_per_step = [], []

    def fake_pack(comp, x_t, plans, labels, t_ds, device):
        counts = [int(p["sizes"].shape[0]) for p in plans]
        N, B2 = sum(counts), len(plans)
        tokens_per_step.append(counts[: B2 // 2])
        return dict(
            label=labels.reshape(1, B2), grid=torch.zeros(1, 2, N),
            mask=torch.ones(1, N), size=torch.full((1, B2, 2), 16.0),
            tsize=torch.cat([p["sizes"] for p in plans])[None],
            doc_ids=torch.zeros(1, N, dtype=torch.long), block_mask=None,
            feature=torch.zeros(1, N, 8), t=t_ds.reshape(1, B2),
            counts=counts)

    def fake_model(feature, t, struct=None, **kw):
        seen.append(struct.clone())
        return feature

    def fake_comp(tokens, plans, counts, x, mode):
        return torch.zeros_like(x)

    orig = G.pack
    G.pack = fake_pack
    try:
        for cond in ("gt_struct", "prior_struct"):
            seen.clear()
            tokens_per_step.clear()
            if cond == "gt_struct":
                kw = dict(gt_latents=lat)
                exp = expected
            else:
                # A full tree from the "prior": reuse the uncapped GT trees.
                full = G.plans_from_gt(lat, 1)
                kw = dict(prior_plans=full)
                exp = expected
            G.sample_batch(cond, fake_model, fake_comp,
                           torch.tensor([1, 2, 3]), num_steps=8,
                           device="cpu", noise=torch.randn(3, 4, 32, 32), **kw)
            assert len(seen) == 8
            want = torch.cat([exp, exp], 0)[None]      # both CFG branches
            for s in seen:
                assert torch.equal(s, want), f"{cond}: struct is not the full tree"
            # ...while the plans really were pruned at high t.
            assert tokens_per_step[0] == [4, 4, 4], tokens_per_step[0]
            assert tokens_per_step[-1] != tokens_per_step[0]
            print(f"[sampler] ok: {cond} gets the full-tree struct at all 8 "
                  f"steps; tokens/image {tokens_per_step[0]} -> "
                  f"{tokens_per_step[-1]}")
    finally:
        G.pack = orig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()
    assert not torch.cuda.is_available() or \
        os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
        "run with CUDA_VISIBLE_DEVICES= (CPU only)"
    files = _some_files(args.dataset, 16)
    test_model()
    test_dataset(files)
    test_sampler(files)
    print("all struct-conditioning tests passed")
