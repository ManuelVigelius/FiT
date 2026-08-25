"""Class-conditional autoregressive prior over quadtree structures.

At sampling time there is no x0, so the ground-truth-variance tree of
`fit.data.in1k_gt_quadtree_latent_dataset` is unavailable. This package trains a
small transformer to GENERATE a plausible tree from the class label alone; it is
run once before image generation and its output feeds the quadtree compressor in
place of an oracle plan.

    structure  -- the 17-token sequence encoding, and conversion to/from the
                  `levels/positions/sizes` plan layout the compressor consumes.
    dataset    -- GT trees from latent files, tokenized into that sequence.
    model      -- decoder-only transformer, class-conditioned via adaLN.
    train      -- training entry point.
    sample     -- batched ancestral sampling of plans.
"""
