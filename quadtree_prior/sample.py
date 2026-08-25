"""Ancestral sampling of quadtree structures from a trained prior.

Run once before image generation: given class labels, produce one plan per label
in exactly the layout the quadtree compressor consumes. There is no timestep —
the prior runs before the diffusion process starts.

Decoding is 17 sequential forward passes over the whole prefix. With 17 positions
and a small model there is no reason for a KV cache; the whole batch decodes in
well under a second.

The root's output decides which region positions carry information. Covered
positions are neither sampled nor fed back: they receive the COVERED embedding,
exactly as during training, so the model never sees an input distribution it was
not trained on.
"""

import torch
import torch.nn.functional as F

from quadtree_prior import structure as S


@torch.no_grad()
def sample_structures(model, labels, temperature=1.0, top_k=None,
                      cfg_scale=1.0, generator=None, device=None):
    """Sample one structure per label.

    model       : trained QuadtreePrior (eval mode).
    labels      : (B,) long class labels.
    temperature : softmax temperature; 0 means greedy (argmax).
    top_k       : keep only the k most likely classes per step.
    cfg_scale   : classifier-free guidance on the logits. 1.0 disables it; the
                  unconditional branch uses the label-dropout embedding, so the
                  model must have been trained with class_dropout_prob > 0.
    generator   : optional torch.Generator for reproducibility.

    Returns (tokens, sizes_grids):
        tokens      (B, 17) long   sampled classes; covered positions hold COVERED
        sizes_grids (B, 16, 16)    decoded per-patch-cell leaf sizes
    """
    device = device or next(model.parameters()).device
    labels = labels.to(device).long()
    B = labels.shape[0]
    was_training = model.training
    model.eval()

    use_cfg = cfg_scale != 1.0
    if use_cfg:
        # Conditional and unconditional branches in one batch of 2B.
        y_in = torch.cat([labels, labels])
        drop = torch.cat([torch.zeros(B, dtype=torch.long, device=device),
                          torch.ones(B, dtype=torch.long, device=device)])
        c = model.conditioning(y_in, force_drop_ids=drop)
    else:
        c = model.conditioning(labels)

    # Sequence being built. Start every position at COVERED; positions that turn
    # out to need a prediction are overwritten as we go.
    inputs = torch.full((B, S.SEQ_LEN), S.COVERED, dtype=torch.long, device=device)
    inputs[:, 0] = S.BOS
    tokens = torch.full((B, S.SEQ_LEN), S.COVERED, dtype=torch.long, device=device)

    covered = torch.zeros(B, S.N_REGIONS, dtype=torch.bool, device=device)

    for pos in range(S.SEQ_LEN):
        # Regions the root already determined need no forward pass.
        if pos > 0 and bool(covered[:, pos - 1].all()):
            continue

        model_in = inputs[:, :pos + 1]
        if use_cfg:
            model_in = torch.cat([model_in, model_in])
        logits = model(model_in, c=c)[:, pos]                      # (B or 2B, 17)

        if use_cfg:
            cond, uncond = logits[:B], logits[B:]
            # The root's STOP_4 column is -inf in both branches; the guidance
            # difference would be inf - inf = nan there, so only combine the
            # finite entries and let the masked class stay masked.
            guided = uncond + cfg_scale * (cond - uncond)
            logits = torch.where(torch.isfinite(cond) & torch.isfinite(uncond),
                                 guided, cond)

        sampled = _sample_from_logits(logits, temperature, top_k, generator)
        tokens[:, pos] = sampled

        if pos == 0:
            # Expand each root mask into the 16 covered flags.
            for q in range(4):
                split = ((sampled >> q) & 1).bool()          # (B,) True == descend
                qy, qx = (q // 2) * 2, (q % 2) * 2           # 8x8 quadrant -> region block
                for ry in range(2):
                    for rx in range(2):
                        covered[:, (qy + ry) * 4 + (qx + rx)] = ~split
        else:
            # A covered region emitted nothing; keep its token as COVERED.
            tokens[:, pos] = torch.where(covered[:, pos - 1],
                                         torch.full_like(sampled, S.COVERED),
                                         sampled)

        # Feed forward: next position sees this one's class, or COVERED.
        if pos + 1 < S.SEQ_LEN:
            inputs[:, pos + 1] = torch.where(covered[:, pos],
                                             torch.full_like(tokens[:, pos], S.COVERED),
                                             tokens[:, pos])

    grids = torch.stack([S.decode_sizes(tokens[i].cpu()) for i in range(B)])
    if was_training:
        model.train()
    return tokens, grids


def _sample_from_logits(logits, temperature, top_k, generator):
    """Categorical sample (or argmax at temperature 0) from (B, V) logits."""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def sample_plans(model, labels, device=None, **kwargs):
    """Sample structures and return compressor-ready plan dicts.

    Returns (plans, n_tokens):
        plans    list of B dicts with levels/positions/sizes, on `plan_device`
                 (CPU by default — where `plan_to_masks` wants them).
        n_tokens (B,) long, the sequence length each plan will produce.
    """
    plan_device = kwargs.pop('plan_device', None)
    _, grids = sample_structures(model, labels, device=device, **kwargs)
    plans = [S.plan_from_sizes_grid(g, device=plan_device) for g in grids]
    n_tokens = torch.tensor([p['sizes'].shape[0] for p in plans], dtype=torch.long)
    return plans, n_tokens
