"""Small class-conditional transformer over quadtree structure sequences.

Decoder-only, 17 positions, causal. Conditioning is the class label alone — the
prior runs once BEFORE the diffusion process, so there is no timestep in play —
and it enters only through adaptive layer norm — adaLN-Zero, the same scheme
`fit.model.modules.FiTBlock` uses — so no conditioning tokens occupy sequence
positions and classifier-free guidance is a single vector swap.

Two output heads rather than one, because the two positions have different
alphabets: position 0 emits one of 16 quadrant masks, positions 1..16 emit one of
17 (mask, or "stop at size 4"). Sharing a 17-way head would force the root to
learn that STOP_4 is unreachable; two heads make it structurally impossible.

Positions are learned embeddings, one per sequence slot — with only 17 slots
there is nothing to generalize across, so a table is both simplest and best.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fit.model.modules import LabelEmbedder
from fit.model.norms import create_norm
from fit.model.utils import modulate

from quadtree_prior import structure as S


class PriorBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning and causal attention.

    A trimmed FiTBlock: no rotary embeddings (positions are learned, and the
    sequence is 17 long), no packing, plain causal SDPA.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 norm_layer='layernorm', qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")

        self.norm1 = create_norm(norm_layer, hidden_size)
        self.norm2 = create_norm(norm_layer, hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (B, H, N, hd)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))

    def forward(self, x, c):
        # c: (B, D) — one conditioning vector per sequence.
        mods = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            m.unsqueeze(1) for m in mods]
        x = x + gate_msa * self._attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class QuadtreePrior(nn.Module):
    """Autoregressive prior p(structure | class) over 17-token quadtree codes.

    forward(inputs, y) -> (B, 17, 17) logits. Column 16 (STOP_4) is masked to
    -inf at position 0, where it has no meaning.

    Args:
        hidden_size, depth, num_heads, mlp_ratio: transformer shape.
        num_classes:       ImageNet classes (1000).
        class_dropout_prob: label dropout for classifier-free guidance.
    """

    def __init__(self, hidden_size=384, depth=6, num_heads=6, mlp_ratio=4.0,
                 num_classes=1000, class_dropout_prob=0.1, norm_layer='layernorm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.seq_len = S.SEQ_LEN

        self.token_embed = nn.Embedding(S.INPUT_VOCAB, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, S.SEQ_LEN, hidden_size))

        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList([
            PriorBlock(hidden_size, num_heads, mlp_ratio, norm_layer)
            for _ in range(depth)])

        self.norm_final = create_norm(norm_layer, hidden_size)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        # Separate heads: the root has no STOP_4 class.
        self.root_head = nn.Linear(hidden_size, S.ROOT_VOCAB)
        self.region_head = nn.Linear(hidden_size, S.REGION_VOCAB)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic)

        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # adaLN-Zero: every block starts as the identity, so depth costs nothing
        # at init.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        for head in (self.root_head, self.region_head):
            nn.init.constant_(head.weight, 0)
            nn.init.constant_(head.bias, 0)

    def conditioning(self, y, force_drop_ids=None):
        """Build the (B, D) adaLN conditioning vector from the class label."""
        return self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)

    def forward(self, inputs, y=None, force_drop_ids=None, c=None):
        """inputs (B, 17) long -> (B, 17, REGION_VOCAB) logits.

        `c` lets a caller pass a precomputed conditioning vector (the sampler
        reuses one across the 17 decode steps, and builds the CFG pair once).
        """
        if c is None:
            c = self.conditioning(y, force_drop_ids=force_drop_ids)

        x = self.token_embed(inputs) + self.pos_embed[:, :inputs.shape[1]]
        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))

        root_logits = self.root_head(x[:, :1])          # (B, 1, 16)
        region_logits = self.region_head(x[:, 1:])      # (B, 16, 17)
        # Pad the root to the wider alphabet with -inf so a single tensor can hold
        # both and STOP_4 stays unreachable at position 0.
        pad = torch.full((root_logits.shape[0], 1, S.REGION_VOCAB - S.ROOT_VOCAB),
                         float('-inf'), device=x.device, dtype=root_logits.dtype)
        return torch.cat([torch.cat([root_logits, pad], dim=-1), region_logits], dim=1)


def structure_loss(logits, targets, loss_mask=None):
    """Mean cross-entropy over supervised positions.

    Covered region positions carry IGNORE_INDEX in `targets`, which
    `cross_entropy` skips, so `loss_mask` is redundant here and accepted only for
    callers that already have it. Returns (loss, stats) where stats holds
    per-position-group accuracies for logging.
    """
    B, N, V = logits.shape
    flat_logits = logits.reshape(B * N, V).float()
    flat_targets = targets.reshape(B * N)
    loss = F.cross_entropy(flat_logits, flat_targets,
                           ignore_index=S.IGNORE_INDEX)

    with torch.no_grad():
        pred = flat_logits.argmax(dim=-1).reshape(B, N)
        sup = targets != S.IGNORE_INDEX
        correct = (pred == targets) & sup
        root_acc = correct[:, 0].float().mean()
        n_region = sup[:, 1:].sum()
        region_acc = (correct[:, 1:].sum() / n_region.clamp(min=1)
                      if n_region > 0 else torch.zeros((), device=logits.device))
        # A tree is only usable if every one of its decisions is right.
        exact = (correct.sum(dim=1) == sup.sum(dim=1)).float().mean()
    return loss, dict(root_acc=root_acc, region_acc=region_acc, exact=exact)
