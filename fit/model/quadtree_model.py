"""Quadtree-specific FiT model.

A trimmed variant of :class:`fit.model.fit_model.FiT`, specialised for the
variance-guided quadtree latents produced by
:mod:`fit.data.in1k_quadtree_latent_dataset`. Compared to the general model this
one drops two pieces of machinery that the quadtree pipeline never uses:

  * the learned-upsampler path (``use_upsampler`` and all the ``up_*`` blocks /
    bicubic + quadtree upsampling helpers), and
  * the original *unpacked* (dense, one-image-per-row) forward path.

The quadtree dataset always emits a single packed sequence (``B == 1`` with
per-token ``doc_ids`` and a FlexAttention block mask), so only the packed path
survives here.

Two things differ substantively from the base model:

  1.  **Per-leaf-size conditioning.** In the quadtree layout every token carries a
      leaf side ``tsize`` in ``LEAF_SIZES == (1, 2, 4, 8)`` — the compression level
      of that token. Instead of the sinusoidal :class:`SizeEmbedder` (which encoded
      a per-image pixel resolution), we use a small learned embedding table with one
      vector per leaf size and add it to the *per-token* conditioning ``c`` that
      drives adaLN in every block. It is zero-initialised so size conditioning
      starts as a no-op.

  2.  **Optional learned resampling.** With ``use_learned_resampling=False`` the
      token in/out projections are plain linears (:class:`PatchEmbedder` /
      :class:`FinalLayer`), i.e. the leaf values (mean-pooled latents already
      produced by the compressor) are embedded directly. With
      ``use_learned_resampling=True`` we instead down/up-sample each leaf with the
      shared learned :class:`Merge` / :class:`Split` blocks from
      :mod:`fit.quadtree_compression.adaptive_patch_pyramid`: a size-``N`` leaf is
      ``merge`` composed ``log2(N)`` times on the way in and ``split`` composed the
      same number of times on the way out. This is the "with / without learned
      up- and down-sampling" ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fit.model.modules import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder,
    FiTBlock, FinalLayer
)
from fit.model.utils import get_parameter_dtype
from fit.utils.eval_utils import init_from_ckpt
from fit.model.rope import VisionRotaryEmbedding
from fit.quadtree_compression.quadtree_compression import LEAF_SIZES
from fit.quadtree_compression.adaptive_patch_pyramid import Merge, Split


#################################################################################
#                        Learned per-leaf-size resampling                       #
#################################################################################

class LeafResampler(nn.Module):
    """Down/up-sample quadtree leaf tokens with the shared pyramid Merge/Split.

    Each token from the compressor is a 2x2 block of size-``N`` leaf values with
    ``4 * latent_channels`` channels (row-major over the 2x2 leaves). We treat
    that token as a tiny ``(2, 2)`` spatial grid of per-leaf channel vectors and
    fold it down to a single ``hidden_size`` vector.

    A size-``N`` leaf (N in ``LEAF_SIZES``) is the average of an ``N x N`` latent
    region. To keep the token space scale-consistent — exactly the design goal of
    the shared-weight pyramid — a size-``N`` leaf is embedded by composing the
    same :class:`Merge` block ``log2(N)`` extra times on top of the base stem, and
    reconstructed by composing :class:`Split` the same number of times. The
    weights are shared across all leaf sizes, so every compression level trains
    the same resampling function.

    encode: (n, 4*latent_ch) leaf tokens of side ``N`` -> (n, hidden_size)
    decode: (n, hidden_size) -> (n, 4*out_ch_per_leaf) for side-``N`` leaves
    """

    def __init__(self, latent_channels, hidden_size, out_channels):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        # Base grid of a token is 2x2 leaves. `stem` lifts a raw leaf channel
        # vector into the hidden width; `merge`/`split` fold/unfold one octave of
        # leaf size and are shared across levels (level-l leaf == merge^l(stem)).
        self.stem = nn.Conv2d(latent_channels, hidden_size, 1)
        self.merge = Merge(hidden_size)
        self.split = Split(hidden_size, hidden_size)
        # 2x2 leaf block -> one token vector (folds the base 2x2 patch grid).
        self.merge_patch = Merge(hidden_size)
        self.split_patch = Split(hidden_size, hidden_size)
        # Per-leaf output head after unfolding back to the 2x2 patch grid.
        self.out = nn.Conv2d(hidden_size, out_channels, 1)

    @staticmethod
    def _n_octaves(size):
        # size in {1, 2, 4, 8} -> {0, 1, 2, 3}
        return int(size).bit_length() - 1

    def encode(self, tokens, size):
        """(n, 4*latent_ch) size-``size`` leaf tokens -> (n, hidden_size)."""
        n = tokens.shape[0]
        # (n, 4*C) row-major over 2x2 leaves -> (n, C, 2, 2)
        x = tokens.reshape(n, 2, 2, self.latent_channels).permute(0, 3, 1, 2)
        x = self.stem(x)                                    # (n, D, 2, 2)
        # Fold `log2(size)` extra octaves so all leaf sizes land in one space.
        for _ in range(self._n_octaves(size)):
            # Merge halves the grid; a 2x2 grid would vanish, so tile before each
            # fold to keep a >=2 grid and let the shared Merge see a 2x2 context.
            x = self.merge(x.repeat_interleave(2, 2).repeat_interleave(2, 3))
        x = self.merge_patch(x)                             # (n, D, 1, 1)
        return x.reshape(n, self.hidden_size)

    def decode(self, feats, size):
        """(n, hidden_size) -> (n, 4*out_ch) size-``size`` leaf tokens."""
        n = feats.shape[0]
        x = feats.reshape(n, self.hidden_size, 1, 1)
        x = self.split_patch(x)                             # (n, D, 2, 2)
        for _ in range(self._n_octaves(size)):
            # Mirror of encode: unfold an octave, then pool back to a 2x2 grid.
            x = F.avg_pool2d(self.split(x), 2)
        x = self.out(x)                                     # (n, out_ch, 2, 2)
        return x.permute(0, 2, 3, 1).reshape(n, 4 * self.out_channels)


#################################################################################
#                              Quadtree FiT Model                               #
#################################################################################

class QuadtreeFiT(nn.Module):
    """Packed-only FiT specialised for variance-guided quadtree latents."""

    def __init__(
        self,
        context_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        use_checkpoint: bool = False,
        use_swiglu: bool = False,
        use_swiglu_large: bool = False,
        rel_pos_embed: Optional[str] = 'rope',
        norm_type: str = "layernorm",
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        adaln_bias: bool = True,
        adaln_type: str = "normal",
        adaln_lora_dim: int = None,
        rope_theta: float = 10000.0,
        custom_freqs: str = 'normal',
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        online_rope: bool = False,
        add_rel_pe_to_v: bool = False,
        pretrain_ckpt: str = None,
        ignore_keys: list = None,
        finetune: str = None,
        use_size_cond: bool = True,
        use_learned_resampling: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = self.in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.adaln_type = adaln_type
        self.online_rope = online_rope
        self.use_learned_resampling = use_learned_resampling

        # ---- token in/out projections ----------------------------------------
        # Without learned resampling: the compressor's leaf values (already a
        # 2x2 block => 4*in_channels per token) are embedded / read out with a
        # plain linear. With learned resampling: the shared Merge/Split pyramid
        # down/up-samples each leaf by its size (see LeafResampler).
        if use_learned_resampling:
            self.resampler = LeafResampler(in_channels, hidden_size, self.out_channels)
            self.x_embedder = None
            self.final_layer = None
        else:
            self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
            self.resampler = None

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Learned per-leaf-size embedding: one vector per LEAF_SIZES entry
        # (1/2/4/8), looked up per token and folded into the per-token
        # conditioning c. Zero-init (below) so it starts as a no-op.
        self.use_size_cond = use_size_cond
        if use_size_cond:
            self.size_embedder = nn.Embedding(len(LEAF_SIZES), hidden_size)
        # Map raw leaf side (1/2/4/8) -> contiguous index (0..3) for the table.
        self.register_buffer(
            "_size_to_idx",
            self._build_size_lut(LEAF_SIZES),
            persistent=False,
        )

        self.rel_pos_embed = VisionRotaryEmbedding(
            head_dim=hidden_size//num_heads, theta=rope_theta, custom_freqs=custom_freqs, online_rope=online_rope,
            max_pe_len_h=max_pe_len_h, max_pe_len_w=max_pe_len_w, decouple=decouple, ori_max_pe_len=ori_max_pe_len,
        )

        if adaln_type == 'lora':
            self.global_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
            )
        else:
            self.global_adaLN_modulation = None

        self.blocks = nn.ModuleList([FiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, swiglu=use_swiglu, swiglu_large=use_swiglu_large,
            rel_pos_embed=rel_pos_embed, add_rel_pe_to_v=add_rel_pe_to_v, norm_layer=norm_type,
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight, qkv_bias=qkv_bias, ffn_bias=ffn_bias,
            adaln_bias=adaln_bias, adaln_type=adaln_type, adaln_lora_dim=adaln_lora_dim
        ) for _ in range(depth)])
        if not use_learned_resampling:
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type)
            # adaLN-conditioned final norm reused by the learned-resampling head.
            self.final_norm = None
        else:
            # The learned resampler produces the per-leaf output; it still needs an
            # adaLN-modulated final norm before the resampler's decode.
            self.final_norm = FinalLayer(hidden_size, 1, hidden_size, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type)

        self.initialize_weights(pretrain_ckpt=pretrain_ckpt, ignore=ignore_keys)
        if finetune != None:
            self.finetune(type=finetune, unfreeze=ignore_keys)

    @staticmethod
    def _build_size_lut(leaf_sizes):
        """LUT tensor mapping a raw leaf side (1/2/4/8) to its table index."""
        lut = torch.zeros(max(leaf_sizes) + 1, dtype=torch.long)
        for idx, n in enumerate(leaf_sizes):
            lut[n] = idx
        return lut

    def initialize_weights(self, pretrain_ckpt=None, ignore=None):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.x_embedder is not None:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in list(self.blocks):
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)
        if self.adaln_type == 'lora':
            nn.init.constant_(self.global_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation[-1].bias, 0)

        # Zero-out output layer adaLN + linear so the model starts near identity.
        head = self.final_layer if self.final_layer is not None else self.final_norm
        if self.adaln_type == 'swiglu':
            nn.init.constant_(head.adaLN_modulation.fc2.weight, 0)
            nn.init.constant_(head.adaLN_modulation.fc2.bias, 0)
        else:   # adaln_type in ['normal', 'lora']
            nn.init.constant_(head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(head.adaLN_modulation[-1].bias, 0)
        if self.final_layer is not None:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
        else:
            # Learned-resampling head: zero the resampler's output conv so the
            # prediction starts at zero, matching the zeroed FinalLayer above.
            nn.init.constant_(self.resampler.out.weight, 0)
            nn.init.constant_(self.resampler.out.bias, 0)

        keys = list(self.state_dict().keys())
        ignore_keys = []
        if ignore != None:
            for ign in ignore:
                for key in keys:
                    if ign in key:
                        ignore_keys.append(key)
        ignore_keys = list(set(ignore_keys))
        if pretrain_ckpt != None:
            init_from_ckpt(self, pretrain_ckpt, ignore_keys, verbose=True)

        # Zero-init the learned size embedding so per-leaf-size conditioning
        # starts as a no-op. Must run after init_from_ckpt so checkpoint weights
        # don't overwrite the zeros.
        if self.use_size_cond:
            nn.init.constant_(self.size_embedder.weight, 0)

    def _rope_freqs(self, grid, size):
        """Compute RoPE cos/sin frequencies for a token grid.

        Returns (freqs_cos, freqs_sin), each (B, 1, N, head_dim).
        """
        if self.online_rope:
            # online_get_2d_rope_from_grid expects size (B, 1, 2). In packed mode
            # size is (B, max_n_pack, 2); use the per-batch max so the RoPE scale
            # covers the largest image in each pack.
            if size is not None and size.dim() == 3 and size.shape[1] > 1:
                rope_size = size.max(dim=1, keepdim=True).values  # (B, 1, 2)
            else:
                rope_size = size
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, rope_size)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
        return freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

    def _run_blocks(self, blocks, x, c, mask, freqs_cos, freqs_sin, global_adaln, block_mask):
        if not self.use_checkpoint:
            for block in blocks:
                x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln, block_mask)
        else:
            for block in blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln, block_mask
                )
        return x

    def _embed_tokens(self, x, tsize):
        """Token in-projection. x: (1, N, 4*C_in), tsize: (1, N) leaf side.

        Raw path: plain linear. Learned path: run each leaf-size group through
        the shared Merge stack (LeafResampler.encode) and scatter back.
        """
        if not self.use_learned_resampling:
            return self.x_embedder(x)
        B, N, _ = x.shape
        out = x.new_zeros(B, N, self.hidden_size)
        for n in LEAF_SIZES:
            sel = (tsize == n)                              # (1, N)
            if not bool(sel.any()):
                continue
            out[sel] = self.resampler.encode(x[sel], n).to(out.dtype)
        return out

    def _readout_tokens(self, x, c, tsize):
        """Token out-projection. x: (1, N, D) -> (1, N, 4*C_out).

        Raw path: adaLN FinalLayer linear. Learned path: adaLN final norm then
        per-leaf-size Split stack (LeafResampler.decode) scattered back.
        """
        if not self.use_learned_resampling:
            return self.final_layer(x, c)
        x = self.final_norm(x, c)                           # (1, N, D)
        B, N, _ = x.shape
        out = x.new_zeros(B, N, 4 * self.out_channels)
        for n in LEAF_SIZES:
            sel = (tsize == n)
            if not bool(sel.any()):
                continue
            out[sel] = self.resampler.decode(x[sel], n).to(out.dtype)
        return out

    def forward(self, x, t, y, grid, mask, size=None, tsize=None,
                doc_ids=None, block_mask=None, **kwargs):
        """Packed forward pass.

        x:        (1, N_total, 4*C_in)  packed quadtree leaf tokens
        t:        (1, max_n_pack)       timestep per packed image
        y:        (1, max_n_pack)       class label per packed image
        grid:     (1, 2, N_total)       per-token (w, h) center positions
        mask:     (1, N_total)          1 for valid tokens, 0 for padding
        size:     (1, max_n_pack, 2)    per-image full latent grid size (h, w)
        tsize:    (1, N_total)          per-token leaf side (1/2/4/8)
        doc_ids:  (1, N_total)          image index per token, -1 for padding
        block_mask:                     precomputed FlexAttention BlockMask

        return:   (1, N_total, 4*C_out)
        """
        assert doc_ids is not None, "QuadtreeFiT only supports the packed path"
        B = x.shape[0]
        D = self.hidden_size

        x = self._embed_tokens(x, tsize)                    # (1, N, D)

        # ---- per-image conditioning (t + y), expanded to per-token -----------
        max_n_pack = t.shape[1]
        t_flat = t.reshape(B * max_n_pack).to(x.dtype)
        y_flat = y.reshape(B * max_n_pack).to(torch.int)

        t_emb = self.t_embedder(t_flat).reshape(B, max_n_pack, D)   # (B, P, D)
        y_emb = self.y_embedder(y_flat, self.training).reshape(B, max_n_pack, D)
        c_pack = t_emb + y_emb                                        # (B, P, D)

        # Expand conditioning to per-token: each token inherits the embedding of
        # the image it belongs to. Padding tokens (doc_id=-1) are zeroed below.
        safe_ids = doc_ids.clamp(min=0)                               # (B, N)
        c = c_pack[torch.arange(B, device=x.device)[:, None], safe_ids]  # (B, N, D)

        # ---- per-token leaf-size conditioning --------------------------------
        # One learned vector per leaf size (1/2/4/8), added to the per-token c so
        # adaLN in every block sees the token's compression level.
        if self.use_size_cond and tsize is not None:
            size_idx = self._size_to_idx[tsize.clamp(min=0).long()]  # (B, N)
            c = c + self.size_embedder(size_idx)                     # (B, N, D)

        c = c * (doc_ids >= 0).to(c.dtype).unsqueeze(-1)             # zero padding

        if self.global_adaLN_modulation is not None:
            global_adaln = self.global_adaLN_modulation(c)           # (B, N, 6*D)
        else:
            global_adaln = 0.0

        # get RoPE frequencies in advance, then calculate attention.
        freqs_cos, freqs_sin = self._rope_freqs(grid, size)

        x = self._run_blocks(self.blocks, x, c, mask, freqs_cos, freqs_sin, global_adaln, block_mask)
        x = self._readout_tokens(x, c, tsize)               # (1, N, 4*C_out)
        x = x * mask[..., None]                             # zero out padding tokens
        return x

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    @property
    def dtype(self) -> torch.dtype:
        """`torch.dtype`: dtype of the module (assumes uniform param dtype)."""
        return get_parameter_dtype(self)

    def finetune(self, type, unfreeze):
        if type == 'full':
            return
        for name, param in self.named_parameters():
            param.requires_grad = False
        for unf in unfreeze:
            for name, param in self.named_parameters():
                if unf in name:  # LN means Layer Norm
                    param.requires_grad = True
