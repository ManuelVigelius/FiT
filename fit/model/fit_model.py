import torch
import torch.nn as nn
from typing import Optional

from fit.model.modules import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder, SizeEmbedder,
    FiTBlock, FinalLayer
)
from fit.model.utils import get_parameter_dtype
from fit.utils.eval_utils import init_from_ckpt
from fit.model.rope import VisionRotaryEmbedding

#################################################################################
#                                 Core FiT Model                                #
#################################################################################



class FiT(nn.Module):
    """
    Flexible Diffusion model with a Transformer backbone.
    """
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
        use_checkpoint: bool=False,
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
        use_size_cond: bool = False,
        use_upsampler: bool = False,
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

        self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.use_size_cond = use_size_cond
        if use_size_cond:
            self.size_embedder = SizeEmbedder(hidden_size)
        self.use_upsampler = use_upsampler
        # Number of trailing blocks that operate at full resolution. The first
        # (depth - upsampler_split) blocks run on the low-res latents; after
        # them the latents are bicubically upsampled to full resolution,
        # enriched with the projected full-res noisy image, and the remaining
        # `upsampler_split` blocks run at full resolution.
        self.upsampler_split = 4
        if use_upsampler:
            assert depth > self.upsampler_split, \
                "depth must exceed upsampler_split for the upsampler path"
            # Projects the patchified full-res noisy image into the inner
            # dimension. Zero-initialized in initialize_weights() so the
            # full-res enrichment starts as a no-op when fine-tuning.
            self.fr_embedder = PatchEmbedder(
                in_channels * patch_size**2, hidden_size, bias=True
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
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, norm_layer=norm_type, adaln_bias=adaln_bias, adaln_type=adaln_type)
        self.initialize_weights(pretrain_ckpt=pretrain_ckpt, ignore=ignore_keys)
        if finetune != None:
            self.finetune(type=finetune, unfreeze=ignore_keys)


    def initialize_weights(self, pretrain_ckpt=None, ignore=None):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if self.adaln_type in ['normal', 'lora']:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif self.adaln_type == 'swiglu':
                nn.init.constant_(block.adaLN_modulation.fc2.weight, 0)
                nn.init.constant_(block.adaLN_modulation.fc2.bias, 0)
        if self.adaln_type == 'lora':
            nn.init.constant_(self.global_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.global_adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        if self.adaln_type == 'swiglu':
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation.fc2.bias, 0)
        else:   # adaln_type in ['normal', 'lora']
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
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

        # Zero-init size embedder output projection so conditioning starts as a no-op.
        # Must run after init_from_ckpt so checkpoint weights don't overwrite the zeros.
        if self.use_size_cond:
            nn.init.constant_(self.size_embedder.mlp[2].weight, 0)
            nn.init.constant_(self.size_embedder.mlp[2].bias, 0)

        # Zero-init the full-res embedder so the full-res enrichment starts as
        # a no-op (the upsampled low-res latents pass through unchanged).
        # Must run after init_from_ckpt so checkpoint weights don't overwrite.
        if self.use_upsampler:
            nn.init.constant_(self.fr_embedder.proj.weight, 0)
            nn.init.constant_(self.fr_embedder.proj.bias, 0)


    def _rope_freqs(self, grid, size):
        """Compute RoPE cos/sin frequencies for a token grid.

        Returns (freqs_cos, freqs_sin), each (B, 1, N, head_dim).
        """
        if self.online_rope:
            # online_get_2d_rope_from_grid expects size (B, 1, 2).
            # In packed mode size is (B, max_n_pack, 2); use the per-batch max
            # so the RoPE scale covers the largest image in each pack.
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

    def forward(self, x, t, y, grid, mask, size=None, doc_ids=None, block_mask=None,
                x_fullres=None, grid_fullres=None, mask_fullres=None, size_fullres=None):
        """
        Forward pass of FiT.

        Unpacked mode (original):
            x:       (B, N, p**2*C_in)
            t:       (B,)
            y:       (B,)
            grid:    (B, 2, N)
            mask:    (B, N)   — 1 for valid tokens, 0 for padding
            size:    (B, 1, 2)
            doc_ids: None
            block_mask: None

        Packed mode (document-masking):
            x:       same shape but N = N_total (concatenated images, padded to multiple of 128)
            t:       (B, max_n_pack) — one timestep per image in the pack
            y:       (B, max_n_pack) — one label per image in the pack
            grid:    (B, 2, N_total)
            mask:    (B, N_total)
            size:    (B, max_n_pack, 2)
            doc_ids: (B, N_total)  — image index within sequence, -1 for padding
            block_mask: precomputed FlexAttention BlockMask (built outside the compiled region)

        Upsampler mode (use_upsampler=True):
            The low-res inputs are *packed* (variable per-image sizes) and feed
            the first (depth - upsampler_split) blocks exactly like the packed
            path above (B=1, doc_ids + block_mask). After those blocks the
            per-image latents are bicubically upsampled to the common full-res
            grid and stacked into a *dense* (n_pack, N_fr, D) batch; the last
            upsampler_split blocks then run densely at full resolution.
            x:            (1, N_total_lr, p**2*C_in) — packed low-res noisy
            doc_ids:      (1, N_total_lr)
            x_fullres:    (n_pack, N_fr, p**2*C_in)  — dense full-res noisy
            grid_fullres: (n_pack, 2, N_fr)          — all images share one size
            mask_fullres: (n_pack, N_fr)
            size_fullres: (1, n_pack, 2)
            Output is dense with shape (n_pack, N_fr, p**2*C_out).

        return: (B, N, p**2*C_out), or (n_pack, N_fr, p**2*C_out) in upsampler mode.
        """
        upsample = self.use_upsampler and x_fullres is not None
        if upsample:
            assert doc_ids is not None, "upsampler path requires packed low-res inputs"
            assert x.shape[0] == 1, "upsampler path expects B=1 packed low-res batch"
        B = x.shape[0]
        D = self.hidden_size

        x = self.x_embedder(x)                          # (B, N, D)

        c_pack = None  # (B, max_n_pack, D) per-image conditioning, set in packed path
        if doc_ids is not None:
            # ---- Packed path ------------------------------------------------
            # t and y are (B, max_n_pack); embed each, then expand to per-token.
            max_n_pack = t.shape[1]
            t_flat = t.reshape(B * max_n_pack).to(x.dtype)
            y_flat = y.reshape(B * max_n_pack).to(torch.int)

            t_emb = self.t_embedder(t_flat).reshape(B, max_n_pack, D)   # (B, P, D)
            y_emb = self.y_embedder(y_flat, self.training).reshape(B, max_n_pack, D)
            c_pack = t_emb + y_emb                                        # (B, P, D)

            if self.use_size_cond and size is not None:
                # size: (B, max_n_pack, 2); encode height grid dim as pixel resolution
                size_flat = size.reshape(B * max_n_pack, 2)[:, 0]
                size_px = (size_flat * self.patch_size * 8).to(x.dtype)   # (B*P,)
                size_emb = self.size_embedder(size_px).reshape(B, max_n_pack, D)
                c_pack = c_pack + size_emb

            # Expand conditioning to per-token: each token inherits the
            # embedding of the image it belongs to. Padding tokens (doc_id=-1)
            # are zeroed out via the mask below.
            safe_ids = doc_ids.clamp(min=0)                               # (B, N)
            c = c_pack[torch.arange(B, device=x.device)[:, None], safe_ids]  # (B, N, D)
            c = c * (doc_ids >= 0).to(c.dtype).unsqueeze(-1)              # zero padding

            if self.global_adaLN_modulation is not None:
                # global_adaln needs to be (B, N, 6*D) in packed mode
                global_adaln = self.global_adaLN_modulation(c)            # (B, N, 6*D)
            else:
                global_adaln = 0.0
        else:
            # ---- Original (unpacked) path -----------------------------------
            t = t.to(x.dtype)
            t_emb = self.t_embedder(t)                                    # (B, D)
            y_emb = self.y_embedder(y, self.training)                     # (B, D)
            c = t_emb + y_emb                                             # (B, D)

            if self.use_size_cond and size is not None:
                # size: (B, 1, 2); encode height grid dim as pixel resolution
                size_px = (size[:, 0, 0] * self.patch_size * 8).to(x.dtype)  # (B,)
                c = c + self.size_embedder(size_px)

            if self.global_adaLN_modulation is not None:
                global_adaln = self.global_adaLN_modulation(c)            # (B, 6*D)
            else:
                global_adaln = 0.0

        # get RoPE frequencies in advance, then calculate attention.
        freqs_cos, freqs_sin = self._rope_freqs(grid, size)

        if not upsample:
            x = self._run_blocks(self.blocks, x, c, mask, freqs_cos, freqs_sin, global_adaln, block_mask)
            x = self.final_layer(x, c)                  # (B, N, p**2 * C_out)
            x = x * mask[..., None]                     # zero out padding tokens
            return x

        # ---- Upsampler path ------------------------------------------------
        # 1. Run the first (depth - upsampler_split) blocks on the *packed*
        #    low-res latents (variable per-image sizes).
        split = self.depth - self.upsampler_split
        x = self._run_blocks(
            self.blocks[:split], x, c, mask, freqs_cos, freqs_sin, global_adaln, block_mask
        )

        # 2. Per image: bicubically upsample its low-res latents to the common
        #    full-res grid, then stack into a dense (n_pack, N_fr, D) batch.
        n_pack = c_pack.shape[1]
        H_fr = int(size_fullres[0, 0, 0]); W_fr = int(size_fullres[0, 0, 1])
        N_fr = H_fr * W_fr
        doc_ids_lr = doc_ids[0]                                 # (N_total_lr,)
        x_fr = x.new_zeros(n_pack, N_fr, D)
        for i in range(n_pack):
            H_lr = int(size[0, i, 0]); W_lr = int(size[0, i, 1])
            tok_i = x[0, doc_ids_lr == i]                       # (H_lr*W_lr, D)
            sp_i = tok_i.transpose(0, 1).reshape(1, D, H_lr, W_lr)
            sp_i = nn.functional.interpolate(
                sp_i.float(), size=(H_fr, W_fr), mode='bicubic', align_corners=True
            ).to(x.dtype)
            x_fr[i] = sp_i.reshape(D, N_fr).transpose(0, 1)     # (N_fr, D)

        # 3. Project the full-res noisy image and add it to the upsampled latents.
        x = x_fr + self.fr_embedder(x_fullres)                  # (n_pack, N_fr, D)

        # 4. Dense conditioning for the full-res blocks: one vector per image.
        c_fr = c_pack[0]                                         # (n_pack, D)
        if self.global_adaLN_modulation is not None:
            global_adaln_fr = self.global_adaLN_modulation(c_fr)  # (n_pack, 6*D)
        else:
            global_adaln_fr = 0.0

        # 5. Run the last upsampler_split blocks densely at full resolution.
        #    The dense FR batch has n_pack rows (all the same size), so RoPE
        #    needs a per-row size of shape (n_pack, 1, 2).
        rope_size_fr = size_fullres[0].unsqueeze(1)             # (n_pack, 1, 2)
        freqs_cos_fr, freqs_sin_fr = self._rope_freqs(grid_fullres, rope_size_fr)
        x = self._run_blocks(
            self.blocks[split:], x, c_fr, mask_fullres, freqs_cos_fr, freqs_sin_fr,
            global_adaln_fr, None,
        )

        x = self.final_layer(x, c_fr)               # (n_pack, N_fr, p**2 * C_out)
        x = x * mask_fullres[..., None]             # zero out padding tokens
        return x
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    
    
    def finetune(self, type, unfreeze):
        if type == 'full':
            return
        for name, param in self.named_parameters():
                param.requires_grad = False
        for unf in unfreeze:
            for name, param in self.named_parameters():
                if unf in name: # LN means Layer Norm
                    param.requires_grad = True
        