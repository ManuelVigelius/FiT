import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange
from fit.model.modules import (
    PatchEmbedder, TimestepEmbedder, LabelEmbedder, SizeEmbedder,
    ResNetUpsampler, FiTBlock, FinalLayer
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
        use_sit: bool = False,
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
        time_shifting: int = 1,
        use_size_cond: bool = False,
        use_upsampler: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        assert not (learn_sigma and use_sit)
        self.learn_sigma = learn_sigma
        self.use_sit = use_sit
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
        self.time_shifting = time_shifting

        self.x_embedder = PatchEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.use_size_cond = use_size_cond
        if use_size_cond:
            self.size_embedder = SizeEmbedder(hidden_size)
        self.use_upsampler = use_upsampler
        if use_upsampler:
            self.upsampler = ResNetUpsampler()


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

        # Zero-init upsampler output projection so it starts as a no-op.
        if self.use_upsampler:
            nn.init.constant_(self.upsampler.output_proj.weight, 0)
            nn.init.constant_(self.upsampler.output_proj.bias, 0)


    def unpatchify(self, x, hw):
        """
        args:
            x: (B, p**2 * C_out, N)
            N = h//p * w//p
        return: 
            imgs: (B, C_out, H, W)
        """
        h, w = hw
        p = self.patch_size
        if self.use_sit:
            x = rearrange(x, "b (h w) c -> b h w c", h=h//p, w=w//p) # (B, h//2 * w//2, 16) -> (B, h//2, w//2, 16)
            x = rearrange(x, "b h w (c p1 p2) -> b c (h p1) (w p2)", p1=p, p2=p) # (B, h//2, w//2, 16) -> (B, h, w, 4)
        else:
            x = rearrange(x, "b c (h w) -> b c h w", h=h//p, w=w//p) # (B, 16, h//2 * w//2) -> (B, 16, h//2, w//2)
            x = rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=p, p2=p) # (B, 16, h//2, w//2) -> (B, h, w, 4)
        return x

    def forward(self, x, t, y, grid, mask, size=None, doc_ids=None):
        """
        Forward pass of FiT.

        Unpacked mode (original):
            x:       (B, p**2*C_in, N) or (B, N, p**2*C_in) depending on use_sit
            t:       (B,)
            y:       (B,)
            grid:    (B, 2, N)
            mask:    (B, N)   — 1 for valid tokens, 0 for padding
            size:    (B, 1, 2)
            doc_ids: None

        Packed mode (document-masking):
            x:       same shape but N = N_total (concatenated images, padded to multiple of 128)
            t:       (B, max_n_pack) — one timestep per image in the pack
            y:       (B, max_n_pack) — one label per image in the pack
            grid:    (B, 2, N_total)
            mask:    (B, N_total)
            size:    (B, max_n_pack, 2)
            doc_ids: (B, N_total)  — image index within sequence, -1 for padding

        return: same shape as x.
        """
        B = x.shape[0]
        D = self.hidden_size

        if not self.use_sit:
            x = rearrange(x, 'B C N -> B N C')
        x = self.x_embedder(x)                          # (B, N, D)

        if doc_ids is not None:
            # ---- Packed path ------------------------------------------------
            # t and y are (B, max_n_pack); embed each, then expand to per-token.
            max_n_pack = t.shape[1]
            t_flat = t.reshape(B * max_n_pack)
            y_flat = y.reshape(B * max_n_pack).to(torch.int)

            t_flat = torch.clamp(
                self.time_shifting * t_flat / (1 + (self.time_shifting - 1) * t_flat), max=1.0
            ).float().to(x.dtype)

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
            t = torch.clamp(
                self.time_shifting * t / (1 + (self.time_shifting - 1) * t), max=1.0
            ).float().to(x.dtype)
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
        if self.online_rope:
            freqs_cos, freqs_sin = self.rel_pos_embed.online_get_2d_rope_from_grid(grid, size)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        else:
            freqs_cos, freqs_sin = self.rel_pos_embed.get_cached_2d_rope_from_grid(grid)
            freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

        if not self.use_checkpoint:
            for block in self.blocks:
                x = block(x, c, mask, freqs_cos, freqs_sin, global_adaln, doc_ids)
        else:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, c, mask, freqs_cos, freqs_sin, global_adaln, doc_ids
                )

        x = self.final_layer(x, c)                      # (B, N, p**2 * C_out)
        x = x * mask[..., None]                         # zero out padding tokens
        if not self.use_sit:
            x = rearrange(x, 'B N C -> B C N')
        return x
    
    
    
    def forward_with_cfg(self, x, t, y, grid, mask, size, cfg_scale, scale_pow=0.0):
        """
        Forward pass with classifier free guidance of FiT.
        x: (2B, N, p**2 * C_in) if use_sit else (2B, p**2 * C_in, N) tensor of sequential inputs (flattened latent features of images, N=H*W/(p**2))
        t: (2B,) tensor of diffusion timesteps
        y: (2B,) tensor of class labels
        grid: (2B, 2, N): tensor of height and weight indices that spans a grid
        mask: (2B, N): tensor of the mask for the sequence
        cfg_scale: float > 1.0
        return: (B, p**2 * C_out, N), where C_out=2*C_in if leran_sigma, C_out=C_in otherwise.
        """
        half = x[: len(x) // 2]                     # (2B, ...) -> (B, ...)
        combined = torch.cat([half, half], dim=0)   # (2B, ...)
        model_out = self.forward(combined, t, y, grid, mask, size)  # (2B, N, C) is use_sit else (2B, C, N) , where C = p**2 * C_out
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # C_cfg = self.in_channels * self.patch_size * self.patch_size
        C_cfg = 3 * self.patch_size * self.patch_size
        if self.use_sit:
            eps, rest = model_out[:, :, :C_cfg], model_out[:, :, C_cfg:]  # eps: (2B, N, C_cfg), where C_cfg = p**2 * 3
        else:
            eps, rest = model_out[:, :C_cfg], model_out[:, C_cfg:]  # eps: (2B, C_cfg, N), where C_cfg = p**2 * 3
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)   # (2B, C_cfg, H, W) -> (B, C_cfg, H, W)        
        # from https://github.com/sail-sg/MDT/blob/main/masked_diffusion/models.py#L506
        # improved classifier-free guidance
        if scale_pow == 0.0:
            real_cfg_scale = cfg_scale
        else:
            scale_step = (
                1-torch.cos(((1-torch.clamp_max(t, 1.0))**scale_pow)*torch.pi)
            )*1/2 # power-cos scaling 
            real_cfg_scale = (cfg_scale-1)*scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1)
        t = t / (self.time_shifting + (1 - self.time_shifting) * t)
        half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)    # (B, ...)
        eps = torch.cat([half_eps, half_eps], dim=0)    # (B, ...) -> (2B, ...)
        if self.use_sit:
            return torch.cat([eps, rest], dim=2)          # (2B, N, C), where C = p**2 * C_out
        else:
            return torch.cat([eps, rest], dim=1)          # (2B, C, N), where C = p**2 * C_out
        
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
        