import torch
import math
import torch.nn.functional as F
from torch import nn
from timm.layers.mlp import SwiGLU, Mlp  
from typing import Callable, Optional
from fit.model.rope import rotate_half
from fit.model.utils import modulate
from fit.model.norms import create_norm

#################################################################################
#           Embedding Layers for Patches, Timesteps and Class Labels            #
#################################################################################

class PatchEmbedder(nn.Module):
    """
    Embeds latent features into vector representations
    """
    def __init__(self, 
        input_dim, 
        embed_dim, 
        bias: bool = True,
        norm_layer: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.proj = nn.Linear(input_dim, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # (B, L, patch_size ** 2 * C) -> (B, L, D)
        x = self.norm(x)
        return x  

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1]).to(device=t.device)], dim=-1)
        return embedding.to(dtype=t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SizeEmbedder(nn.Module):
    """
    Encodes a scalar pixel resolution (e.g. 256) into a hidden-size vector.
    Follows the same design as TimestepEmbedder. The output projection is
    zero-initialized in FiT.initialize_weights() so that size conditioning
    starts as a no-op when fine-tuning from a pretrained checkpoint.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def size_embedding(s, dim, max_period=10000):
        """
        Sinusoidal embedding for scalar size values.
        s: (N,) float tensor of size values (pixel resolution, e.g. 64..256)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=s.device)
        args = s[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=s.dtype)

    def forward(self, s):
        # s: (N,) flat tensor of pixel resolutions
        s_freq = self.size_embedding(s, self.frequency_embedding_size)
        return self.mlp(s_freq)




class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResNetUpsampler(nn.Module):
    """
    Lightweight convolutional residual network for Loss C.

    Takes the concatenation of:
      - upsampled low-res transformer output  (B, C_vae, H_fr_sp, W_fr_sp)
      - full-res noisy input xt_fr            (B, C_vae, H_fr_sp, W_fr_sp)
    and predicts the full-res velocity field  (B, C_vae, H_fr_sp, W_fr_sp).

    Operates in the *latent spatial* domain (after unpatchify), so spatial dims
    are H_grid * patch_size (e.g. 32x32 for a 16x16 grid with patch_size=2).
    The output_proj is zero-initialized in FiT.initialize_weights() so the
    upsampler starts as a no-op when loading from a pretrained checkpoint.
    """
    def __init__(self, in_channels: int = 8, hidden_channels: int = 128,
                 out_channels: int = 4, num_blocks: int = 3):
        super().__init__()
        self.input_proj  = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.blocks      = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(num_blocks)])
        self.output_proj = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x_lr_up: torch.Tensor, xt_fr: torch.Tensor) -> torch.Tensor:
        """
        x_lr_up : (B, C_vae, H_sp, W_sp)  — upsampled low-res latent
        xt_fr   : (B, C_vae, H_sp, W_sp)  — full-res noisy observation
        returns : (B, C_vae, H_sp, W_sp)  — predicted full-res velocity
        """
        x = torch.cat([x_lr_up, xt_fr], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


#################################################################################
#                                  Attention                                    #
#################################################################################

# modified from timm and eva-02
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
# https://github.com/baaivision/EVA/blob/master/EVA-02/asuka/modeling_finetune.py


class Attention(nn.Module):

    def __init__(self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        rel_pos_embed: Optional[str] = None,
        add_rel_pe_to_v: bool = False, 
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if q_norm == 'layernorm' and qk_norm_weight == True:
            q_norm = 'w_layernorm'
        if k_norm == 'layernorm' and qk_norm_weight == True:
            k_norm = 'w_layernorm'
        
        self.q_norm = create_norm(q_norm, self.head_dim)
        self.k_norm = create_norm(k_norm, self.head_dim)
        

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rel_pos_embed = None if rel_pos_embed==None else rel_pos_embed.lower() 
        self.add_rel_pe_to_v = add_rel_pe_to_v
        
        

    def forward(self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        block_mask=None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # (B, n_h, N, D_h)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rel_pos_embed in ['rope', 'xpos']:  # multiplicative rel_pos_embed
            if self.add_rel_pe_to_v:
                v = v * freqs_cos + rotate_half(v) * freqs_sin
            q = (q * freqs_cos + rotate_half(q) * freqs_sin).to(v.dtype)
            k = (k * freqs_cos + rotate_half(k) * freqs_sin).to(v.dtype)

        if block_mask is not None:
            # Packed / document-masking path via FlexAttention.
            # block_mask is precomputed outside the compiled region to avoid
            # triggering Inductor recompilations on variable sequence lengths.
            from torch.nn.attention.flex_attention import flex_attention
            x = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        else:
            # Original padding-mask path (used when doc_ids is not provided,
            # e.g. during inference with a single image).
            attn_mask = mask[:, None, None, :]  # (B, N) -> (B, 1, 1, N)
            attn_mask = (attn_mask == attn_mask.transpose(-2, -1))  # (B, 1, N, N)

            if x.device.type == "cpu":
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                with torch.backends.cuda.sdp_kernel(enable_flash=True):
                    x = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask,
                        dropout_p=self.attn_drop.p if self.training else 0.,
                    )

        mask = torch.not_equal(mask, torch.zeros_like(mask)).to(mask)   # (B, N) -> (B, N)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = x * mask[..., None] # mask: (B, N) -> (B, N, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#                               Basic FiT Module                                #
#################################################################################

class FiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0, 
        swiglu=True,
        swiglu_large=False,
        rel_pos_embed=None,
        add_rel_pe_to_v=False,
        norm_layer: str = 'layernorm',
        q_norm: Optional[str] = None,
        k_norm: Optional[str] = None,
        qk_norm_weight: bool = False,
        qkv_bias=True,
        ffn_bias=True,
        adaln_bias=True,
        adaln_type='normal',
        adaln_lora_dim: int = None,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = create_norm(norm_layer, hidden_size)
        self.norm2 = create_norm(norm_layer, hidden_size)
        
        self.attn = Attention(
            hidden_size, num_heads=num_heads, rel_pos_embed=rel_pos_embed, 
            q_norm=q_norm, k_norm=k_norm, qk_norm_weight=qk_norm_weight,
            qkv_bias=qkv_bias, add_rel_pe_to_v=add_rel_pe_to_v, 
            **block_kwargs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if swiglu:
            if swiglu_large:
                self.mlp = SwiGLU(in_features=hidden_size, hidden_features=mlp_hidden_dim, bias=ffn_bias)
            else:
                self.mlp = SwiGLU(in_features=hidden_size, hidden_features=(mlp_hidden_dim*2)//3, bias=ffn_bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), bias=ffn_bias)
        if adaln_type == 'normal':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=adaln_bias)
            )
        elif adaln_type == 'lora':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=adaln_bias),
                nn.Linear(adaln_lora_dim, 6 * hidden_size, bias=adaln_bias)
            )
        elif adaln_type == 'swiglu':
            self.adaLN_modulation = SwiGLU(
                in_features=hidden_size, hidden_features=(hidden_size//4)*3, out_features=6*hidden_size, bias=adaln_bias
            )

    def forward(self, x, c, mask, freqs_cos, freqs_sin, global_adaln=0.0, block_mask=None):
        if c.dim() == 2:
            # Standard (unpacked) path: c is (B, D), one vector per batch element.
            mods = (self.adaLN_modulation(c) + global_adaln).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [m.unsqueeze(1) for m in mods]
        else:
            # Packed path: c is (B, N, D), one vector per token.
            # global_adaln is broadcast-compatible (scalar 0.0 or (B, 1, 6*D)).
            mods = (self.adaLN_modulation(c) + global_adaln).chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask, freqs_cos, freqs_sin, block_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, norm_layer: str = 'layernorm', adaln_bias=True, adaln_type='normal'):
        super().__init__()
        self.norm_final = create_norm(norm_type=norm_layer, dim=hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if adaln_type == 'swiglu':
            self.adaLN_modulation = SwiGLU(in_features=hidden_size, hidden_features=hidden_size//2, out_features=2*hidden_size, bias=adaln_bias)
        else:   # adaln_type in ['normal', 'lora']
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=adaln_bias)
            )
        
    def forward(self, x, c):
        if c.dim() == 2:
            # Standard path: c is (B, D); modulate() will unsqueeze to (B, 1, D).
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        else:
            # Packed path: c is (B, N, D); modulate() uses it as-is.
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
