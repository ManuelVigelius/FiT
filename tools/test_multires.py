"""
Sanity checks for multi-resolution fine-tuning components.

Run from the project root:
    python tools/test_multires.py

Tests:
  1. Dataset random resize  — shapes, mask length, valid grid values
  2. SizeEmbedder zero-init — output projection starts at zero
  3. Model forward          — correct output shape with size conditioning
  4. Checkpoint load        — output projection remains zero after ckpt load
  5. Loss B                 — upsample loss computes a finite scalar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Dataset random resize
# ---------------------------------------------------------------------------
def test_dataset_resize():
    from fit.data.in1k_latent_dataset import IN1kLatentDataset
    data_path = 'datasets/imagenet1k_latents_256_sd_vae_ft_ema'
    if not os.path.isdir(data_path):
        print(f'[SKIP] dataset not found: {data_path}')
        return
    ds = IN1kLatentDataset(data_path, target_len=256, random='crop',
                           resize_range=(8, 16))
    item = ds[0]
    seq_len = int(item['mask'].sum())
    H_g = int(item['size'][0, 0])
    W_g = int(item['size'][0, 1])
    assert item['feature'].shape == (256, 16), f"unexpected feature shape {item['feature'].shape}"
    assert seq_len == H_g * W_g, f"mask sum {seq_len} != grid area {H_g*W_g}"
    assert H_g in [8, 10, 12, 14, 16], f"unexpected grid size {H_g}"
    assert H_g == W_g, "expected square grid"
    print(f'[PASS] dataset resize: grid={H_g}x{W_g}, tokens={seq_len}')


# ---------------------------------------------------------------------------
# 2. SizeEmbedder zero-init (no checkpoint needed)
# ---------------------------------------------------------------------------
def test_size_embedder_zeroinit():
    from fit.model.fit_model import FiT
    model = FiT(
        context_size=256, patch_size=2, in_channels=4, hidden_size=256,
        depth=2, num_heads=4, use_sit=True, use_swiglu=True,
        adaln_type='lora', adaln_lora_dim=64, use_size_cond=True,
    )
    w = model.size_embedder.mlp[2].weight
    b = model.size_embedder.mlp[2].bias
    assert w.abs().max() == 0 and b.abs().max() == 0, \
        f'Output proj not zero-init! max_w={w.abs().max()}, max_b={b.abs().max()}'
    print('[PASS] SizeEmbedder output projection is zero-initialized')


# ---------------------------------------------------------------------------
# 3. Model forward with size conditioning (unpacked)
# ---------------------------------------------------------------------------
def test_model_forward_unpacked():
    from fit.model.fit_model import FiT
    B, N = 2, 64   # 8x8 grid
    model = FiT(
        context_size=256, patch_size=2, in_channels=4, hidden_size=256,
        depth=2, num_heads=4, use_sit=True, use_swiglu=True,
        adaln_type='lora', adaln_lora_dim=64, use_size_cond=True, online_rope=True,
    ).eval()
    x    = torch.randn(B, N, 16)
    t    = torch.rand(B)
    y    = torch.zeros(B, dtype=torch.long)
    hs   = torch.arange(8, dtype=torch.float32)
    ws   = torch.arange(8, dtype=torch.float32)
    gh, gw = torch.meshgrid(hs, ws, indexing='ij')
    grid = torch.stack([gh.reshape(-1), gw.reshape(-1)]).unsqueeze(0).expand(B, -1, -1)
    mask = torch.ones(B, N, dtype=torch.uint8)
    size = torch.tensor([[[8, 8]]], dtype=torch.int32).expand(B, -1, -1)
    with torch.no_grad():
        out = model(x, t, y, grid.long(), mask, size)
    assert out.shape == (B, N, 16), f'unexpected output shape {out.shape}'
    print('[PASS] model forward (unpacked) with size conditioning')


# ---------------------------------------------------------------------------
# 4. Checkpoint load — output proj stays zero
# ---------------------------------------------------------------------------
def test_checkpoint_load():
    from fit.model.fit_model import FiT
    ckpt = 'checkpoints/fitv2_xl.safetensors'
    if not os.path.exists(ckpt):
        print(f'[SKIP] checkpoint not found: {ckpt}')
        return
    model = FiT(
        context_size=256, patch_size=2, in_channels=4, hidden_size=1152,
        depth=36, num_heads=16, use_sit=True, use_swiglu=True,
        adaln_type='lora', adaln_lora_dim=288, use_size_cond=True,
        online_rope=True, pretrain_ckpt=ckpt,
    )
    w = model.size_embedder.mlp[2].weight
    assert w.abs().max() == 0, \
        f'Output proj not zero after ckpt load! max={w.abs().max()}'
    print('[PASS] pretrained checkpoint loads; size_embedder output proj remains zero')


# ---------------------------------------------------------------------------
# 5. Loss B — upsample loss
# ---------------------------------------------------------------------------
def test_loss_b():
    from fit.model.fit_model import FiT
    from fit.scheduler.transport import create_transport

    B = 2
    H_lr, W_lr = 8, 8     # low-res grid (64 tokens)
    H_fr, W_fr = 16, 16   # full-res grid (256 tokens)
    N_lr = H_lr * W_lr
    N_fr = H_fr * W_fr

    model = FiT(
        context_size=256, patch_size=2, in_channels=4, hidden_size=256,
        depth=2, num_heads=4, use_sit=True, use_swiglu=True,
        adaln_type='lora', adaln_lora_dim=64, use_size_cond=True, online_rope=True,
    ).eval()

    transport = create_transport(multires_loss='B')

    hs = torch.arange(H_lr, dtype=torch.float32)
    ws = torch.arange(W_lr, dtype=torch.float32)
    gh, gw = torch.meshgrid(hs, ws, indexing='ij')
    grid_lr = torch.stack([gh.reshape(-1), gw.reshape(-1)]).unsqueeze(0).expand(B, -1, -1)

    x1_lr  = torch.randn(B, N_lr, 16)
    x1_fr  = torch.randn(B, N_fr, 16)
    mask_lr = torch.ones(B, N_lr, dtype=torch.uint8)
    mask_fr = torch.ones(B, N_fr, dtype=torch.uint8)
    size_lr = torch.tensor([[[H_lr, W_lr]]], dtype=torch.int32).expand(B, -1, -1)
    size_fr = torch.tensor([[[H_fr, W_fr]]], dtype=torch.int32).expand(B, -1, -1)
    y = torch.zeros(B, dtype=torch.long)

    model_kwargs = dict(
        y=y, grid=grid_lr.long(), mask=mask_lr, size=size_lr,
        doc_ids=None, n_pack=None,
        x1_fullres=x1_fr, mask_fullres=mask_fr, size_fullres=size_fr,
    )

    with torch.no_grad():
        terms = transport.training_losses(model, x1_lr, model_kwargs)
    loss = terms['loss'].mean()
    assert loss.isfinite(), f'Loss B is not finite: {loss}'
    print(f'[PASS] Loss B: value={loss.item():.4f}')


# ---------------------------------------------------------------------------
# 6. Loss C — synchronized noise + ResNet upsampler
# ---------------------------------------------------------------------------
def test_loss_c():
    from fit.model.fit_model import FiT
    from fit.scheduler.transport import create_transport

    B = 2
    H_lr, W_lr = 8, 8
    H_fr, W_fr = 16, 16
    N_lr = H_lr * W_lr
    N_fr = H_fr * W_fr

    model = FiT(
        context_size=256, patch_size=2, in_channels=4, hidden_size=256,
        depth=2, num_heads=4, use_sit=True, use_swiglu=True,
        adaln_type='lora', adaln_lora_dim=64, use_size_cond=True,
        online_rope=True, use_upsampler=True,
    ).eval()

    # Verify upsampler output_proj is zero-initialized
    w = model.upsampler.output_proj.weight
    assert w.abs().max() == 0, f'upsampler output_proj not zero-init! max={w.abs().max()}'

    transport = create_transport(multires_loss='C')

    hs = torch.arange(H_lr, dtype=torch.float32)
    ws = torch.arange(W_lr, dtype=torch.float32)
    gh, gw = torch.meshgrid(hs, ws, indexing='ij')
    grid_lr = torch.stack([gh.reshape(-1), gw.reshape(-1)]).unsqueeze(0).expand(B, -1, -1)

    x1_lr   = torch.randn(B, N_lr, 16)
    x1_fr   = torch.randn(B, N_fr, 16)
    mask_lr = torch.ones(B, N_lr, dtype=torch.uint8)
    mask_fr = torch.ones(B, N_fr, dtype=torch.uint8)
    size_lr = torch.tensor([[[H_lr, W_lr]]], dtype=torch.int32).expand(B, -1, -1)
    size_fr = torch.tensor([[[H_fr, W_fr]]], dtype=torch.int32).expand(B, -1, -1)
    y = torch.zeros(B, dtype=torch.long)

    model_kwargs = dict(
        y=y, grid=grid_lr.long(), mask=mask_lr, size=size_lr,
        doc_ids=None, n_pack=None,
        x1_fullres=x1_fr, mask_fullres=mask_fr, size_fullres=size_fr,
    )

    with torch.no_grad():
        terms = transport.training_losses(model, x1_lr, model_kwargs)
    loss = terms['loss'].mean()
    assert loss.isfinite(), f'Loss C is not finite: {loss}'
    print(f'[PASS] Loss C: value={loss.item():.4f}')


if __name__ == '__main__':
    test_dataset_resize()
    test_size_embedder_zeroinit()
    test_model_forward_unpacked()
    test_checkpoint_load()
    test_loss_b()
    test_loss_c()
    print('\nAll tests passed.')
