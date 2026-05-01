import torch
import torch.nn.functional as F
from einops import rearrange


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """Spatial latent → token sequence.

    Args:
        x: (B, C, H, W) spatial tensor
        p: patch size
    Returns:
        (B, N, C*p*p) token sequence in (c p1 p2) layout, N = (H//p)*(W//p)
    """
    return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)


def unpatchify(x: torch.Tensor, hw: tuple, p: int) -> torch.Tensor:
    """Token sequence → spatial latent.

    Args:
        x:  (B, N, C*p*p) token sequence in (c p1 p2) layout
        hw: (H, W) output spatial size in pixels (not grid cells)
        p:  patch size
    Returns:
        (B, C, H, W) spatial tensor
    """
    h, w = hw
    return rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                     h=h//p, w=w//p, p1=p, p2=p)


def spatial_resize(x: torch.Tensor, H: int, W: int,
                   H_out: int, W_out: int,
                   p: int = 2,
                   mode: str = 'bilinear') -> torch.Tensor:
    """Resize a patchified token sequence via interpolation.

    Args:
        x:            (B, N, C*p*p) or (N, C*p*p) token sequence in (c p1 p2) layout
        H, W:         input grid dims (in patch units)
        H_out, W_out: output grid dims (in patch units)
        p:            patch size
        mode:         'bilinear' for upsampling, 'area' for downsampling
    Returns:
        same leading dims, N replaced by H_out*W_out
    """
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
    sp = unpatchify(x, (H * p, W * p), p)
    kwargs = {} if mode == 'area' else {'align_corners': True}
    sp = F.interpolate(sp.float(), size=(H_out * p, W_out * p), mode=mode, **kwargs).to(x.dtype)
    out = patchify(sp, p)
    return out if batched else out.squeeze(0)


class EasyDict:

    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def log_state(state):
    result = []
    
    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    
    return '\n'.join(result)

def get_flexible_mask_and_ratio(model_kwargs: dict, x: torch.Tensor):
    '''
    sequential case (fit): 
        x: (B, N, C)
        model_kwargs: {y: (B,), mask: (B, N), grid: (B, 2, N)}
        mask: (B, N) -> (B, 1, N)
    spatial case (dit):
        x: (B, C, H, W)
        model_kwargs: {y: (B,)}
        mask: (B, C) -> (B, C, 1, 1)
    '''
    mask = model_kwargs.get('mask', torch.ones(x.shape[:2]))    # (B, N) or (B, C)
    ratio = float(mask.shape[-1]) / torch.count_nonzero(mask, dim=-1)  # (B,)
    if len(x.shape) == 3:               # sequential x: (B, N, C)
        mask = mask[..., None]         # (B, N) -> (B, N, 1)
    elif len(x.shape) == 4:             # spatial x: (B, C, H, W)
        mask = mask[..., None, None]    # (B, C) -> (B, C, 1, 1)
    else:
        raise NotImplementedError
    return mask.to(x), ratio.to(x)
    