import importlib
import re
from collections import OrderedDict
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange

from safetensors.torch import load_file


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    if hasattr(ema_model, 'module'):
        ema_model = ema_model.module
    if hasattr(ema_model, '_orig_mod'):
        ema_model = ema_model._orig_mod
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def init_from_ckpt(
    model, checkpoint_dir, ignore_keys=None, verbose=False
) -> None:
    if checkpoint_dir.endswith(".safetensors"):
        try:
            model_state_dict=load_file(checkpoint_dir)
        except: # 历史遗留问题，千万别删
            model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    else:
        model_state_dict=torch.load(checkpoint_dir,  map_location="cpu")
    model_new_ckpt=dict()
    for i in model_state_dict.keys():
        model_new_ckpt[i] = model_state_dict[i]
    keys = list(model_new_ckpt.keys())
    for k in keys:
        if ignore_keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    print("Deleting key {} from state_dict.".format(k))
                    del model_new_ckpt[k]
    missing, unexpected = model.load_state_dict(model_new_ckpt, strict=False)
    if verbose:
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    if verbose:
        print("")


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
    if H == H_out and W == W_out:
        return x
    
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
    sp = unpatchify(x, (H * p, W * p), p)
    kwargs = {} if mode == 'area' else {'align_corners': True}
    sp = F.interpolate(sp.float(), size=(H_out * p, W_out * p), mode=mode, **kwargs).to(x.dtype)
    out = patchify(sp, p)
    return out if batched else out.squeeze(0)

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

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
