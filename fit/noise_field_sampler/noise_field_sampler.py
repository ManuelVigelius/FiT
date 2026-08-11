import torch

from .area_resampling import area_resample_nn_correction, area_resample_ref
from .noise_field_generator import sample_noise_fields_2d


def _coarsen_window(field_fr, y0, x0, h, w, k_h, k_w):
    """Area-consistent coarsening of a full-res noise window to (k_h, k_w).

    Restricts `field_fr` (b, d, H_fr, W_fr) to the full-res window
    [y0:y0+h, x0:x0+w] and area-resamples it to (k_h, k_w). Area resampling
    *averages* (weights sum to 1), reducing per-pixel std by 1/sqrt(block area),
    so we rescale by sqrt((h*w)/(k_h*k_w)) to restore unit per-pixel std. The
    result is the cross-resolution-consistent coarse noise for that window — the
    same property sample_noise_fields_2d guarantees globally (avg of a fine block
    == coarse / scale), here obtained directly from the full-res field so it
    works for arbitrary windows and non-square cells.
    """
    win = field_fr[:, :, y0:y0 + h, x0:x0 + w]
    coarse = area_resample_ref(win, k_h, k_w)
    scale = ((h * w) / (k_h * k_w)) ** 0.5
    return coarse * scale


@torch.no_grad()
def sample(
    model,
    scale_schedule: list[int],
    b: int,
    d: int,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    **model_kwargs,
) -> torch.Tensor:
    """Euler flow-matching sampler with progressive resolution upscaling.

    The persistent state is `x_0_hat`, the drift accumulator. With unit-std
    noise ε at full resolution, after Euler steps `x_t = ε + ∫ v dt` so we
    track `x_0_hat = x_t - sigma * ε = ∫ v dt + t * ε`. At convergence
    (sigma → 0) this equals x1 (the clean data).

    Per step at the current resolution k:
        sigma = 1 - t
        r = k / full_size
        sigma_inj = sigma * r / (1 - sigma * (1 - r))         (noise rescale)
        x = (x_0_hat / t) * (1 - sigma_inj) + sigma_inj * noise_k
        v = model(x, t_model = 1 - sigma_inj, **model_kwargs)
        x_0_hat += dt * (noise_k + v)

    Between steps where the resolution changes, x_0_hat is area-resampled to
    the new resolution. Noise is drawn once via sample_noise_fields_2d for the
    unique resolutions, so all per-resolution noise tensors are consistent.

    Args:
        model: callable (x, t, **kwargs) -> velocity prediction.
        scale_schedule: list of length num_steps with the spatial size for
            each Euler step.
        b: batch size.
        d: number of channels.
        t0, t1: integration interval; Euler goes t0 -> t1 in num_steps steps.
        device, dtype: device/dtype for the running state.

    Returns:
        Tensor of shape (b, d, scale_schedule[-1], scale_schedule[-1]).
    """
    num_steps = len(scale_schedule)
    assert num_steps >= 1, "scale_schedule must have at least one step"

    unique_sizes: list[int] = []
    for s in scale_schedule:
        if s not in unique_sizes:
            unique_sizes.append(s)
    noise_fields_list = sample_noise_fields_2d(unique_sizes, d, b)
    noise_fields = {
        k: nf.to(device=device, dtype=dtype)
        for k, nf in zip(unique_sizes, noise_fields_list)
    }

    ts = torch.linspace(t0, t1, num_steps + 1, device=device, dtype=dtype)
    full_size = max(scale_schedule)

    cur_size = scale_schedule[0]
    # x_0_hat = ∫v dt + t · noise; at t = t0 the integral is zero.
    x_0_hat = ts[0] * noise_fields[cur_size]

    for i in range(num_steps):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t

        sigma = 1.0 - t
        r = cur_size / full_size
        sigma_inj = sigma * r / (1.0 - sigma * (1.0 - r))
        t_model = 1.0 - sigma_inj

        noise_cur = noise_fields[cur_size]

        # Build the model input. x_0_hat has scale t (= 1 - sigma) at full res
        # — it's a sum of t / dt contributions — so divide by t to put it on
        # unit scale before convex-combining with noise. At t=0, x_0_hat=0 and
        # the data term vanishes; use the noise as the input directly.
        if float(t) > 0.0:
            x = (1.0 - sigma_inj) * (x_0_hat / t) + sigma_inj * noise_cur
        else:
            x = noise_cur

        t_batch = torch.full((b,), float(t_model), device=device, dtype=dtype)
        v = model(x, t_batch, **model_kwargs)

        x_0_hat = x_0_hat + dt * (noise_cur + v)

        if i + 1 < num_steps:
            next_size = scale_schedule[i + 1]
            if next_size != cur_size:
                x_0_hat = area_resample_nn_correction(x_0_hat, next_size, next_size)
                cur_size = next_size

    return x_0_hat


@torch.no_grad()
def sample_upsampler(
    model,
    scale_schedule: list[int],
    b: int,
    d: int,
    *,
    t0: float = 0.0,
    t1: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Euler flow-matching sampler for the learned-upsampler model (Loss C).

    Unlike :func:`sample`, the integration state ``x_0_hat`` stays at full
    resolution for the entire trajectory. The schedule size ``k`` only controls
    the resolution of the *low-res conditioning input* fed to the model: the
    model runs its pretrained block stack at ``k×k`` and its learned upsampler
    tail always reads out a *full-res* velocity, which updates the full-res
    state directly. (Contrast with :func:`sample`, where the state itself lives
    at the schedule resolution and is area-resampled between sizes.)

    Per step at schedule size k (full size F = max(scale_schedule)):
        sigma     = 1 - t
        r         = k / F
        sigma_inj = sigma * r / (1 - sigma * (1 - r))         (low-res noise rescale)
        t_model   = 1 - sigma_inj
        x_fr      = (1 - sigma) * (x_0_hat / t) + sigma * noise_F          (full-res, time t)
        x_lr      = (1 - sigma_inj) * down(x_0_hat, k) + sigma_inj * noise_k  (low-res, time t_model)
        v_fr      = model(x_lr [low-res cond], x_fr [full-res], t_model)
        x_0_hat  += dt * (noise_F + v_fr)

    The low-res branch is noised at the adjusted level sigma_inj (not sigma): the
    coarse noise field carries lower variance, so the model is conditioned on the
    resolution-adjusted t_model. This matches the Loss C training data.

    The low-res and full-res noise come from one cross-resolution-consistent
    family (drawn jointly via :func:`sample_noise_fields_2d`), mirroring how the
    dataset builds the upsampler training pair. ``model`` is a callable
    ``model(x_lr, x_fr, t, k) -> v_fr`` that handles all packing/patchifying.

    Returns:
        Tensor of shape (b, d, F, F).
    """
    num_steps = len(scale_schedule)
    assert num_steps >= 1, "scale_schedule must have at least one step"

    full_size = max(scale_schedule)

    # One consistent noise family over the unique schedule sizes *and* full-res,
    # so each low-res noise field is the cross-resolution-consistent coarsening
    # of the full-res field (same family the dataset trains on).
    unique_sizes: list[int] = []
    for s in list(scale_schedule) + [full_size]:
        if s not in unique_sizes:
            unique_sizes.append(s)
    noise_fields_list = sample_noise_fields_2d(unique_sizes, d, b)
    noise_fields = {
        k: nf.to(device=device, dtype=dtype)
        for k, nf in zip(unique_sizes, noise_fields_list)
    }
    noise_fr = noise_fields[full_size]

    ts = torch.linspace(t0, t1, num_steps + 1, device=device, dtype=dtype)

    # x_0_hat = ∫v dt + t · noise; at t = t0 the integral is zero. State is
    # full-res throughout.
    x_0_hat = ts[0] * noise_fr

    for i in range(num_steps):
        t = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t
        sigma = 1.0 - t

        cur_size = scale_schedule[i]
        noise_cur = noise_fields[cur_size]

        # The low-res noise field is a coarsening of the full-res one and so
        # carries a lower effective noise level: at global sigma the low-res
        # input actually sits at sigma_inj < sigma, i.e. a different time. Mirror
        # the noise rescale used in :func:`sample` (and matched by the Loss C
        # training data in in1k_latent_dataset):
        #   r = cur_size / full_size
        #   sigma_inj = sigma * r / (1 - sigma * (1 - r)),  t_model = 1 - sigma_inj.
        r = cur_size / full_size
        sigma_inj = sigma * r / (1.0 - sigma * (1.0 - r))
        t_model = 1.0 - sigma_inj

        if float(t) > 0.0:
            x0_unit = x_0_hat / t
            # Full-res noisy input at the global timestep t.
            x_fr = (1.0 - sigma) * x0_unit + sigma * noise_fr
            # Low-res input: downsample the *clean estimate* to k×k, then convex-
            # combine with the matching low-res noise at the *adjusted* level
            # sigma_inj (= 1 - t_model), consistent with how the model is
            # conditioned below.
            x0_lr = area_resample_ref(x0_unit, cur_size, cur_size)
            x_lr = (1.0 - sigma_inj) * x0_lr + sigma_inj * noise_cur
        else:
            x_fr = noise_fr
            x_lr = noise_cur

        # Condition the model on the resolution-adjusted low-res timestep.
        t_batch = torch.full((b,), float(t_model), device=device, dtype=dtype)
        v_fr = model(x_lr, x_fr, t_batch, cur_size)

        x_0_hat = x_0_hat + dt * (noise_fr + v_fr)

    return x_0_hat


@torch.no_grad()
def sample_upsampler_quadtree(
    model,
    qt,
    num_steps: int,
    b: int,
    d: int,
    *,
    patch_size: int = 1,
    per_leaf_sigma: bool = True,
    t0: float = 0.0,
    t1: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Mixed-resolution (quadtree) Euler sampler for the learned-upsampler model.

    Like :func:`sample_upsampler`, the integration state ``x_0_hat`` stays at full
    resolution (F = qt.H_fr, assumed square: qt.H_fr == qt.W_fr) for the whole
    trajectory. The difference: the low-res conditioning input is *non-uniform* —
    each quadtree cell carries its own resolution (k_h, k_w). Per step:

        x_fr      = (1 - sigma) * x0_unit + sigma * noise_F            (full-res, time t)
        for each cell c (window y0,x0,h,w at density k_h,k_w):
            r_c         = mean(k_h/h_full, k_w/w_full)   (cell coarsening ratio)
            sigma_inj_c = sigma * r_c / (1 - sigma * (1 - r_c))   [per-leaf]
                        = single global value                     [if per_leaf_sigma=False]
            x0_c        = downsample(x0_unit[window], k_h, k_w)
            noise_c     = area-consistent coarsening of noise_F[window] to (k_h,k_w)
            x_lr_c      = (1 - sigma_inj_c) * x0_c + sigma_inj_c * noise_c
        v_fr      = model(cell_blocks, x_fr, t_model, qt)            (full-res velocity)
        x_0_hat  += dt * (noise_F + v_fr)

    Per-token timestep is *not* implemented (a single scalar t_model is passed,
    derived from a representative full-res ratio r=1 -> t_model=t). With
    per_leaf_sigma=True each cell is noised at its own level while the model sees
    one t — a deliberate, flagged mismatch for the demo (set per_leaf_sigma=False
    to noise every cell at one global level instead).

    ``model`` is a callable ``model(cell_blocks, x_fr, t, qt) -> v_fr`` where
    ``cell_blocks`` is a list of (b, d, k_h*p, k_w*p)-equivalent spatial blocks
    (here returned as (b, d, k_h, k_w) latents; the caller patchifies). It
    handles all packing/patchifying and returns the full-res velocity (b, d, F, F).

    Returns:
        Tensor of shape (b, d, F, F).
    """
    assert qt.H_fr == qt.W_fr, "quadtree sampler assumes a square full-res frame"
    p = patch_size
    full_size = qt.H_fr * p          # spatial latent size (state/noise live here)

    # Full-res noise field. Cell noise is derived from it directly (area-consistent
    # coarsening per window), so the whole family stays cross-resolution-consistent
    # without enumerating per-cell sizes.
    noise_fr = sample_noise_fields_2d([full_size], d, b)[0].to(device=device, dtype=dtype)

    ts = torch.linspace(t0, t1, num_steps + 1, device=device, dtype=dtype)
    x_0_hat = ts[0] * noise_fr                       # full-res state throughout

    for i in range(num_steps):
        t = ts[i]
        dt = ts[i + 1] - t
        sigma = 1.0 - t

        # Single scalar t_model for the model (per-token timestep skipped).
        # TEMP: instead of the full-res ratio (r=1 -> t_model=t), use an
        # area-weighted effective ratio so the announced timestep matches the
        # cells' actual noise level. For a *uniform* quadtree every cell shares
        # one ratio r, so r_eff == r and t_model == 1 - sigma_inj exactly matches
        # the working sample_upsampler path. For mixed quadtrees this is a single
        # -scalar approximation (the real fix is a per-token timestep).
        total_area = sum(c.h * c.w for c in qt.cells)
        r_eff = sum((c.h * c.w) * (0.5 * (c.k_h / c.h + c.k_w / c.w))
                    for c in qt.cells) / total_area
        sigma_inj_eff = sigma * r_eff / (1.0 - sigma * (1.0 - r_eff))
        t_model = float(1.0 - sigma_inj_eff)


        if float(t) > 0.0:
            x0_unit = x_0_hat / t
            x_fr = (1.0 - sigma) * x0_unit + sigma * noise_fr
            cell_blocks = []
            for c in qt.cells:
                # Cell geometry is in grid (token) units; the latent state is
                # spatial (p× larger). Scale window + resolution to spatial.
                y0, x0, h, w = c.y0 * p, c.x0 * p, c.h * p, c.w * p
                kh, kw = c.k_h * p, c.k_w * p
                r_c = 0.5 * (c.k_h / c.h + c.k_w / c.w)
                if per_leaf_sigma:
                    sigma_inj = sigma * r_c / (1.0 - sigma * (1.0 - r_c))
                else:
                    sigma_inj = sigma
                x0_c = area_resample_ref(
                    x0_unit[:, :, y0:y0 + h, x0:x0 + w], kh, kw
                )
                noise_c = _coarsen_window(noise_fr, y0, x0, h, w, kh, kw)
                cell_blocks.append((1.0 - sigma_inj) * x0_c + sigma_inj * noise_c)
        else:
            x_fr = noise_fr
            cell_blocks = []
            for c in qt.cells:
                y0, x0, h, w = c.y0 * p, c.x0 * p, c.h * p, c.w * p
                kh, kw = c.k_h * p, c.k_w * p
                cell_blocks.append(_coarsen_window(noise_fr, y0, x0, h, w, kh, kw))

        t_batch = torch.full((b,), t_model, device=device, dtype=dtype)
        v_fr = model(cell_blocks, x_fr, t_batch, qt)

        x_0_hat = x_0_hat + dt * (noise_fr + v_fr)

    return x_0_hat
