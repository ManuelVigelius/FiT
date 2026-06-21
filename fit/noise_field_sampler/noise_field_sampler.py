import torch

from .area_resampling import area_resample_nn_correction, area_resample_ref
from .noise_field_generator import sample_noise_fields_2d


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
        x_fr      = (1 - sigma) * (x_0_hat / t) + sigma * noise_F     (full-res noisy)
        x_lr      = downsample(x_0_hat, k);  combined with noise_k the same way
        v_fr      = model(x_lr [low-res cond], x_fr [full-res], t_model=1-sigma)
        x_0_hat  += dt * (noise_F + v_fr)

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

        # Full-res noisy input: convex-combine the (unit-scale) clean estimate
        # with the full-res noise. Loss C shares one timestep across both
        # resolutions, so no per-resolution noise rescaling here (the schedule
        # size is a conditioning resolution, not the integration resolution).
        if float(t) > 0.0:
            x0_unit = x_0_hat / t
            x_fr = (1.0 - sigma) * x0_unit + sigma * noise_fr
            # Low-res input: downsample the *clean estimate* to k×k, then add the
            # matching low-res noise from the consistent family.
            x0_lr = area_resample_ref(x0_unit, cur_size, cur_size)
            x_lr = (1.0 - sigma) * x0_lr + sigma * noise_cur
        else:
            x_fr = noise_fr
            x_lr = noise_cur

        t_model = float(t)  # Loss C uses the shared image timestep directly
        t_batch = torch.full((b,), t_model, device=device, dtype=dtype)
        v_fr = model(x_lr, x_fr, t_batch, cur_size)

        x_0_hat = x_0_hat + dt * (noise_fr + v_fr)

    return x_0_hat
