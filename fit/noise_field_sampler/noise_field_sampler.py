import torch

from .area_resampling import area_resample_nn_correction
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
