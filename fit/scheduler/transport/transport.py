import torch
import torch.nn.functional as F
import numpy as np

import enum
from einops import rearrange

from . import path
from .utils import mean_flat, get_flexible_mask_and_ratio
from .integrators import ode, sde

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class SNRType(enum.Enum):
    UNIFORM = enum.auto()
    LOGNORM = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        snr_type,
        multires_loss: str = 'A',
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.snr_type = snr_type
        self.multires_loss = multires_loss  # 'A' = velocity loss, 'B' = upsample loss

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = torch.tensor(z.size())
        N = torch.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - torch.sum(x ** 2) / 2.
        return torch.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = torch.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        
        if self.snr_type == SNRType.UNIFORM:
            t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif self.snr_type == SNRType.LOGNORM:
            u = torch.normal(mean=0.0, std=1.0, size=(x1.shape[0],))
            t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")
        
        t = t.to(x1)
        return t, x0, x1
    
    def _forward_packed(self, model, x1, model_kwargs):
        """Run the packed (sequence-packed multi-image) forward pass.

        Returns (xt, ut, model_output, t_expanded, sigma_t, x0).
        """
        doc_ids = model_kwargs['doc_ids']
        B = x1.shape[0]
        n_pack_per_elem = model_kwargs['n_pack']          # (B,)
        max_n_pack = int(n_pack_per_elem.max())

        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if self.snr_type == SNRType.UNIFORM:
            t_per_image = torch.rand((B, max_n_pack), device=x1.device) * (t1 - t0) + t0
        elif self.snr_type == SNRType.LOGNORM:
            u = torch.normal(mean=0.0, std=1.0, size=(B, max_n_pack), device=x1.device)
            t_per_image = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown snr type: {self.snr_type}")
        t_per_image = t_per_image.to(x1)                  # (B, max_n_pack)

        # Expand t to per-token: each token gets the t of its image.
        safe_ids = doc_ids.clamp(min=0)                   # (B, N_total)
        t_per_token = t_per_image[
            torch.arange(B, device=x1.device)[:, None], safe_ids
        ]                                                  # (B, N_total)
        # Zero t for padding tokens so they don't affect xt/ut computation.
        t_per_token = t_per_token * (doc_ids >= 0).to(x1)

        x0 = torch.randn_like(x1)

        # Compute xt and ut using per-token t.
        # compute_mu_t / compute_xt / compute_ut all call expand_t_like_x
        # which does t.view(B, 1) — we bypass that by expanding ourselves.
        t_expanded = t_per_token.unsqueeze(-1)             # (B, N_total, 1)
        alpha_t    = t_expanded                            # ICPlan: alpha_t = t
        sigma_t    = 1 - t_expanded                        # ICPlan: sigma_t = 1-t
        xt = alpha_t * x1 + sigma_t * x0
        ut = x1 - x0                                      # ICPlan: d_alpha=1, d_sigma=-1

        # Pass per-image t (not per-token) to the model for timestep embedding.
        # n_pack is consumed by the transport layer; strip it before forwarding.
        model_kwargs_fwd = {k: v for k, v in model_kwargs.items() if k != 'n_pack'}
        model_output = model(xt, t_per_image, **model_kwargs_fwd)
        return xt, ut, model_output, t_expanded, sigma_t, x0

    def _mean_per_image(self, sq_err, doc_ids, model_kwargs, x1):
        """Average squared error per image, then average over images.

        Unpacked (doc_ids is None): apply mask, mean_flat over (N, C) then * ratio,
        which already normalises by the number of valid tokens per image.

        Packed (doc_ids is not None): for each image in each pack slot we sum
        the per-token errors and divide by that image's token count, giving
        every image equal weight regardless of resolution.
        """
        if doc_ids is None:
            mask_b, ratio = get_flexible_mask_and_ratio(model_kwargs, x1)
            return mean_flat(sq_err * mask_b) * ratio

        # sq_err: (B, N_total, C)  doc_ids: (B, N_total)  values: image-index or -1 for padding
        n_pack_per_elem = model_kwargs['n_pack']          # (B,)
        max_n_pack = int(n_pack_per_elem.max())
        B = x1.shape[0]
        device = x1.device

        # Mean over C dim first → (B, N_total)
        sq_err_mean_c = sq_err.mean(dim=-1)

        # Accumulate sum and count per image slot using scatter_add
        safe_ids = doc_ids.clamp(min=0)                   # (B, N_total)
        valid = (doc_ids >= 0).float()                     # (B, N_total)

        token_sum   = torch.zeros(B, max_n_pack, device=device, dtype=sq_err.dtype)
        token_count = torch.zeros(B, max_n_pack, device=device, dtype=sq_err.dtype)
        token_sum.scatter_add_(1, safe_ids, sq_err_mean_c * valid)
        token_count.scatter_add_(1, safe_ids, valid)

        # per-image mean; guard against empty slots (count == 0)
        per_image = token_sum / token_count.clamp(min=1)   # (B, max_n_pack)

        # Average over the valid image slots in each batch element
        n_images = n_pack_per_elem.float().to(device)      # (B,)
        slot_mask = (torch.arange(max_n_pack, device=device)[None] < n_images[:, None]).float()
        loss_per_batch = (per_image * slot_mask).sum(dim=1) / n_images  # (B,)
        return loss_per_batch

    def _loss_a(self, x1, x0, xt, ut, t, model_output,
                doc_ids, t_expanded, sigma_t, model_kwargs):
        """Standard velocity / noise / score loss (all resolutions, packed and unpacked)."""
        terms = {}
        if self.model_type == ModelType.VELOCITY:
            sq_err = (model_output - ut) ** 2
            terms['loss'] = self._mean_per_image(sq_err, doc_ids, model_kwargs, x1)
        else:
            if doc_ids is not None:
                # ICPlan-specific: alpha_t=t, sigma_t=1-t, d_alpha=1, d_sigma=-1.
                # Only ICPlan (LINEAR path) is supported in packed mode.
                assert hasattr(self.path_sampler, 'sigma') and not hasattr(self.path_sampler, 'sigma_min'), \
                    "Packed mode with non-VELOCITY model type only supports ICPlan (LINEAR path)."
                drift_var_expanded = t_expanded
                sigma_t_val = sigma_t
            else:
                _, drift_var_expanded = self.path_sampler.compute_drift(xt, t)
                sigma_t_val, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var_expanded / sigma_t_val) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var_expanded / (sigma_t_val ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                sq_err = weight * (model_output - x0) ** 2
            else:
                sq_err = weight * (model_output * sigma_t_val + x0) ** 2
            terms['loss'] = self._mean_per_image(sq_err, doc_ids, model_kwargs, x1)
        return terms

    def _loss_b(self, x1_fullres, xt, t, model_output, model_kwargs):
        """Loss B: predict clean low-res, upsample, compare to full-res target."""
        p = 2; C_in = 4
        # For ICPlan: x1_hat = xt + (1-t) * v_pred.
        _sigma = 1.0 - t
        x1_hat = xt + _sigma.view(-1, 1, 1) * model_output   # (B, N_lr, 16)
        size_lr = model_kwargs['size']                        # (B, 1, 2)
        size_fr = model_kwargs['size_fullres']                # (B, 1, 2)
        H_lr = int(size_lr[0, 0, 0]); W_lr = int(size_lr[0, 0, 1])
        H_fr = int(size_fr[0, 0, 0]); W_fr = int(size_fr[0, 0, 1])
        # Unpatchify → bilinear upsample → re-patchify
        x1_sp = rearrange(x1_hat, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                          h=H_lr, w=W_lr, p1=p, p2=p, c=C_in)
        x1_up = F.interpolate(x1_sp.float(), size=(H_fr * p, W_fr * p),
                              mode='bilinear', align_corners=True).to(x1_hat.dtype)
        x1_hat_fr = rearrange(x1_up, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                              p1=p, p2=p)                    # (B, N_fr, 16)
        mask_fr, ratio_fr = get_flexible_mask_and_ratio(
            {'mask': model_kwargs['mask_fullres']}, x1_fullres
        )
        # Correction factor 1/sigma^2 makes this equivalent to the standard
        # velocity loss: ||x1_hat - x1||^2 = sigma^2 * ||v_pred - v*||^2.
        correction = 1.0 / (_sigma ** 2).mean()
        return {'loss': mean_flat(((x1_hat_fr - x1_fullres) * mask_fr) ** 2) * ratio_fr * correction}

    def _loss_c(self, model, x1_fullres, model_kwargs):
        """Loss C: low-res FiT → ResNet → full-res velocity.

        Fully self-contained: does its own sampling so xt_lr is always the
        bilinear downsample of xt_fr, eliminating any cross-resolution leakage.
        """
        p = 2; C_in = 4
        size_lr = model_kwargs['size']          # (B, 1, 2)
        size_fr = model_kwargs['size_fullres']  # (B, 1, 2)
        H_lr = int(size_lr[0, 0, 0]); W_lr = int(size_lr[0, 0, 1])
        H_fr = int(size_fr[0, 0, 0]); W_fr = int(size_fr[0, 0, 1])

        # 1. Sample full-res noise and timestep
        t_c, x0_fr, _ = self.sample(x1_fullres)
        t_exp = t_c.view(-1, 1, 1)

        # 2. Full-res noisy sample (ICPlan: xt = t*x1 + (1-t)*x0)
        xt_fr = t_exp * x1_fullres + (1 - t_exp) * x0_fr    # (B, N_fr, 16)

        # 3. Low-res noisy sample by bilinearly downsampling xt_fr.
        # This makes xt_lr = bilinear_downsample(xt_fr) an exact identity,
        # matching inference and eliminating any exploitable statistical leakage.
        xt_fr_sp = rearrange(xt_fr, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                             h=H_fr, w=W_fr, p1=p, p2=p, c=C_in)
        xt_lr_sp = F.interpolate(xt_fr_sp.float(), size=(H_lr * p, W_lr * p),
                                 mode='bilinear', align_corners=True).to(xt_fr.dtype)
        xt_lr = rearrange(xt_lr_sp, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p, p2=p)                         # (B, N_lr, 16)

        # 4. FiT transformer at low resolution
        lr_kwargs = {k: v for k, v in model_kwargs.items()
                     if k not in ('x1_fullres', 'mask_fullres', 'size_fullres')}
        model_out_lr = model(xt_lr, t_c, **lr_kwargs)        # (B, N_lr, 16)

        # 5. Recover predicted clean latent and convert to spatial
        x1_lr_hat = xt_lr + (1 - t_exp) * model_out_lr      # (B, N_lr, 16)
        x1_lr_sp  = rearrange(x1_lr_hat,
                              'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                              h=H_lr, w=W_lr, p1=p, p2=p, c=C_in)

        # 6. Bilinear upsample to full-res spatial size
        x1_lr_up = F.interpolate(x1_lr_sp.float(), size=(H_fr * p, W_fr * p),
                                 mode='bilinear', align_corners=True).to(xt_lr.dtype)

        # 7. ResNet predicts full-res velocity (spatial)
        v_fr_sp = model.upsampler(x1_lr_up, xt_fr_sp)        # (B, 4, sp_fr, sp_fr)

        # 8. Re-patchify to token sequence
        v_fr = rearrange(v_fr_sp, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                         p1=p, p2=p)                          # (B, N_fr, 16)

        # 9. Velocity target and MSE loss over valid full-res tokens
        ut_fr = x1_fullres - x0_fr                           # (B, N_fr, 16)
        mask_fr, ratio_fr = get_flexible_mask_and_ratio(
            {'mask': model_kwargs['mask_fullres']}, x1_fullres
        )
        return {'loss': mean_flat(((v_fr - ut_fr) * mask_fr) ** 2) * ratio_fr}

    def training_losses(self, model, x1, model_kwargs=None):
        """Loss for training the score model.
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs is None:
            model_kwargs = {}

        doc_ids  = model_kwargs.get('doc_ids', None)
        x1_fullres = model_kwargs.get('x1_fullres', None)

        # Loss C is fully self-contained (its own sampling, no shared forward pass).
        if self.multires_loss == 'C' and x1_fullres is not None and doc_ids is None:
            terms = self._loss_c(model, x1_fullres, model_kwargs)
            terms['pred'] = None
            return terms

        # Run the appropriate forward pass.
        t_expanded = sigma_t = None
        if doc_ids is not None:
            xt, ut, model_output, t_expanded, sigma_t, x0 = self._forward_packed(model, x1, model_kwargs)
            t = None
        else:
            t, x0, x1 = self.sample(x1)
            t, xt, ut = self.path_sampler.plan(t, x0, x1)
            model_output = model(xt, t, **model_kwargs)

        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {'pred': model_output}

        if self.multires_loss == 'B' and x1_fullres is not None and doc_ids is None:
            terms.update(self._loss_b(x1_fullres, xt, t, model_output, model_kwargs))
        else:
            terms.update(self._loss_a(x1, x0, xt, ut, t, model_output,
                                      doc_ids, t_expanded, sigma_t, model_kwargs))

        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = torch.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, torch.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = torch.randint(2, x.size(), dtype=torch.float, device=x.device) * 2 - 1
            t = torch.ones_like(t) * (1 - t)
            with torch.enable_grad():
                x.requires_grad = True
                grad = torch.autograd.grad(torch.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = torch.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = torch.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn