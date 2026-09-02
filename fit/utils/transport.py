import torch

from fit.utils.utils import mean_flat, get_flexible_mask_and_ratio

# Floor for the quadtree loss's noise weight; see _loss_quadtree.
QUADTREE_WEIGHT_EPS = 1e-3


class Transport:
    """Linear-path flow matching with a velocity-predicting model.

    The path is the linear interpolant xt = t*x1 + (1-t)*x0, whose velocity
    field is the constant ut = x1 - x0. Timesteps are drawn lognormally,
    t = sigmoid(u) with u ~ N(0, 1), over the full [0, 1] interval — the linear
    path needs no epsilon margin since its coefficients stay finite at both ends.

    The upstream VP/GVP paths and the noise-/score-prediction objectives have
    been removed; this fork only ever trains velocity on the linear path.
    """

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        x0 = torch.randn_like(x1)
        t = torch.sigmoid(torch.randn((x1.shape[0],)))
        t = t.to(x1)
        return t, x0, x1


    def _forward_packed(self, model, x1, model_kwargs):
        """Run the packed (sequence-packed multi-image) forward pass.

        Returns (xt, ut, model_output).
        """
        doc_ids = model_kwargs['doc_ids']
        B = x1.shape[0]
        n_pack_per_elem = model_kwargs['n_pack']          # (B,)
        max_n_pack = int(n_pack_per_elem.max())

        t_per_image = torch.sigmoid(
            torch.randn((B, max_n_pack), device=x1.device)
        ).to(x1)                                          # (B, max_n_pack)

        # Expand t to per-token: each token gets the t of its image.
        safe_ids = doc_ids.clamp(min=0).long()            # (B, N_total)
        t_per_token = t_per_image[
            torch.arange(B, device=x1.device)[:, None], safe_ids
        ]                                                  # (B, N_total)
        # Zero t for padding tokens so they don't affect xt/ut computation.
        t_per_token = t_per_token * (doc_ids >= 0).to(x1)

        x0 = torch.randn_like(x1)

        # Linear path with per-token t (rather than the usual per-sample t).
        t_expanded = t_per_token.unsqueeze(-1)             # (B, N_total, 1)
        xt = t_expanded * x1 + (1 - t_expanded) * x0
        ut = x1 - x0

        # Pass per-image t (not per-token) to the model for timestep embedding.
        # n_pack is consumed by the transport layer; strip it before forwarding.
        model_kwargs_fwd = {k: v for k, v in model_kwargs.items() if k != 'n_pack'}
        model_output = model(xt, t_per_image, **model_kwargs_fwd)
        return xt, ut, model_output

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
        safe_ids = doc_ids.clamp(min=0).long()            # (B, N_total)
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

    def _loss_a(self, x1, ut, model_output, doc_ids, model_kwargs):
        """Velocity loss (all resolutions, packed and unpacked)."""
        sq_err = (model_output - ut) ** 2
        return {'loss': self._mean_per_image(sq_err, doc_ids, model_kwargs, x1)}

    def training_losses(self, model, x1, model_kwargs=None):
        """Loss for training the score model.
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs is None:
            model_kwargs = {}

        doc_ids = model_kwargs.get('doc_ids', None)

        # Upsampler path: the dataset has already noised both resolutions and
        # supplied the full-res velocity target, so this collapses to a plain
        # velocity loss at full resolution.
        if 'feature_fullres' in model_kwargs:
            return self._loss_upsampler(model, x1, model_kwargs)

        # Quadtree path: the dataset noised the latent *before* compression (the
        # tree structure depends on the noise draw), so we cannot re-noise here.
        # The model input x1 is the already-noisy compressed feature, and the
        # clean x0 compressed on the same tree is supplied as `target`. The model
        # predicts the clean tokens; the velocity objective is recovered by a
        # 1/(1-t)^2 weight (see _loss_quadtree).
        if 'target' in model_kwargs:
            return self._loss_quadtree(model, x1, model_kwargs)

        # Run the appropriate forward pass.
        if doc_ids is not None:
            xt, ut, model_output = self._forward_packed(model, x1, model_kwargs)
        else:
            t, x0, x1 = self.sample(x1)
            t_expanded = t.view(t.size(0), *([1] * (x1.dim() - 1)))
            xt = t_expanded * x1 + (1 - t_expanded) * x0
            ut = x1 - x0
            model_output = model(xt, t, **model_kwargs)

        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {'pred': model_output}
        terms.update(self._loss_a(x1, ut, model_output, doc_ids, model_kwargs))

        return terms

    def _loss_upsampler(self, model, xt_lr, model_kwargs):
        """Upsampler velocity loss (merged Loss C).

        The dataset noised both resolutions from one consistent noise family and
        shares a single timestep per image, so here we only forward the model
        and compare its dense full-res prediction to the full-res velocity
        target ut_fr = x1_fr - x0_fr.

        x1 (the positional arg) is the packed low-res noisy input xt_lr.
        """
        # The model is conditioned on the resolution-adjusted timestep t_model
        # (not the shared image t): the low-res input xt_lr was built at the
        # effective noise level sigma_inj = 1 - t_model, matching the inference
        # sampler's noise rescale. See in1k_latent_dataset for the derivation.
        t_model = model_kwargs['t_model']           # (1, n_pack)
        ut_fr = model_kwargs['ut_fullres']          # (n_pack, N_fr, 16)
        mask_fr = model_kwargs['mask_fullres']      # (n_pack, N_fr)

        # Map dataset fields to the model's forward signature and strip the
        # loss-only / transport-only keys.
        fwd = {k: v for k, v in model_kwargs.items()
               if k not in ('ut_fullres', 'n_pack', 't', 't_model', 'feature_fullres')}
        fwd['x_fullres'] = model_kwargs['feature_fullres']

        v_fr = model(xt_lr, t_model, **fwd)         # (n_pack, N_fr, 16)

        mask = mask_fr[..., None].to(v_fr.dtype)
        sq_err = ((v_fr - ut_fr) * mask) ** 2
        # Per-image mean over valid elements, then mean over images. The
        # denominator must count tokens * channels (not just tokens): sq_err is
        # summed over both the token and the 16-channel axis, so the mask is
        # broadcast to the channel dim before summing. Counting tokens only here
        # under-counts by a factor of C (=16), inflating the loss 16x.
        denom = mask.expand_as(sq_err).sum(dim=(1, 2)).clamp(min=1)
        per_image = sq_err.sum(dim=(1, 2)) / denom  # (n_pack,)
        return {'loss': per_image.mean().unsqueeze(0), 'pred': v_fr}

    def _loss_quadtree(self, model, feature, model_kwargs):
        """Clean-target (x1-prediction) loss for the packed quadtree path.

        The dataset already noised the latent and compressed it; `feature` is the
        packed *noisy* tokens (xt) and `target` is the *clean* tokens (x1) on the
        same tree. The model predicts x1 directly (so the upsampling stack never
        has to reproduce high-frequency noise). We recover the velocity objective
        in expectation with a time-dependent weight:

            xt = t·x1 + (1-t)·x0,   v = x1 - x0 = (x1 - xt)/(1-t)
            a model predicting x̂1 implies v̂ = (x̂1 - xt)/(1-t), hence
            ‖v̂ - v‖² = ‖x̂1 - x1‖² / (1-t)²

        so the velocity loss equals a clean-target MSE weighted by 1/(1-t)².

        Timestep convention: the dataset stores t_ds with t_ds=0 clean, t_ds=1
        noise (xt = (1-t_ds)·x1 + t_ds·x0). We convert to the transport's linear-path
        convention t = 1 - t_ds (t=0 noise, t=1 clean) so x1 is weighted by t and
        the noise weight is (1-t) = t_ds. The model is conditioned on the same
        transport-convention t. As t→1 (clean) the weight 1/(1-t)² diverges, so we
        clamp (1-t) with QUADTREE_WEIGHT_EPS.
        """
        target = model_kwargs['target']                  # (1, N, 4C) clean x1
        doc_ids = model_kwargs['doc_ids']
        t_ds = model_kwargs['t']                          # (1, n_pack) dataset t
        t = 1.0 - t_ds                                    # transport convention

        # Forward: model predicts x1 from the noisy feature, conditioned on t.
        fwd = {k: v for k, v in model_kwargs.items()
               if k not in ('target', 'n_pack', 't')}
        x1_hat = model(feature, t, **fwd)                 # (1, N, 4C)

        # Per-token velocity weight 1/(1-t)² = 1/t_ds², expanded from per-image t.
        # The weight diverges as t_ds→0 (clean end), so floor the noise weight.
        eps = QUADTREE_WEIGHT_EPS
        B = feature.shape[0]
        safe_ids = doc_ids.clamp(min=0).long()            # (1, N)
        t_ds_tok = t_ds[torch.arange(B, device=feature.device)[:, None], safe_ids]
        noise_w = t_ds_tok.clamp(min=eps)                 # (1, N)  == (1 - t)
        weight = (1.0 / (noise_w ** 2)).unsqueeze(-1)     # (1, N, 1)

        sq_err = weight * (x1_hat - target) ** 2          # (1, N, 4C)
        loss = self._mean_per_image(sq_err, doc_ids, model_kwargs, feature)
        return {'loss': loss, 'pred': x1_hat}

    def loss_quadtree_dense(self, model, compressor, selection, packed):
        """Clean-target loss taken at FULL latent resolution (pyramid-decoder path).

        Same 1/(1-t)^2 velocity re-weighting as `_loss_quadtree`, but the model's
        token output is handed to the compressor's PyramidDecoder and compared
        against the dense clean latent x0 instead of mean-pooled leaf tokens.
        Every latent pixel is supervised, so coarse leaves are no longer trained
        only on their own average.

        selection : the dict from QuadtreePlanPool (x_t, x0, t, plans, ...).
        packed    : the dict from PredictiveVarianceCompressor.forward.
        """
        t_ds = packed['t']                                # (1, n_pack) dataset t
        t = 1.0 - t_ds                                    # transport convention

        fwd = dict(y=packed['label'].to(torch.int).clamp(min=0),
                   grid=packed['grid'], mask=packed['mask'].float(),
                   size=packed['size'], tsize=packed['tsize'],
                   doc_ids=packed['doc_ids'].long(),
                   block_mask=packed['block_mask'])
        tokens = model(packed['feature'], t, **fwd)       # (1, N, D)

        # Routed through __call__ (not .decode_packed) so the decoder's forward
        # runs inside DDP's wrapper: under DDP the bare method is not exposed,
        # and bypassing the wrapper breaks gradient synchronisation. See
        # PredictiveVarianceCompressor.forward.
        x_hat = compressor(
            tokens, selection['plans'], packed['counts'], selection['x_t'],
            mode='decode')

        # Per-IMAGE velocity weight 1/(1-t)^2 == 1/t_ds^2, floored at the clean end.
        noise_w = t_ds[0].clamp(min=QUADTREE_WEIGHT_EPS)  # (n_pack,)
        weight = (1.0 / (noise_w ** 2)).view(-1, 1, 1, 1)

        sq_err = weight * (x_hat - selection['x0']) ** 2  # (B, C, H, W)
        # Mean over each image, then over images: equal weight per image.
        loss = sq_err.flatten(1).mean(dim=1).mean()
        return {'loss': loss.reshape(1), 'pred': x_hat}
