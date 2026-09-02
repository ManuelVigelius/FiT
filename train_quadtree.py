"""Training entry point for the quadtree-compressed FiT model.

A focused sibling of ``train_fitv2.py`` for the variance-guided quadtree pipeline.
The training *core* (Accelerate setup, optimizer, EMA, checkpointing, logging,
transport loss) mirrors the base trainer, but the data path is fundamentally
different and cannot reuse the base sampler/DataLoader:

  * The quadtree packing is GPU-side and *needs a model*: a frozen
    :class:`VariancePredictor` decides each image's quadtree structure. The
    packer (``INQuadtreeLatentLoader``) runs it, compresses the noisy latent (and
    the clean latent on the same tree, for the target), and emits packed B=1
    sequences. There is no fixed-length sampler to prepare with Accelerate.

  * The diffusion target is clean-x1 prediction with a 1/(1-t)^2 weight (see
    ``Transport._loss_quadtree``), keyed by the ``target`` field the packer emits.
    The model input is the already-noisy compressed ``feature``.

Model kwargs threaded to the network: ``y, grid, mask, size, tsize, doc_ids,
block_mask`` plus the transport-only ``target, t, n_pack``.
"""

import os
import contextlib
import shutil
import time
import logging
import datetime
import argparse

import torch
import diffusers
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from copy import deepcopy

from fit.utils.transport import Transport
from fit.utils.utils import (
    instantiate_from_config, default, get_obj_from_str, update_ema, init_from_ckpt,
)
from fit.utils.lr_scheduler import get_scheduler

logger = get_logger(__name__, log_level="INFO")


def _pooled_targets(selection, packed):
    """Mean-pooled clean-x0 tokens for the token-space (non-pyramid) loss.

    The learned encoder no longer produces pooled leaf values, so the target for
    the direct-readout path is built here: pool x0 over each leaf's N x N region
    on the SAME tree the plan chose, patchified 2x2 exactly like the tokens.

    Returns (1, N_total, 4*C) aligned with packed['feature'].
    """
    import torch.nn.functional as _F
    x0 = selection['x0']                       # (B, C, H, W)
    C = x0.shape[1]
    device = x0.device
    N = packed['feature'].shape[1]
    out = torch.zeros(1, N, 4 * C, device=device, dtype=x0.dtype)

    off = 0
    for i, (plan, n_i) in enumerate(zip(selection['plans'], packed['counts'])):
        sizes = plan['sizes']
        pos = plan['positions']
        # Pool once per distinct leaf size present, then gather each token's 2x2.
        pooled = {}
        for n in sizes.unique().tolist():
            pooled[int(n)] = (x0[i] if n == 1 else
                              _F.avg_pool2d(x0[i][None].float(), int(n),
                                            stride=int(n))[0])
        for k in range(n_i):
            n = int(sizes[k])
            cy, cx = float(pos[k, 0]), float(pos[k, 1])
            ly = int(round((cy - n) / n))       # top-left leaf index on the n-grid
            lx = int(round((cx - n) / n))
            block = pooled[n][:, ly:ly + 2, lx:lx + 2]        # (C, 2, 2)
            out[0, off + k] = block.permute(1, 2, 0).reshape(4 * C).to(out.dtype)
        off += n_i
    return out


def resolve_tuple(*args):
    return tuple(args)
OmegaConf.register_new_resolver("tuple", resolve_tuple)


# Which parameters the pretrained checkpoint actually supplies. Everything else
# is new (random or zero init) and is what a warmup phase should train.
#
#   model       the checkpoint is a plain FiT, so the trunk (blocks, t_embedder,
#               y_embedder, global_adaLN_modulation) loads. `size_embedder` has no
#               counterpart there, and `final_layer` is either excluded via
#               ignore_keys (use_pyramid_decoder rebuilds it at hidden_size) or
#               loaded with the WRONG target — the pretrained head predicts
#               velocity, this trainer supervises clean x0 — so it is treated as
#               new in both wirings.
#
#   compressor  only `encoder.base_proj` is seeded (from x_embedder.proj) and,
#               when load_base_head is on, `decoder.base_head`. The pyramid
#               itself is entirely new. res_gamma is zero-init, so at step 0 the
#               pyramids are no-ops and the warmup starts from exactly the
#               baseline tokenisation.
DEFAULT_WARMUP_TRAINABLE = ['size_embedder', 'final_layer']


def _freeze_pretrained(model, compressor, accelerator, spec, seeded_base_head):
    """Freeze pretrained weights, leaving only the new layers trainable.

    Returns the list of (module_label, param_name) pairs left trainable.

    `spec` is the raw --freeze_pretrained value. 'default'/'' selects
    DEFAULT_WARMUP_TRAINABLE on the model plus every compressor parameter that
    was not seeded from the checkpoint; anything else is a comma-separated list
    of substrings matched against both modules.
    """
    raw = (spec or '').strip()
    custom = None if (not raw or raw == 'default') else [
        s.strip() for s in raw.split(',') if s.strip()]

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_comp = accelerator.unwrap_model(compressor)

    # ---- model -----------------------------------------------------------
    model_names = custom if custom is not None else DEFAULT_WARMUP_TRAINABLE
    unwrapped_model.finetune(type='partial', unfreeze=model_names)

    # ---- compressor ------------------------------------------------------
    # No finetune() here, and the default rule is a complement rather than a
    # match: train everything EXCEPT the seeded projections.
    seeded = ['encoder.base_proj']
    if seeded_base_head:
        seeded.append('decoder.base_head')
    for name, p in unwrapped_comp.named_parameters():
        if custom is not None:
            p.requires_grad = any(s in name for s in custom)
        else:
            p.requires_grad = not any(name.startswith(s) for s in seeded)

    trainable = ([('model', n) for n, p in unwrapped_model.named_parameters() if p.requires_grad]
                 + [('compressor', n) for n, p in unwrapped_comp.named_parameters() if p.requires_grad])
    return trainable


def _raw(module, accelerator):
    """Strip both wrappers: DDP (unwrap_model) and torch.compile (_orig_mod).

    accelerate's unwrap_model only removes the DDP shell; torch.compile leaves an
    OptimizedModule whose state_dict keys carry a "_orig_mod." prefix the saved
    checkpoint does not have.
    """
    m = accelerator.unwrap_model(module)
    return getattr(m, '_orig_mod', m)


def _load_weights_only(ckpt_dir, model, compressor, ema_model, accelerator, use_ema):
    """Restore model / compressor / EMA weights from a checkpoint, nothing else.

    accelerate's save_state writes one file per prepared model, numbered by the
    order prepare_model was called in: here model, then ema_model (only when EMA
    is on), then compressor. So the compressor's index shifts with --use_ema, and
    is derived rather than hardcoded.
    """
    from pathlib import Path
    from accelerate.checkpointing import load_model as _load_model

    ckpt = Path(ckpt_dir)

    def _file(idx):
        stem = "model" if idx == 0 else f"model_{idx}"
        safe, binf = ckpt / f"{stem}.safetensors", ckpt / f"pytorch_{stem}.bin"
        if safe.exists():
            return safe
        if binf.exists():
            return binf
        raise FileNotFoundError(
            f"reset_optimizer: no weights for prepared model #{idx} in {ckpt}. "
            f"Expected {safe.name} or {binf.name}. If the warmup ran with a "
            f"different --use_ema setting the file numbering shifts; re-run the "
            f"warmup and the full stage with the same EMA flag.")

    comp_idx = 2 if use_ema else 1

    raw_model = _raw(model, accelerator)
    _load_model(raw_model, str(_file(0)), device=str(accelerator.device))
    logger.info(f"  model      <- {_file(0).name}")

    raw_comp = _raw(compressor, accelerator)
    _load_model(raw_comp, str(_file(comp_idx)), device=str(accelerator.device))
    logger.info(f"  compressor <- {_file(comp_idx).name}")

    # Re-seed EMA from the warmup weights. The deepcopy that built ema_model ran
    # before this load and still holds the pretrained weights, so copying the
    # freshly-loaded model over it is what makes EMA start from the warmup point
    # rather than from the pretrained checkpoint.
    if use_ema and ema_model is not None:
        _raw(ema_model, accelerator).load_state_dict(raw_model.state_dict())
        logger.info("  ema        <- model weights (re-seeded from warmup)")


def parse_args():
    parser = argparse.ArgumentParser(description="Quadtree FiT training.")
    parser.add_argument("--project_name", type=str, const=True, default="", nargs="?")
    parser.add_argument("--main_project_name", type=str, default="image_generation")
    parser.add_argument("--workdir", type=str, default="workdir")
    parser.add_argument("--cfgdir", nargs="*", default=list())
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default='latest')
    parser.add_argument("--load_model_from_checkpoint", type=str, default=None)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--freeze_pretrained",
        type=str,
        default=None,
        metavar="LAYERS",
        help=(
            "Warmup phase: freeze the pretrained weights and train only the newly "
            "added ones. Pass 'default' (or an empty string) to use the built-in "
            "list, which is derived from what the checkpoint actually supplies: "
            "on the model, everything except size_embedder and final_layer is "
            "pretrained; on the compressor, everything except encoder.base_proj "
            "(and decoder.base_head when seeded) is new. Or pass a comma-separated "
            "list of name substrings to keep trainable instead. Names are matched "
            "against BOTH the model and the compressor, since the two share one "
            "optimizer here."
        ),
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        default=False,
        help=(
            "Load only model / compressor / EMA weights from the latest checkpoint, "
            "discarding optimizer and scheduler state and resetting the step counter. "
            "Use this when transitioning from a --freeze_pretrained warmup run to "
            "full training, so the optimizer is freshly initialised over all params."
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def _build_variance_predictor(cfg, device):
    """Instantiate the frozen VariancePredictor that drives quadtree packing.

    Config block (data.params.train.variance_predictor):
        target: fit.quadtree_compression.variance_prediction.VariancePredictor
        params: { latent_channels, region_sizes, ... }
        ckpt:   path to a trained VariancePredictor checkpoint (.pt/.safetensors/.bin)
    """
    vp = instantiate_from_config(cfg).to(device)
    ckpt = getattr(cfg, 'ckpt', None)
    if ckpt is not None:
        # The predictor state dict is stored flat; init_from_ckpt handles both
        # accelerate dirs and plain files and strips wrapper prefixes.
        init_from_ckpt(vp, os.path.abspath(ckpt), ignore_keys=None, verbose=True)
    else:
        logger.warning("No variance_predictor.ckpt set — using randomly-initialised "
                       "VariancePredictor. Quadtree structure will be meaningless.")
    vp.eval()
    for p in vp.parameters():
        p.requires_grad = False
    return vp


def main():
    args = parse_args()

    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not args.project_name:
        raise ValueError("--project_name is required")
    project_name = args.project_name
    workdirnow = os.path.join(args.workdir, project_name)
    os.makedirs(workdirnow, exist_ok=True)
    cfgdir = os.path.join(workdirnow, "configs")
    ckptdir = os.path.join(workdirnow, "checkpoints")
    logging_dir = os.path.join(workdirnow, "logs")
    imagedir = os.path.join(workdirnow, "images")
    for d in (cfgdir, ckptdir, logging_dir, imagedir):
        os.makedirs(d, exist_ok=True)

    configs = [OmegaConf.load(c) for c in args.cfgdir]
    config = OmegaConf.merge(*configs)
    accelerate_cfg = config.accelerate
    diffusion_cfg = config.diffusion
    data_cfg = config.data
    grad_accu_steps = accelerate_cfg.gradient_accumulation_steps

    accelerator_project_cfg = ProjectConfiguration(project_dir=workdirnow, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accu_steps,
        mixed_precision=accelerate_cfg.mixed_precision,
        log_with=getattr(accelerate_cfg, 'logger', 'wandb'),
        project_config=accelerator_project_cfg,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        fh = logging.FileHandler(os.path.join(logging_dir, project_name+"_"+datenow+".log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        fh.setLevel(logging.INFO)
        logger.logger.addHandler(fh)
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- LR scaling (token-budget based, like the packed path) --------------
    max_tokens = getattr(data_cfg.params.train, 'max_tokens', 1024)
    if args.scale_lr:
        tokens_per_step = max_tokens * grad_accu_steps * accelerator.num_processes
        base_tokens = accelerate_cfg.learning_rate_base_batch_size
        learning_rate = accelerate_cfg.learning_rate * tokens_per_step / base_tokens
    else:
        learning_rate = accelerate_cfg.learning_rate

    # ---- Diffusion model ----------------------------------------------------
    model = instantiate_from_config(diffusion_cfg.network_config).to(device=device)
    if args.load_model_from_checkpoint:
        ckpt_path = os.path.abspath(args.load_model_from_checkpoint)
        if os.path.isdir(ckpt_path):
            bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
            if not os.path.exists(bin_path):
                cands = [f for f in os.listdir(ckpt_path) if f.endswith((".bin", ".safetensors"))]
                bin_path = os.path.join(ckpt_path, sorted(cands)[0])
            ckpt_path = bin_path
        logger.info(f"Loading model weights from {ckpt_path}")
        init_from_ckpt(model, ckpt_path, ignore_keys=None, verbose=True)

    if args.use_ema:
        ema_model = deepcopy(model.module if hasattr(model, 'module') else model).to(device=device)
        for p in ema_model.parameters():
            p.requires_grad = False

    model = accelerator.prepare_model(model, device_placement=False)
    if args.use_ema:
        ema_model = accelerator.prepare_model(ema_model, device_placement=False)

    torch._inductor.config.joint_graph_constant_folding = False
    torch._dynamo.config.optimize_ddp = False

    # Persistent reductions keep the whole reduction axis in shared memory. With
    # this model's sequence lengths that can exceed what the GPU has, and Inductor
    # has no smaller fallback config to fall back TO — every candidate is rejected
    # and compilation dies at the BACKWARD pass with:
    #
    #   RuntimeError: No valid triton configs. OutOfResources: out of resource:
    #   shared memory, Required: 102144, Hardware limit: 101376
    #
    # (Seen on the 99KB-shared-memory lab nodes; it overshot by 768 bytes.)
    # Turning them off makes Inductor emit looped reductions instead, whose
    # shared-memory use does not scale with the reduction length. Slightly slower
    # kernels, but they compile. Set FIT_PERSISTENT_REDUCTIONS=1 to restore the
    # default on hardware with more shared memory.
    if os.environ.get("FIT_PERSISTENT_REDUCTIONS", "0") != "1":
        torch._inductor.config.triton.persistent_reductions = False
        logger.info("Inductor persistent_reductions disabled (shared-memory cap). "
                    "Set FIT_PERSISTENT_REDUCTIONS=1 to re-enable.")

    # torch.compile can be disabled entirely as an escape hatch: FIT_COMPILE=0.
    # Training is correct either way — this only affects speed — so it is the
    # first thing to try when a run dies inside inductor/triton rather than in
    # model code.
    if os.environ.get("FIT_COMPILE", "1") != "0":
        model = torch.compile(model, dynamic=True, mode="default")
        if args.use_ema:
            ema_model = torch.compile(ema_model, dynamic=True, mode="default")
    else:
        logger.info("torch.compile disabled via FIT_COMPILE=0 (eager mode).")

    transport = Transport()

    # ---- Frozen variance predictor + quadtree packer ------------------------
    # Optional: the ground-truth-variance loader
    # (fit.data.in1k_gt_quadtree_latent_dataset) decides the quadtree from the
    # clean latent CPU-side and has no predictor block. Its `train_iter` accepts
    # and ignores `vp`, so passing None is the whole difference.
    vp_cfg = getattr(data_cfg.params.train, 'variance_predictor', None)
    vp = _build_variance_predictor(vp_cfg, device) if vp_cfg is not None else None
    if vp is None:
        logger.info("No variance_predictor block — quadtree structure comes from "
                    "the dataset (ground-truth variance).")

    # ---- Learned compressor (PyramidEncoder [+ PyramidDecoder]) -------------
    # Trained jointly with the diffusion model. It runs on exactly the images the
    # plan pool selected for each sequence, so its gradients always belong to the
    # loss computed in the same step.
    use_pyramid_decoder = bool(getattr(
        diffusion_cfg.network_config.params, 'use_pyramid_decoder', False))
    compressor = instantiate_from_config(diffusion_cfg.compressor_config).to(device=device)
    compressor = accelerator.prepare_model(compressor, device_placement=False)
    logger.info(f"Compressor params: "
                f"{sum(p.numel() for p in compressor.parameters())/1e6:.2f}M  "
                f"(pyramid_decoder={use_pyramid_decoder})")

    logger.info("Building quadtree dataset/packer...")
    loader = instantiate_from_config(data_cfg)
    train_len = loader.train_len()
    logger.info(f"Dataset built ({train_len} feature files).")

    # ---- Warmup: freeze the pretrained weights ------------------------------
    # Must run BEFORE the optimizer is built so only trainable params get slots
    # (and so AdamW never allocates state for frozen tensors).
    if args.freeze_pretrained is not None:
        seeded_base_head = bool(getattr(
            diffusion_cfg.compressor_config.params, 'load_base_head', False))
        trainable_names = _freeze_pretrained(
            model, compressor, accelerator, args.freeze_pretrained, seeded_base_head)
        if accelerator.is_main_process:
            n_model = sum(1 for m, _ in trainable_names if m == 'model')
            n_comp = len(trainable_names) - n_model
            logger.info(f"freeze_pretrained: {n_model} model + {n_comp} compressor "
                        f"tensors trainable")
            for mod, n in trainable_names:
                logger.info(f"  [{mod}] {n}")

    # ---- Optimizer / scheduler ----------------------------------------------
    # The compressor trains jointly with the diffusion model, so it shares the
    # optimizer and the grad-norm clip.
    trainable_params = ([p for p in model.parameters() if p.requires_grad]
                        + [p for p in compressor.parameters() if p.requires_grad])
    optimizer_cfg = default(accelerate_cfg.optimizer, {"target": "torch.optim.AdamW"})
    optimizer = get_obj_from_str(optimizer_cfg["target"])(
        trainable_params, lr=learning_rate, **optimizer_cfg.get("params", dict())
    )
    lr_scheduler = get_scheduler(
        accelerate_cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=accelerate_cfg.lr_warmup_steps,
        num_training_steps=accelerate_cfg.max_train_steps,
    )
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    if accelerator.is_main_process and getattr(accelerate_cfg, 'logger', 'wandb') is not None:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), workdirnow)
        accelerator.init_trackers(
            project_name=args.main_project_name,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
            init_kwargs={"wandb": {"group": args.project_name}},
        )

    # ---- Resume -------------------------------------------------------------
    global_steps = 0
    resume_from_path = None
    if args.resume_from_checkpoint and args.resume_from_checkpoint.lower() != "none":
        if args.resume_from_checkpoint != "latest":
            resume_from_path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(ckptdir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from_path = dirs[-1] if dirs else None
        if resume_from_path is None:
            logger.info("No checkpoint to resume; starting fresh.")
        elif args.reset_optimizer:
            # Weights-only resume, for warmup -> full training. Restores model,
            # compressor and EMA but drops optimizer / scheduler state, and leaves
            # the step counter at 0 so the LR warmup runs again over all params.
            ckpt_full_path = os.path.join(ckptdir, resume_from_path)
            logger.info(f"reset_optimizer: loading weights only from {resume_from_path}, "
                        f"resetting step counter.")
            _load_weights_only(ckpt_full_path, model, compressor,
                               ema_model if args.use_ema else None,
                               accelerator, args.use_ema)
        else:
            global_steps = int(resume_from_path.split("-")[1])
            logger.info(f"Resuming from step {global_steps}: {resume_from_path}")
            accelerator.load_state(os.path.join(ckptdir, resume_from_path))

    OmegaConf.save(config=config, f=os.path.join(cfgdir, "config.yaml"))

    logger.info("***** Running quadtree training *****")
    logger.info(f"  Feature files = {train_len}")
    logger.info(f"  max_tokens per seq = {max_tokens}")
    logger.info(f"  Learning rate = {learning_rate}")
    logger.info(f"  Grad accumulation = {grad_accu_steps}")
    logger.info(f"  Total steps = {accelerate_cfg.max_train_steps}")

    progress_bar = tqdm(range(0, accelerate_cfg.max_train_steps),
                        disable=not accelerator.is_main_process)
    progress_bar.set_description("Optim Steps")
    progress_bar.update(global_steps)
    if args.use_ema:
        ema_model.eval()

    model.train()
    compressor.train()
    train_loss = None

    # The packer is an epoch-scoped iterator (its raw loader is a per-epoch shard).
    # Loop epochs until we hit max_train_steps.
    epoch = 0
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    stop = False
    while not stop:
        packed_iter = loader.train_iter(vp, epoch, rank=rank, world_size=world_size,
                                        device=device)
        for selection in packed_iter:
            with accelerator.accumulate(model):
                # ---- learned compression (INSIDE the accumulate block) -------
                # The plan pool already sized this image batch to the token
                # budget using the FROZEN predictor, so every image whose encoder
                # forward runs here also contributes to this step's loss. That is
                # what keeps the compressor's gradients from being stranded by
                # the zero_grad below. The batch size varies step to step.
                packed = compressor(selection['x_t'], selection['plans'],
                                    selection['label'], selection['t'],
                                    x0=selection['x0'])

                di = packed['doc_ids'].long()

                # FlexAttention block mask, built outside the compiled model.
                from torch.nn.attention.flex_attention import create_block_mask
                _di = di
                def doc_mask_mod(b, h, q_idx, kv_idx):
                    return _di[b, q_idx] == _di[b, kv_idx]
                N_bm = di.shape[1]
                block_mask = create_block_mask(doc_mask_mod, 1, None, N_bm, N_bm, device=di.device)

                with accelerator.autocast():
                    if use_pyramid_decoder:
                        # Full-resolution loss: model tokens -> PyramidDecoder ->
                        # dense latent, compared against the clean x0.
                        packed['block_mask'] = block_mask
                        loss_dict = transport.loss_quadtree_dense(
                            model, compressor, selection, packed)
                    else:
                        # Token-space loss against the mean-pooled x0 tokens.
                        model_kwargs = dict(
                            y=packed['label'].to(torch.int).clamp(min=0),
                            grid=packed['grid'], mask=packed['mask'].float(),
                            size=packed['size'], tsize=packed['tsize'], doc_ids=di,
                            block_mask=block_mask,
                            target=_pooled_targets(selection, packed),
                            t=packed['t'], n_pack=packed['n_pack'],
                        )
                        loss_dict = transport.training_losses(
                            model, packed['feature'], model_kwargs)
                loss = loss_dict["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients and accelerate_cfg.max_grad_norm > 0.:
                    all_norm = accelerator.clip_grad_norm_(trainable_params, accelerate_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = accelerator.gather(loss.repeat(1)).mean()
            avg_loss_scaled = avg_loss.detach() / grad_accu_steps
            train_loss = avg_loss_scaled if train_loss is None else train_loss + avg_loss_scaled

            if accelerator.sync_gradients:
                if args.use_ema:
                    update_ema(ema_model, model, args.ema_decay)
                progress_bar.update(1)
                global_steps += 1
                if getattr(accelerate_cfg, 'logger', 'wandb') is not None:
                    accelerator.log({"train_loss": train_loss.item()}, step=global_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)
                    if accelerate_cfg.max_grad_norm != 0.0:
                        accelerator.log({"grad_norm": all_norm.item()}, step=global_steps)
                train_loss = None

                if global_steps % accelerate_cfg.checkpointing_steps == 0:
                    _prune_checkpoints(ckptdir, accelerate_cfg, accelerator)
                    save_path = os.path.join(ckptdir, f"checkpoint-{global_steps}")
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    accelerator.wait_for_everyone()

                if global_steps in accelerate_cfg.checkpointing_steps_list:
                    save_path = os.path.join(ckptdir, f"save-checkpoint-{global_steps}")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()

            if global_steps % accelerate_cfg.logging_steps == 0:
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if accelerator.is_main_process:
                    logger.info(f"step={global_steps} / {accelerate_cfg.max_train_steps}, "
                                f"step_loss={logs['step_loss']}, lr={logs['lr']}")

            if global_steps >= accelerate_cfg.max_train_steps:
                stop = True
                break
        epoch += 1

    accelerator.wait_for_everyone()
    accelerator.end_training()


def _prune_checkpoints(ckptdir, accelerate_cfg, accelerator):
    if accelerate_cfg.checkpoints_total_limit is None:
        return
    checkpoints = [d for d in os.listdir(ckptdir) if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if accelerator.is_main_process and len(checkpoints) >= accelerate_cfg.checkpoints_total_limit:
        num_to_remove = len(checkpoints) - accelerate_cfg.checkpoints_total_limit + 1
        for rm in checkpoints[:num_to_remove]:
            shutil.rmtree(os.path.join(ckptdir, rm))


if __name__ == "__main__":
    main()
