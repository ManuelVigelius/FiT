import os
import contextlib
import torch
import argparse
import datetime
import time
import logging
import shutil
import torch
import diffusers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from copy import deepcopy
from fit.scheduler.transport import create_transport
from fit.utils.utils import (
    instantiate_from_config,
    default,
    get_obj_from_str,
    update_ema
)
from fit.utils.lr_scheduler import get_scheduler
from fit.utils.eval_utils import init_from_ckpt

logger = get_logger(__name__, log_level="INFO")

# For Omegaconf Tuple
def resolve_tuple(*args):
    return tuple(args)
OmegaConf.register_new_resolver("tuple", resolve_tuple)

def parse_args():
    parser = argparse.ArgumentParser(description="Argument.")
    parser.add_argument(
        "--project_name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="if setting, the logdir will be like: project_name",
    )
    parser.add_argument(
        "--main_project_name",
        type=str,
        default="image_generation",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="workdir",
        help="workdir",
    )
    parser.add_argument( # if resume, you change it none. i will load from the resumedir
        "--cfgdir",
        nargs="*",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_model_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be loaded from a pretrained model checkpoint."
            "Or you can set diffusion.pretrained_model_path in Config for loading!!!"
        ),
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="The decay rate for ema."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--freeze_new_layers",
        type=str,
        default=None,
        metavar="LAYERS",
        help=(
            "Comma-separated list of layer name substrings to keep trainable while "
            "freezing everything else (warmup phase). "
            "Examples: --freeze_new_layers size_embedder  (Loss A warmup) "
            "          --freeze_new_layers size_embedder,upsampler  (Loss C warmup)"
        ),
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        default=False,
        help=(
            "Load only model and EMA weights from the latest checkpoint, discarding "
            "optimizer and scheduler state. Use this when transitioning from a warmup "
            "run to full training so the optimizer is freshly initialised for all params."
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()
    
    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = None
    workdir = None
    workdirnow = None
    cfgdir = None
    ckptdir = None
    logging_dir = None
    imagedir = None
    
    if args.project_name:
        project_name = args.project_name
        if os.path.exists(os.path.join(args.workdir, project_name)): #open resume
            workdir=os.path.join(args.workdir, project_name)
        else: # new a workdir
            workdir = os.path.join(args.workdir, project_name)
            # if accelerator.is_main_process:
            os.makedirs(workdir, exist_ok=True)
        workdirnow = workdir

        cfgdir = os.path.join(workdirnow, "configs")
        ckptdir = os.path.join(workdirnow, "checkpoints")
        logging_dir = os.path.join(workdirnow, "logs")
        imagedir = os.path.join(workdirnow, "images")

        # if accelerator.is_main_process:
        os.makedirs(cfgdir, exist_ok=True)
        os.makedirs(ckptdir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(imagedir, exist_ok=True)
    if args.cfgdir:
        load_cfgdir = args.cfgdir
    
    # setup config
    configs_list = load_cfgdir # read config from a config dir
    configs = [OmegaConf.load(cfg) for cfg in configs_list]
    config = OmegaConf.merge(*configs)
    accelerate_cfg = config.accelerate
    diffusion_cfg = config.diffusion
    data_cfg = config.data
    grad_accu_steps = accelerate_cfg.gradient_accumulation_steps
    
    accelerator_project_cfg = ProjectConfiguration(project_dir=workdirnow, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accu_steps,
        mixed_precision=accelerate_cfg.mixed_precision,
        log_with=getattr(accelerate_cfg, 'logger', 'wandb'),
        project_config=accelerator_project_cfg,
    )
    device = accelerator.device
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        File_handler = logging.FileHandler(os.path.join(logging_dir, project_name+"_"+datenow+".log"), encoding="utf-8")
        File_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        File_handler.setLevel(logging.INFO)
        logger.logger.addHandler(File_handler)
        
        diffusers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32: # for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    packed_cfg = getattr(data_cfg.params.train, 'packed', None)
    use_packed = packed_cfg is not None and getattr(packed_cfg, 'enabled', False)

    if args.scale_lr:
        if use_packed:
            # In packed mode, tokens per step is the meaningful unit, not samples.
            # Base token count = mean tokens per sample under the resize distribution * base_batch_size.
            # Valid grids are even values in [min_g, max_g]; each contributes g^2 tokens.
            resize_range = getattr(data_cfg.params.train, 'resize_range', None)
            if resize_range is not None:
                min_g, max_g = resize_range
                valid_grids = torch.arange(min_g, max_g + 1, 2)
                mean_tokens_per_sample = (valid_grids ** 2).float().mean().item()
            else:
                mean_tokens_per_sample = data_cfg.params.train.target_len
            learning_rate_base_tokens = mean_tokens_per_sample * accelerate_cfg.learning_rate_base_batch_size
            tokens_per_step = (
                packed_cfg.max_tokens *
                grad_accu_steps *
                accelerator.num_processes
            )
            learning_rate = (
                accelerate_cfg.learning_rate *
                tokens_per_step / learning_rate_base_tokens
            )
        else:
            learning_rate = (
                accelerate_cfg.learning_rate *
                grad_accu_steps *
                data_cfg.params.train.loader.batch_size *   # local batch size per device
                accelerator.num_processes / accelerate_cfg.learning_rate_base_batch_size    # global batch size
            )
    else:
        learning_rate = accelerate_cfg.learning_rate


    model = instantiate_from_config(diffusion_cfg.network_config).to(device=device)
    if args.load_model_from_checkpoint:
        # Load model weights from an accelerate checkpoint directory or a plain
        # .bin/.safetensors file, overriding whatever pretrain_ckpt was set in config.
        ckpt_path = os.path.abspath(args.load_model_from_checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"--load_model_from_checkpoint path does not exist: {ckpt_path}"
            )
        if os.path.isdir(ckpt_path):
            # Accelerate saves the unwrapped model as pytorch_model.bin inside the dir.
            bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
            if not os.path.exists(bin_path):
                # Sharded saves use a different name; find the first shard.
                candidates = [f for f in os.listdir(ckpt_path) if f.endswith(".bin") or f.endswith(".safetensors")]
                if not candidates:
                    raise FileNotFoundError(f"No model weights found in {ckpt_path}")
                bin_path = os.path.join(ckpt_path, sorted(candidates)[0])
            ckpt_path = bin_path
        logger.info(f"Loading model weights from {ckpt_path}")
        init_from_ckpt(model, ckpt_path, ignore_keys=None, verbose=True)
    # update ema
    if args.use_ema:
        # ema_dtype = torch.float32
        if hasattr(model, 'module'):
            ema_model = deepcopy(model.module).to(device=device)
        else:
            ema_model = deepcopy(model).to(device=device)
        for p in ema_model.parameters():
            p.requires_grad = False

    if args.use_ema:
        model = accelerator.prepare_model(model, device_placement=False)
        ema_model = accelerator.prepare_model(ema_model, device_placement=False)
    else:
        model = accelerator.prepare_model(model, device_placement=False)

    # joint_graph_constant_folding=False works around a bug in stable_topological_sort
    # (pattern_matcher.py:2291) present in PyTorch 2.10 nightlies where the joint
    # forward+backward graph partition pass produces a cycle the sort can't resolve.
    torch._inductor.config.joint_graph_constant_folding = False
    model = torch.compile(model, dynamic=True, mode="default")
    if args.use_ema:
        ema_model = torch.compile(ema_model, dynamic=True, mode="default")

    # In SiT, we use transport instead of diffusion
    transport = create_transport(**OmegaConf.to_container(diffusion_cfg.transport))  # default: velocity; 
    # schedule_sampler = create_named_schedule_sampler()

    # Setup Dataloader
    # In packed mode the effective batch size per step is determined by max_tokens,
    # not by loader.batch_size (which is ignored).  We still use loader.batch_size
    # as the global-sampler unit so get_train_sampler draws enough indices.
    # (packed_cfg and use_packed are already defined above for LR scaling.)
    loader_batch_size = data_cfg.params.train.loader.batch_size
    total_batch_size = loader_batch_size * accelerator.num_processes * grad_accu_steps

    # In packed mode the sampler pre-allocates max_steps * global_batch_size flat
    # indices. global_batch_size must reflect how many *samples* are consumed per
    # optimizer step, not just loader.batch_size (which is 1 packed sequence).
    # Estimate: (max_tokens * num_processes * grad_accu_steps) / mean_tokens_per_sample.
    if use_packed:
        _resize_range = getattr(data_cfg.params.train, 'resize_range', None)
        if _resize_range is not None:
            _min_g, _max_g = _resize_range
            _valid = torch.arange(_min_g, _max_g + 1)
            _mean_toks = (_valid ** 2).float().mean().item()
        else:
            _mean_toks = data_cfg.params.train.target_len
        _tokens_per_step = (
            packed_cfg.max_tokens * grad_accu_steps * accelerator.num_processes
        )
        sampler_batch_size = max(1, round(_tokens_per_step / _mean_toks))
    else:
        sampler_batch_size = total_batch_size

    global_steps = 0
    if args.resume_from_checkpoint and args.resume_from_checkpoint.lower() != "none":
        # normal read with safety check
        if args.resume_from_checkpoint != "latest":
            resume_from_path = os.path.basename(args.resume_from_checkpoint)
        else:   # Get the most recent checkpoint
            dirs = os.listdir(ckptdir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            resume_from_path = dirs[-1] if len(dirs) > 0 else None

        if resume_from_path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            if args.reset_optimizer:
                # Weights-only resume: step counter stays at 0 so optimizer, scheduler,
                # and dataloader all start fresh. Used when continuing from a warmup run.
                logger.info(f"reset_optimizer: loading weights from {resume_from_path}, resetting step counter.")
            else:
                global_steps = int(resume_from_path.split("-")[1]) # gs not calculate the gradient_accumulation_steps
                logger.info(f"Resuming from steps: {global_steps}")

    get_train_dataloader = instantiate_from_config(data_cfg)
    train_len = get_train_dataloader.train_len()
    train_dataloader = get_train_dataloader.train_dataloader(
        global_batch_size=sampler_batch_size, max_steps=accelerate_cfg.max_train_steps,
        resume_step=global_steps, seed=args.seed,
        packed=use_packed,
        max_tokens=getattr(packed_cfg, 'max_tokens', 512) if use_packed else 512,
        pad_to_multiple=getattr(packed_cfg, 'pad_to_multiple', 128) if use_packed else 128,
    )

    # Warmup phase: freeze everything except the newly-added layers.
    # Run this before building the optimizer so only trainable params are included.
    if args.freeze_new_layers is not None:
        raw = args.freeze_new_layers.strip()
        if not raw or raw == 'default':
            new_layer_names = ['size_embedder', 'upsampler']
        else:
            new_layer_names = [s.strip() for s in raw.split(',') if s.strip()]
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.finetune(type='partial', unfreeze=new_layer_names)
        if accelerator.is_main_process:
            trainable = [n for n, p in unwrapped.named_parameters() if p.requires_grad]
            logger.info(f"freeze_new_layers: training only {trainable}")

    # Setup optimizer and lr_scheduler
    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_cfg = default(
        accelerate_cfg.optimizer, {"target": "torch.optim.AdamW"}
    )
    optimizer = get_obj_from_str(optimizer_cfg["target"])(
        params, lr=learning_rate, **optimizer_cfg.get("params", dict())
    )
    lr_scheduler = get_scheduler(
        accelerate_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=accelerate_cfg.lr_warmup_steps,
        num_training_steps=accelerate_cfg.max_train_steps,
    )
    
    # Prepare Accelerate
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and getattr(accelerate_cfg, 'logger', 'wandb') != None:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), workdirnow)
        accelerator.init_trackers(
            project_name=args.main_project_name, 
            config=config, 
            init_kwargs={"wandb": {"group": args.project_name}}
        )
    
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {train_len}")
    logger.info(f"  Instantaneous batch size per device = {data_cfg.params.train.loader.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Learning rate = {learning_rate}")
    logger.info(f"  Gradient Accumulation steps = {grad_accu_steps}")
    logger.info(f"  Total optimization steps = {accelerate_cfg.max_train_steps}")
    logger.info(f"  Current optimization steps = {global_steps}")
    logger.info(f"  Train dataloader length = {len(train_dataloader)} ")
    logger.info(f"  Training Mixed-Precision = {accelerate_cfg.mixed_precision}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and args.resume_from_checkpoint.lower() != "none":
        ckpt_full_path = os.path.join(ckptdir, resume_from_path)
        if args.reset_optimizer:
            # Weights-only load: restore model + EMA weights but discard optimizer /
            # scheduler state so full training starts with a fresh optimizer over all params.
            from pathlib import Path
            ckpt_dir = Path(ckpt_full_path)
            unwrapped_model = accelerator.unwrap_model(model)
            model_file = ckpt_dir / "model.safetensors"
            if not model_file.exists():
                model_file = ckpt_dir / "pytorch_model.bin"
            logger.info(f"reset_optimizer: loading model weights from {model_file}")
            from accelerate.checkpointing import load_model as _load_model
            _load_model(unwrapped_model, str(model_file), device=str(accelerator.device))
            # Initialize EMA from the warmup model weights. The deepcopy above ran before
            # this load and still holds pretrained weights, so overwrite it here.
            if args.use_ema:
                unwrapped_ema = accelerator.unwrap_model(ema_model)
                unwrapped_ema.load_state_dict(unwrapped_model.state_dict())
        else:
            # Normal resume: restore model, EMA, optimizer, scheduler, and step counter.
            error_times=0
            while(True):
                if error_times >= 100:
                    raise
                try:
                    logger.info(f"Resuming from checkpoint {resume_from_path}")
                    accelerator.load_state(ckpt_full_path)
                    break
                except (RuntimeError, Exception) as err:
                    error_times+=1
                    if accelerator.is_local_main_process:
                        logger.warning(err)
                        logger.warning(f"Failed to resume from checkpoint {resume_from_path}")
                        shutil.rmtree(ckpt_full_path)
                    else:
                        time.sleep(2)
    
    # save config
    OmegaConf.save(config=config, f=os.path.join(cfgdir, "config.yaml"))
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(0, accelerate_cfg.max_train_steps), 
        disable = not accelerator.is_main_process
    )
    progress_bar.set_description("Optim Steps")
    progress_bar.update(global_steps)
    
    if args.use_ema:
        # ema_model = ema_model.to(ema_dtype)
        ema_model.eval()
    # Training Loop
    # Profiling config: capture steps [PROFILE_START, PROFILE_START + PROFILE_STEPS).
    # Set PROFILE_START high enough to be past torch.compile warmup (typically <50 steps).
    PROFILE_START = 20
    PROFILE_STEPS = 30
    _profiling_active = False

    model.train()
    train_loss = None  # accumulated as tensor to defer CPU-GPU sync to logging
    for step, batch in enumerate(train_dataloader, start=global_steps):
        # --- nsys profiler window ---
        if accelerator.is_main_process:
            if step == PROFILE_START:
                torch.cuda.cudart().cudaProfilerStart()
                _profiling_active = True
                logger.info(f"[profiler] cudaProfilerStart at step {step}")
            elif step == PROFILE_START + PROFILE_STEPS and _profiling_active:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                _profiling_active = False
                logger.info(f"[profiler] cudaProfilerStop at step {step}")

        with torch.cuda.nvtx.range(f"step_{step}") if _profiling_active else contextlib.nullcontext():
            with torch.cuda.nvtx.range("data_to_gpu") if _profiling_active else contextlib.nullcontext():
                for batch_key in batch.keys():
                    if not isinstance(batch[batch_key], list):
                        batch[batch_key] = batch[batch_key].to(device=device)
            x = batch['feature']        # (B, N, C)
            grid = batch['grid']        # (B, 2, N)
            mask = batch['mask']        # (B, N)
            y = batch['label']          # (B, 1) unpacked  or  (B, max_n_pack) packed
            size = batch['size']        # (B, N_pack, 2), order: h, w.
            doc_ids = batch.get('doc_ids', None)   # (B, N_total) or None in unpacked mode
            n_pack = batch.get('n_pack', None)     # (B,) or None in unpacked mode
            optimizer.zero_grad(set_to_none=True)
            with accelerator.accumulate(model):
                # No trimming: tensors are already padded to a 128-multiple by the
                # collate fn, so sequence length is always in {128, 256, 384, 512}.
                # Removing the trim eliminates the CPU/GPU sync that caused the
                # CUDAGraph partition, letting the full graph run as one unit.

                # prepare other parameters
                if doc_ids is not None:
                    # Packed mode: y is already (B, max_n_pack); keep as-is for the model.
                    # Replace padding slots (-1) with 0 so LabelEmbedder doesn't get OOB indices;
                    # those tokens are masked out and don't contribute to loss.
                    y = y.to(torch.int).clamp(min=0)
                else:
                    y = y.squeeze(dim=-1).to(torch.int)

                # Build the FlexAttention block mask outside the compiled model so
                # create_block_mask never appears inside an Inductor graph, which
                # caused BackendCompilerFailed on variable-length sequences.
                block_mask = None
                if doc_ids is not None:
                    from torch.nn.attention.flex_attention import create_block_mask
                    B_bm, N_bm = doc_ids.shape
                    _doc_ids = doc_ids
                    def doc_mask_mod(b, h, q_idx, kv_idx):
                        return _doc_ids[b, q_idx] == _doc_ids[b, kv_idx]
                    block_mask = create_block_mask(doc_mask_mod, B_bm, None, N_bm, N_bm, device=doc_ids.device)

                model_kwargs = dict(y=y, grid=grid.long(), mask=mask, size=size,
                                    doc_ids=doc_ids, n_pack=n_pack, block_mask=block_mask)
                if 'feature_fullres' in batch:
                    model_kwargs['x1_fullres']   = batch['feature_fullres'].to(device=device)
                    model_kwargs['mask_fullres'] = batch['mask_fullres'].to(device=device)
                    model_kwargs['size_fullres'] = batch['size_fullres'].to(device=device)
                    if 'doc_ids_fr' in batch:
                        model_kwargs['doc_ids_fr'] = batch['doc_ids_fr'].to(device=device)
                with torch.cuda.nvtx.range("forward") if _profiling_active else contextlib.nullcontext():
                    # forward model and compute loss
                    with accelerator.autocast():
                        loss_dict = transport.training_losses(model, x, model_kwargs)
                loss = loss_dict["loss"].mean()
                with torch.cuda.nvtx.range("backward") if _profiling_active else contextlib.nullcontext():
                    # Backpropagate
                    accelerator.backward(loss)
                if accelerator.sync_gradients and accelerate_cfg.max_grad_norm > 0.:
                    all_norm = accelerator.clip_grad_norm_(
                        model.parameters(), accelerate_cfg.max_grad_norm
                    )
                with torch.cuda.nvtx.range("optimizer") if _profiling_active else contextlib.nullcontext():
                    optimizer.step()
                lr_scheduler.step()
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(1)).mean()
            avg_loss_scaled = avg_loss.detach() / grad_accu_steps
            train_loss = avg_loss_scaled if train_loss is None else train_loss + avg_loss_scaled
            
        # Checks if the accelerator has performed an optimization step behind the scenes; Check gradient accumulation
        if accelerator.sync_gradients: 
            if args.use_ema:
                # update_ema(ema_model, deepcopy(model).type(ema_dtype), args.ema_decay)
                update_ema(ema_model, model, args.ema_decay)
                
            progress_bar.update(1)
            global_steps += 1
            if getattr(accelerate_cfg, 'logger', 'wandb') != None:
                accelerator.log({"train_loss": train_loss.item() if train_loss is not None else 0.0}, step=global_steps)
                accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)
                if accelerate_cfg.max_grad_norm != 0.0:
                    accelerator.log({"grad_norm": all_norm.item()}, step=global_steps)
            train_loss = None
            if global_steps % accelerate_cfg.checkpointing_steps == 0:
                if accelerate_cfg.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(ckptdir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if accelerator.is_main_process and len(checkpoints) >= accelerate_cfg.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - accelerate_cfg.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(ckptdir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(ckptdir, f"checkpoint-{global_steps}")
                if accelerator.is_main_process:
                    os.makedirs(save_path)
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
                
            if global_steps in accelerate_cfg.checkpointing_steps_list:
                save_path = os.path.join(ckptdir, f"save-checkpoint-{global_steps}")
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
            
        if global_steps % accelerate_cfg.logging_steps == 0:
            logs = {"step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                logger.info("step="+str(global_steps)+" / total_step="+str(accelerate_cfg.max_train_steps)+", step_loss="+str(logs["step_loss"])+', lr='+str(logs["lr"]))

        if global_steps >= accelerate_cfg.max_train_steps:
            logger.info(f'global step ({global_steps}) >= max_train_steps ({accelerate_cfg.max_train_steps}), stop training!!!')
            break
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()