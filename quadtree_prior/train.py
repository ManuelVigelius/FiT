"""Training entry point for the class-conditional quadtree structure prior.

Mirrors `train_quadtree.py`'s core (Accelerate, EMA, checkpointing, logging,
resume) but the task is plain next-token prediction over 17-position sequences,
so the data path is an ordinary fixed-shape DataLoader and the loss is a masked
cross-entropy — no packing, no transport, no compressor.

Run:
    accelerate launch -m quadtree_prior.train \
        --project_name quadtree_prior_s \
        --cfgdir configs/quadtree_prior/config_prior_s.yaml
"""

import argparse
import datetime
import logging
import os
import shutil
from copy import deepcopy

import diffusers
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (DistributedDataParallelKwargs,
                              ProjectConfiguration, set_seed)
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fit.utils.lr_scheduler import get_scheduler
from fit.utils.utils import (default, get_obj_from_str,
                             instantiate_from_config, init_from_ckpt, update_ema)
from quadtree_prior import structure as S
from quadtree_prior.model import structure_loss

logger = get_logger(__name__, log_level="INFO")


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Quadtree structure prior training.")
    parser.add_argument("--project_name", type=str, const=True, default="", nargs="?")
    parser.add_argument("--main_project_name", type=str, default="quadtree_prior")
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
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def _prune_checkpoints(ckptdir, accelerate_cfg, accelerator):
    """Keep at most `checkpoints_total_limit` rolling checkpoints."""
    limit = getattr(accelerate_cfg, 'checkpoints_total_limit', None)
    if not limit or not accelerator.is_main_process:
        return
    dirs = [d for d in os.listdir(ckptdir) if d.startswith("checkpoint-")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    if len(dirs) >= limit:
        for d in dirs[:len(dirs) - limit + 1]:
            shutil.rmtree(os.path.join(ckptdir, d), ignore_errors=True)


def main():
    args = parse_args()

    datenow = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not args.project_name:
        raise ValueError("--project_name is required")
    project_name = args.project_name
    workdirnow = os.path.join(args.workdir, project_name)
    cfgdir = os.path.join(workdirnow, "configs")
    ckptdir = os.path.join(workdirnow, "checkpoints")
    logging_dir = os.path.join(workdirnow, "logs")
    for d in (workdirnow, cfgdir, ckptdir, logging_dir):
        os.makedirs(d, exist_ok=True)

    configs = [OmegaConf.load(c) for c in args.cfgdir]
    config = OmegaConf.merge(*configs)
    accelerate_cfg = config.accelerate
    model_cfg = config.model
    data_cfg = config.data
    grad_accu_steps = accelerate_cfg.gradient_accumulation_steps

    accelerator_project_cfg = ProjectConfiguration(
        project_dir=workdirnow, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
        fh = logging.FileHandler(
            os.path.join(logging_dir, project_name + "_" + datenow + ".log"),
            encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
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

    # ---- LR scaling (plain per-sequence batch here, not a token budget) -----
    batch_size = getattr(data_cfg.params.train.loader, 'batch_size', 256)
    if args.scale_lr:
        total_batch = batch_size * grad_accu_steps * accelerator.num_processes
        base_batch = accelerate_cfg.learning_rate_base_batch_size
        learning_rate = accelerate_cfg.learning_rate * total_batch / base_batch
    else:
        learning_rate = accelerate_cfg.learning_rate

    # ---- Model --------------------------------------------------------------
    model = instantiate_from_config(model_cfg).to(device=device)
    logger.info(f"Prior params: "
                f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    if args.load_model_from_checkpoint:
        ckpt_path = os.path.abspath(args.load_model_from_checkpoint)
        if os.path.isdir(ckpt_path):
            bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
            if not os.path.exists(bin_path):
                cands = [f for f in os.listdir(ckpt_path)
                         if f.endswith((".bin", ".safetensors"))]
                bin_path = os.path.join(ckpt_path, sorted(cands)[0])
            ckpt_path = bin_path
        logger.info(f"Loading model weights from {ckpt_path}")
        init_from_ckpt(model, ckpt_path, ignore_keys=None, verbose=True)

    if args.use_ema:
        ema_model = deepcopy(model.module if hasattr(model, 'module') else model
                             ).to(device=device)
        for p in ema_model.parameters():
            p.requires_grad = False

    model = accelerator.prepare_model(model, device_placement=False)
    if args.use_ema:
        ema_model = accelerator.prepare_model(ema_model, device_placement=False)

    # ---- Data ---------------------------------------------------------------
    logger.info("Building structure dataset...")
    loader = instantiate_from_config(data_cfg)
    train_len = loader.train_len()
    logger.info(f"Dataset built ({train_len} feature files).")

    # ---- Optimizer / scheduler ---------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_cfg = default(accelerate_cfg.optimizer, {"target": "torch.optim.AdamW"})
    optimizer = get_obj_from_str(optimizer_cfg["target"])(
        trainable_params, lr=learning_rate, **optimizer_cfg.get("params", dict()))
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
        else:
            global_steps = int(resume_from_path.split("-")[1])
            logger.info(f"Resuming from step {global_steps}: {resume_from_path}")
            accelerator.load_state(os.path.join(ckptdir, resume_from_path))

    OmegaConf.save(config=config, f=os.path.join(cfgdir, "config.yaml"))

    logger.info("***** Running quadtree structure prior training *****")
    logger.info(f"  Feature files = {train_len}")
    logger.info(f"  Sequence length = {S.SEQ_LEN}")
    logger.info(f"  Batch size per process = {batch_size}")
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
    train_loss = None
    all_norm = torch.zeros((), device=device)

    epoch = 0
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    stop = False
    while not stop:
        for batch in loader.train_iter(epoch, rank=rank, world_size=world_size):
            with accelerator.accumulate(model):
                inputs = batch['inputs'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                # Files without a label fall back to the CFG "null" class, so an
                # unlabelled sample still trains the unconditional branch instead
                # of poisoning an arbitrary class.
                y = batch['label'].to(device, non_blocking=True).long()
                y = torch.where(y < 0, torch.full_like(y, model_num_classes(model)), y)

                with accelerator.autocast():
                    logits = model(inputs, y)
                    loss, stats = structure_loss(logits, targets)

                accelerator.backward(loss)
                if accelerator.sync_gradients and accelerate_cfg.max_grad_norm > 0.:
                    all_norm = accelerator.clip_grad_norm_(
                        trainable_params, accelerate_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = accelerator.gather(loss.detach().repeat(1)).mean()
            avg_loss_scaled = avg_loss / grad_accu_steps
            train_loss = (avg_loss_scaled if train_loss is None
                          else train_loss + avg_loss_scaled)

            if accelerator.sync_gradients:
                if args.use_ema:
                    update_ema(ema_model, model, args.ema_decay)
                progress_bar.update(1)
                global_steps += 1
                if getattr(accelerate_cfg, 'logger', 'wandb') is not None:
                    accelerator.log({"train_loss": train_loss.item()}, step=global_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_steps)
                    accelerator.log(
                        {k: float(v) for k, v in stats.items()}, step=global_steps)
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

                if global_steps in getattr(accelerate_cfg, 'checkpointing_steps_list', []):
                    save_path = os.path.join(ckptdir, f"save-checkpoint-{global_steps}")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()

            if global_steps % accelerate_cfg.logging_steps == 0:
                logs = {"step_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if accelerator.is_main_process:
                    logger.info(
                        f"step={global_steps} / {accelerate_cfg.max_train_steps}, "
                        f"step_loss={logs['step_loss']:.4f}, "
                        f"root_acc={float(stats['root_acc']):.3f}, "
                        f"region_acc={float(stats['region_acc']):.3f}, "
                        f"exact={float(stats['exact']):.3f}, lr={logs['lr']}")

            if global_steps >= accelerate_cfg.max_train_steps:
                stop = True
                break
        epoch += 1

    accelerator.wait_for_everyone()
    accelerator.end_training()


def model_num_classes(model):
    """num_classes of a possibly DDP/compile-wrapped QuadtreePrior."""
    m = model
    while hasattr(m, 'module'):
        m = m.module
    return m.num_classes


if __name__ == '__main__':
    main()
