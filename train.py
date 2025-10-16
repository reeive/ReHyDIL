import os
import sys
import time
import math
import torch
import shutil
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm


import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.cph import CPH
from losses import focal_tversky
from utils.metrics import dice as dice_all
from utils.tac_loss import TACLoss
from utils.metrics import batch_dice
from utils.util import set_logging, Logger, read_list, AverageMeter
from utils import tac_queue
from utils.stage_driver import StageConfig, run_stage
from dataloader.dataset import BaseDataSets, PatientBatchSampler


def parse_args():
    p = argparse.ArgumentParser(description="ReHyDIL stage-by-stage trainer")

    # Data and outputs
    p.add_argument("--data_path", type=str, required=True,
                   help="Root directory of your dataset")
    p.add_argument("--out_root", type=str, default="results",
                   help="Root directory for outputs; each stage will create a subdir like res-<mod>/")
    p.add_argument("--stages", type=str, default="t1,t2,flair,t1ce",
                   help="Training order (comma-separated), e.g., t1,t2,flair,t1ce")

    # List files: either explicitly list them, or provide format templates
    p.add_argument("--train_lists", type=str, default="",
                   help="Comma-separated train lists for each stage; if empty, --train_fmt will be used")
    p.add_argument("--val_lists", type=str, default="",
                   help="Comma-separated val lists for each stage; if empty, --val_fmt will be used")
    p.add_argument("--train_fmt", type=str, default="train_{mod}.list",
                   help="Template to locate train lists when --train_lists is empty; {mod} is replaced with modality")
    p.add_argument("--val_fmt", type=str, default="val_{mod}.list",
                   help="Template to locate val lists when --val_lists is empty; {mod} is replaced with modality")

    # Training hyperparameters (shared by all stages)
    p.add_argument("--max_epoch", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--images_rate", type=float, default=1.0)
    p.add_argument("--base_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--optim_name", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr_scheduler", type=str, default="warmupMultistep",
                   choices=["warmupMultistep", "warmupCosine", "autoReduce"])
    p.add_argument("--step_num_lr", type=int, default=4)

    # Loss weights and parameters (aligned with the paper)
    p.add_argument("--tversky_w", type=float, default=7.0)
    p.add_argument("--imb_w", type=float, default=8.0)
    p.add_argument("--nce_weight", type=float, default=3.5)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=1.2)

    # Replay / queue
    p.add_argument("--mem_size", type=int, default=2000,
                   help="Max size for teacher queue / replay samples (stored on CPU in float16)")
    p.add_argument("--p_keep", type=float, default=0.10,
                   help="Proportion of near-median-loss samples kept at the end of each stage")

    # Loader & device
    p.add_argument("--workers_train", type=int, default=8)
    p.add_argument("--workers_val", type=int, default=4)
    p.add_argument("--gpus", type=str, default="0",
                   help="CUDA_VISIBLE_DEVICES value, e.g., 0 or 0,1")
    p.add_argument("--seed", type=int, default=1111)

    return p.parse_args()


def _resolve_lists(stages, train_lists_arg, val_lists_arg, train_fmt, val_fmt):
    """Resolve per-stage train/val list paths."""
    if train_lists_arg.strip():
        train_lists = [s.strip() for s in train_lists_arg.split(",")]
        assert len(train_lists) == len(stages), "--train_lists must match --stages length"
    else:
        train_lists = [train_fmt.format(mod=m) for m in stages]

    if val_lists_arg.strip():
        val_lists = [s.strip() for s in val_lists_arg.split(",")]
        assert len(val_lists) == len(stages), "--val_lists must match --stages length"
    else:
        val_lists = [val_fmt.format(mod=m) for m in stages]

    return train_lists, val_lists


def main():
    args = parse_args()

    # Device setup
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}, GPUs={os.environ.get('CUDA_VISIBLE_DEVICES','-')}")

    # Stage order
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    assert len(stages) >= 1, "Please provide at least one stage (modality)."

    # Resolve list files
    train_lists, val_lists = _resolve_lists(
        stages, args.train_lists, args.val_lists, args.train_fmt, args.val_fmt
    )

    # Prepare output root
    os.makedirs(args.out_root, exist_ok=True)

    # Chain previous stage info
    seen = []             # completed modalities so far (for prev_img_modes)
    prev_base_dir = None  # directory containing previous stage's best model

    for i, mod in enumerate(stages):
        base_dir = os.path.join(args.out_root, f"res-{mod}")
        prev_img_modes = seen.copy() if len(seen) > 0 else None

        cfg = StageConfig(
            base_dir=base_dir,
            data_path=args.data_path,
            train_list=train_lists[i],
            val_list=val_lists[i],
            img_mode=mod,

            prev_img_modes=prev_img_modes,
            prev_base_dir=prev_base_dir,

            mem_size=args.mem_size,
            p_keep=args.p_keep,

            max_epoch=args.max_epoch,
            batch_size=args.batch_size,
            images_rate=args.images_rate,

            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            optim_name=args.optim_name,

            lr_scheduler=args.lr_scheduler,
            step_num_lr=args.step_num_lr,

            tversky_w=args.tversky_w,
            imb_w=args.imb_w,
            nce_weight=args.nce_weight,

            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,

            num_workers_train=args.workers_train,
            num_workers_val=args.workers_val,
            device=device,
            seed=args.seed
        )

        print(f"\n[Stage {i+1}/{len(stages)}] modality={mod}")
        if prev_img_modes:
            print(f"  prev_img_modes: {prev_img_modes}")
        if prev_base_dir:
            print(f"  prev_base_dir: {prev_base_dir}")

        run_stage(cfg)

        # Update chain for the next stage
        seen.append(mod)
        prev_base_dir = base_dir

    print("\n[All Done] All stages finished successfully.")


if __name__ == "__main__":
    main()