# utils/stage_driver.py

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import math
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from nets.cph import CPH
from losses import focal_tversky
from dataloader.dataset import BaseDataSets, PatientBatchSampler
from utils.metrics import dice as dice_all
from utils.metrics import batch_dice
from utils.util import AverageMeter

from .tac_queue import TACWithQueues, build_teacher_queue


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@dataclass
class StageConfig:
    base_dir: str
    data_path: str
    train_list: str
    val_list: str
    img_mode: str

    prev_img_modes: Optional[List[str]] = None
    prev_base_dir: Optional[str] = None

    mem_size: int = 320
    p_keep: float = 0.10  # near-median P% kept for replay at stage end

    max_epoch: int = 81
    batch_size: int = 16
    images_rate: float = 1.0

    base_lr: float = 1e-3
    weight_decay: float = 3e-4
    optim_name: str = "adam"

    lr_scheduler: str = "warmupMultistep"  # or warmupCosine, autoReduce
    step_num_lr: int = 4

    tversky_w: float = 7.0
    imb_w: float = 8.0
    nce_weight: float = 3.5

    alpha: float = 0.7  # Tversky alpha
    beta: float = 1.5   # Tversky beta
    gamma: float = 1.2  # Focal gamma

    num_workers_train: int = 8
    num_workers_val: int = 4
    device: str = "cuda"
    seed: int = 1111


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, inputs, targets):
        p = inputs.reshape(-1)
        q = targets.reshape(-1)
        tp = (p * q).sum()
        fp = ((1 - q) * p).sum()
        fn = (q * (1 - p)).sum()
        return 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


def _set_seed(seed: int):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _build_optimizer(net: torch.nn.Module, cfg: StageConfig):
    if cfg.optim_name.lower() == "adam":
        return optim.Adam(net.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    if cfg.optim_name.lower() == "adamw":
        return optim.AdamW(net.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    if cfg.optim_name.lower() == "sgd":
        return optim.SGD(net.parameters(), lr=cfg.base_lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optim_name}")


def _build_scheduler(optimizer, cfg: StageConfig):
    warm_up_epochs = int(cfg.max_epoch * 0.1)
    if cfg.lr_scheduler == 'warmupMultistep':
        if cfg.step_num_lr == 2:
            ms = [int(cfg.max_epoch * 0.3), int(cfg.max_epoch * 0.6)]
        elif cfg.step_num_lr == 3:
            ms = [int(cfg.max_epoch * 0.25), int(cfg.max_epoch * 0.4), int(cfg.max_epoch * 0.6)]
        else:
            ms = [int(cfg.max_epoch * 0.15), int(cfg.max_epoch * 0.35), int(cfg.max_epoch * 0.55), int(cfg.max_epoch * 0.7)]
        def lr_lambda(epoch):
            if epoch < warm_up_epochs:
                return (epoch + 1) / max(1, warm_up_epochs)
            steps = sum(int(m <= epoch) for m in ms)
            return 0.1 ** steps
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif cfg.lr_scheduler == 'warmupCosine':
        def lr_lambda(epoch):
            if epoch < warm_up_epochs:
                return (epoch + 1) / max(1, warm_up_epochs)
            return 0.5 * (math.cos((epoch - warm_up_epochs) / max(1, cfg.max_epoch - warm_up_epochs) * math.pi) + 1)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif cfg.lr_scheduler == 'autoReduce':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True,
                                                    cooldown=2, min_lr=0.0)
    else:
        raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")


def _load_prev_model(prev_base_dir: Optional[str], device: str) -> Optional[torch.nn.Module]:
    if not prev_base_dir:
        return None
    model_path = os.path.join(prev_base_dir, 'model_CPH_best.pth')
    if not os.path.exists(model_path):
        return None
    model = CPH(n_classes=3).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_replay(modality: str, device: str = 'cpu') -> Optional[dict]:
    path = os.path.join('replay_buffer', f'replay_buffer_{modality}.pth')
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    ckpt.setdefault('patients', ["unk"] * ckpt['images'].shape[0])
    ckpt.setdefault('modalities', [modality] * ckpt['images'].shape[0])
    return ckpt


def _save_replay(modality: str,
                 images: torch.Tensor,
                 masks: torch.Tensor,
                 losses: torch.Tensor,
                 patients: List[str],
                 modalities: List[str]):
    os.makedirs('replay_buffer', exist_ok=True)
    path = os.path.join('replay_buffer', f'replay_buffer_{modality}.pth')
    torch.save({
        'images': images.cpu(),
        'masks': masks.cpu(),
        'losses': losses.cpu(),
        'patients': patients,
        'modalities': modalities,
    }, path)
    logging.info(f"[Replay] Saved {images.shape[0]} samples to {path}")


def run_stage(cfg: StageConfig):
    os.makedirs(cfg.base_dir, exist_ok=True)
    _set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # ===== Datasets & Loaders =====
    train_ds = BaseDataSets(cfg.data_path, "train", cfg.img_mode, 'masks_all', cfg.train_list, cfg.images_rate)
    val_ds = BaseDataSets(cfg.data_path, "val",   cfg.img_mode, 'masks_all', cfg.val_list)

    train_sampler = PatientBatchSampler(train_ds.sample_list, cfg.batch_size)

    def _seed_worker(_: int):
        return

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=cfg.num_workers_train, pin_memory=True, persistent_workers=True,
        worker_init_fn=_seed_worker
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers_val, pin_memory=True, persistent_workers=True, drop_last=True,
        worker_init_fn=_seed_worker
    )

    # ===== Net & Optim =====
    net = CPH(n_classes=3).to(device)
    optimizer = _build_optimizer(net, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    # Losses
    tversky = TverskyLoss(alpha=cfg.alpha, beta=cfg.beta)
    tac_loss = TACWithQueues(alpha=cfg.alpha, beta=cfg.beta, tau=1.0)

    # ===== Teacher (prev stage) & Queue =====
    prev_model = _load_prev_model(cfg.prev_base_dir, device)
    teacher_queue = None
    if prev_model is not None and cfg.prev_img_modes:
        imgs_list, pats_list, mods_list = [], [], []
        for m in cfg.prev_img_modes:
            rb = _load_replay(m, device='cpu')
            if rb is None:
                continue
            imgs_list.append(rb['images'].float())
            pats_list.extend(list(rb['patients']))
            mods_list.extend(list(rb['modalities']))
        if imgs_list:
            R_imgs = torch.cat(imgs_list, dim=0)
            teacher_queue = build_teacher_queue(prev_model,
                                                images=R_imgs,
                                                patient_ids=pats_list,
                                                modality_ids=mods_list,
                                                max_size=min(cfg.mem_size, R_imgs.shape[0]),
                                                batch_size=max(16, cfg.batch_size),
                                                device=str(device), out_device='cpu', dtype=torch.float16)
            logging.info(f"[TeacherQueue] Built with {teacher_queue.size} items")

    omega = 0.0 if (prev_model is None) else 1.0

    # ===== Training =====
    best_avg = 0.0

    # Accumulators for stage-end near-median replay selection
    stage_imgs, stage_masks = [], []
    stage_losses, stage_patients, stage_modalities = [], [], []

    for epoch in range(cfg.max_epoch):
        net.train()

        # epoch accumulators (train)
        train_loss_total = 0.0
        train_dice_total = 0.0
        train_batches = 0

        for batch in train_loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            slice_ids = batch['idx']

            if imgs.size(0) < cfg.batch_size:
                continue

            probs = torch.sigmoid(net(imgs))

            # supervised
            loss_tv = tversky(probs, masks)
            loss_ft = focal_tversky(probs, masks, alpha=cfg.alpha, gamma=cfg.gamma, smooth=1.0)

            # TAC
            loss_tac = probs.new_tensor(0.0)
            if omega > 0.0 and teacher_queue is not None and teacher_queue.size > 0:
                patients = [str(s).split('_')[0] for s in slice_ids]
                modalities = [cfg.img_mode] * len(patients)
                loss_tac = tac_loss(probs, patients, modalities, teacher_queue)

            loss = cfg.tversky_w * loss_tv + cfg.imb_w * loss_ft + cfg.nce_weight * omega * loss_tac
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            # train dice
            pred = (probs > 0.5).float()
            dice_sum, num = batch_dice(pred.detach().cpu(), masks.detach().cpu())
            batch_dice_avg = float(dice_sum / max(1, num))

            # accumulate epoch stats
            train_loss_total += float(loss.item())
            train_dice_total += batch_dice_avg
            train_batches += 1

            # accumulate for replay selection
            stage_imgs.append(imgs.detach().cpu().to(torch.float16))
            stage_masks.append(masks.detach().cpu())
            stage_losses.append(loss.detach().cpu())
            stage_patients.extend([str(s).split('_')[0] for s in slice_ids])
            stage_modalities.extend([cfg.img_mode] * imgs.shape[0])

        # ===== Validation =====
        net.eval()
        val_loss_total = 0.0
        val_loss_batches = 0
        val_dice_sum_total = 0.0
        val_dice_count = 0

        WT = AverageMeter(); TC = AverageMeter(); ET = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                masks = batch['mask'].to(device=device, dtype=torch.float32)
                probs = torch.sigmoid(net(imgs))
                pred = (probs > 0.5).float()

                # dice (global)
                dice_sum, nidus_num = batch_dice(pred.detach().cpu(), masks.detach().cpu())
                val_dice_sum_total += float(dice_sum)
                val_dice_count += int(nidus_num)

                # loss (weighted sum consistent with training supervised terms)
                loss_val_tv = tversky(probs, masks)
                loss_val_ft = focal_tversky(probs, masks, alpha=cfg.alpha, gamma=cfg.gamma, smooth=1.0)
                loss_val = cfg.tversky_w * loss_val_tv + cfg.imb_w * loss_val_ft
                val_loss_total += float(loss_val.item())
                val_loss_batches += 1

                # per-class dice
                p = pred.detach().cpu().numpy().astype('uint8')
                t = masks.detach().cpu().numpy().astype('uint8')
                d0,_,_ = dice_all(p[:,0], t[:,0]); WT.update(d0, 1)
                d1,_,_ = dice_all(p[:,1], t[:,1]); TC.update(d1, 1)
                d2,_,_ = dice_all(p[:,2], t[:,2]); ET.update(d2, 1)

        # ===== Averages for this epoch =====
        train_loss_avg = train_loss_total / max(1, train_batches)
        train_dice_avg = train_dice_total / max(1, train_batches)

        val_loss_avg = val_loss_total / max(1, val_loss_batches)
        val_dice_avg = val_dice_sum_total / max(1, val_dice_count)

        avg3 = (WT.avg + TC.avg + ET.avg) / 3.0
        lr_now = optimizer.param_groups[0]['lr']

        # ===== Scheduler step =====
        if cfg.lr_scheduler == 'autoReduce':
            # use val loss average as the plateau metric
            scheduler.step(val_loss_avg)
        else:
            scheduler.step()

        # ===== Save best & log =====
        if avg3 > best_avg:
            best_avg = avg3
            torch.save(net.state_dict(), os.path.join(cfg.base_dir, 'model_CPH_best.pth'))
            logging.info(f"[Best] epoch={epoch} Avg3={best_avg:.4f}")

        logging.info(
            f"[{cfg.img_mode}] "
            f"Epoch {epoch+1:03d}/{cfg.max_epoch:03d} | "
            f"train_loss={train_loss_avg:.4f} train_dice={train_dice_avg:.4f} | "
            f"val_loss={val_loss_avg:.4f} val_dice={val_dice_avg:.4f} | "
            f"WT={WT.avg:.4f} TC={TC.avg:.4f} ET={ET.avg:.4f} | "
            f"lr={lr_now:.6g}"
        )

    # ===== Stage end: near-median P% replay selection =====
    if stage_losses:
        X = torch.cat(stage_imgs, dim=0).to(torch.float16)
        Y = torch.cat(stage_masks, dim=0)
        L = torch.stack(stage_losses).float().view(-1)
        mu = torch.median(L)
        dev = torch.abs(L - mu)
        k = max(1, int(cfg.p_keep * len(L)))
        sel = torch.argsort(dev)[:k]

        Xs = X[sel].contiguous()
        Ys = Y[sel].contiguous()
        Ls = L[sel].contiguous()
        pats = [stage_patients[i] for i in sel.tolist()]
        mods = [stage_modalities[i] for i in sel.tolist()]
        _save_replay(cfg.img_mode, Xs, Ys, Ls, pats, mods)

    torch.save(net.state_dict(), os.path.join(cfg.base_dir, 'model_CPH_last.pth'))
    logging.info("[Stage] Done.")
