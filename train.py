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

from dataloader.dataset import BaseDataSets, PatientBatchSampler


class ReplayBuffer:
    def __init__(self, mem_size, image_shape, mask_shape, device='cuda'):

        self.mem_size = mem_size
        self.device = device
        self.images = torch.zeros((mem_size, *image_shape), dtype=torch.float32).to(device)
        self.masks = torch.zeros((mem_size, *mask_shape), dtype=torch.float32).to(device)
        self.losses = torch.zeros((mem_size,), dtype=torch.float32).to(device)
        self.ptr = 0
        self.size = 0

    def add_samples(self, images, masks, losses):

        n = images.size(0)
        if n > self.mem_size:
            images = images[-self.mem_size:]
            masks = masks[-self.mem_size:]
            losses = losses[-self.mem_size:]
            n = self.mem_size

        end_ptr = self.ptr + n
        if end_ptr <= self.mem_size:
            self.images[self.ptr:end_ptr].copy_(images)
            self.masks[self.ptr:end_ptr].copy_(masks)
            self.losses[self.ptr:end_ptr].copy_(losses)
        else:
            first_part = self.mem_size - self.ptr
            self.images[self.ptr:].copy_(images[:first_part])
            self.images[:end_ptr % self.mem_size].copy_(images[first_part:])
            self.masks[self.ptr:].copy_(masks[:first_part])
            self.masks[:end_ptr % self.mem_size].copy_(masks[first_part:])
            self.losses[self.ptr:].copy_(losses[:first_part])
            self.losses[:end_ptr % self.mem_size].copy_(losses[first_part:])
        self.ptr = end_ptr % self.mem_size
        self.size = min(self.size + n, self.mem_size)

    def sample(self, batch_size):

        if self.size == 0:
            return None, None
        current_losses = self.losses[:self.size]
        mean_loss = current_losses.mean()
        deviations = torch.abs(current_losses - mean_loss)
        sorted_indices = torch.argsort(deviations)
        top_count = max(1, int(0.1 * self.size))
        top_indices = sorted_indices[:top_count]
        selected = top_indices.tolist()
        return self.images[selected], self.masks[selected]

    def save(self, save_dir, filename):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filename)
        torch.save({
            'images': self.images[:self.size].cpu(),
            'masks': self.masks[:self.size].cpu(),
            'losses': self.losses[:self.size].cpu(),
            'ptr': self.ptr,
            'size': self.size
        }, save_path)
        print(f"Replay buffer saved to {save_path}")

    def load(self, load_dir, filename):

        load_path = os.path.join(load_dir, filename)
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            loaded_size = checkpoint['size']
            self.images[:loaded_size].copy_(checkpoint['images'].to(self.device))
            self.masks[:loaded_size].copy_(checkpoint['masks'].to(self.device))
            self.losses[:loaded_size].copy_(checkpoint['losses'].to(self.device))
            self.ptr = checkpoint['ptr']
            self.size = loaded_size
            print(f"Replay buffer loaded from {load_path} with {loaded_size} samples.")
        else:
            print(f"No replay buffer found at {load_path}. Starting with an empty buffer.")

    def __len__(self):
        return self.size


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=1.5, smooth=1e-6):

        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky_index



def load_multi_modal_replay_buffer(prev_img_mode, mem_size, image_channels, img_h, img_w, class_num, device):

    modes = prev_img_mode.split('-')
    multi_modal_buffer = ReplayBuffer(mem_size, (image_channels, img_h, img_w), (class_num, img_h, img_w), device)
    for mode in modes:
        mode_dir = os.path.join("replay_buffer")
        filename = f"replay_buffer_{mode}.pth"
        single_buffer = ReplayBuffer(mem_size, (image_channels, img_h, img_w), (class_num, img_h, img_w), device)
        single_buffer.load(mode_dir, filename)
        multi_modal_buffer.add_samples(
            single_buffer.images[:single_buffer.size],
            single_buffer.masks[:single_buffer.size],
            single_buffer.losses[:single_buffer.size]
        )
        print(f"Loaded replay buffer for modality {mode} from {os.path.join(mode_dir, filename)}")
    return multi_modal_buffer


def train_net(start_time, base_dir, prev_base_dir, data_path, prev_data_path,
              train_list, prev_list, val_list, device, img_mode='', prev_img_mode='',
              lr_scheduler='warmupMultistep',
              max_epoch=81,
              batch_size=80,
              images_rate=0.1,
              base_lr=0.001,
              weight_decay=0.0003,
              optim_name='adam',
              loss_name='TAC',
              tversky_w=7.0,
              imb_w=8.0,
              nce_weight=3.5,
              sur_siml='tversky',
              pHead_sur='set_false',
              mem_size=320, ):
    step_num_lr = 4
    local_vars_dict = {}
    for var in train_net.__code__.co_varnames:
        if var == 'local_vars_dict':
            break
        local_vars_dict[var] = locals()[var]

    warm_up_epochs = int(max_epoch * 0.1)

    image_channels = 1
    class_num = 3
    if class_num == 1:
        mask_name = 'masks'
    elif class_num == 3:
        mask_name = 'masks_all'

    net = CPH(n_classes=class_num)
    net.to(device=device)
    net_name = str(net)[0:str(net).find('(')]


    if '-' in prev_img_mode:
        contrast_replay_buffer = load_multi_modal_replay_buffer(prev_img_mode, mem_size, image_channels, img_h, img_w, class_num, device)
    else:
        contrast_replay_buffer = ReplayBuffer(mem_size, (image_channels, img_h, img_w), (class_num, img_h, img_w), device)
        replay_buffer_load_dir = os.path.join("replay_buffer")
        replay_buffer_filename = f"replay_buffer_{prev_img_mode}.pth"
        contrast_replay_buffer.load(replay_buffer_load_dir, replay_buffer_filename)


    current_replay_buffer = ReplayBuffer(mem_size, (image_channels, img_h, img_w), (class_num, img_h, img_w), device)
    current_replay_buffer_save_dir = os.path.join("replay_buffer")
    current_replay_buffer_filename = f"replay_buffer_{img_mode}.pth"
    if os.path.exists(os.path.join(current_replay_buffer_save_dir, current_replay_buffer_filename)):
        current_replay_buffer.load(current_replay_buffer_save_dir, current_replay_buffer_filename)
    print("Replay buffer for contrast learning and current modality initialized.")

    prev_model_loaded = False
    prev_model = CPH(n_classes=class_num)
    prev_model_path = os.path.join(prev_base_dir, 'model_CPH_best.pth')
    if os.path.exists(prev_model_path):
        try:
            state_dict_prev = torch.load(prev_model_path, map_location=device)
            prev_model.load_state_dict(state_dict_prev)
            print(f"Successfully loaded previous model: {prev_model_path}")
            prev_model_loaded = True
        except Exception as e:
            print(f"Error loading previous model: {e}")
            print("Contrast learning will not be used.")
    else:
        print(f"Previous model file does not exist: {prev_model_path}. Contrast learning will not be used.")

    if prev_model_loaded:
        prev_model.to(device=device)
        prev_model.eval()
        for param in prev_model.parameters():
            param.requires_grad = False

    train_dataset = BaseDataSets(data_path, "train", img_mode, mask_name, train_list, images_rate)
    val_dataset = BaseDataSets(data_path, "val", img_mode, mask_name, val_list)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    def worker_init_fn(worker_id):
        random.seed(1111 + worker_id)


    slices_list = train_dataset.sample_list

    patientID_list = list({s.split('_')[0] for s in slices_list})
    train_batch_sampler = PatientBatchSampler(slices_list, batch_size)
    
    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              num_workers=64,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)

    val_loader_2d = DataLoader(val_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=64,
                               pin_memory=True,
                               drop_last=True,
                               worker_init_fn=worker_init_fn)

    day_time = start_time.split(' ')
    time_str = str(day_time[0].split('-')[1] + day_time[0].split('-')[2] +
                   day_time[1].split(':')[0] + day_time[1].split(':')[1])

    if optim_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_name == 'adamW':
        optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=weight_decay)

    if lr_scheduler == 'warmupMultistep':
        if step_num_lr == 2:
            lr1, lr2 = int(max_epoch * 0.3), int(max_epoch * 0.6)
            lr_milestones = [lr1, lr2]
        elif step_num_lr == 3:
            lr1, lr2, lr3 = int(max_epoch * 0.25), int(max_epoch * 0.4), int(max_epoch * 0.6)
            lr_milestones = [lr1, lr2, lr3]
        elif step_num_lr == 4:
            lr1, lr2, lr3, lr4 = int(max_epoch * 0.15), int(max_epoch * 0.35), int(max_epoch * 0.55), int(max_epoch * 0.7)
            lr_milestones = [lr1, lr2, lr3, lr4]
        warm_up_with_multistep_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.1 ** len([m for m in lr_milestones if m <= epoch])
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    elif lr_scheduler == 'warmupCosine':
        warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.5 * (math.cos((epoch - warm_up_epochs) / (max_epoch - warm_up_epochs) * math.pi) + 1)
        scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif lr_scheduler == 'autoReduce':
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True,
                                                            cooldown=2, min_lr=0)



    tversky_loss_fn = TverskyLoss(alpha=0.7, beta=1.5, smooth=1e-6)
    tac_criterion = TACLoss(temperature=0.07, ignore_label=-1, alpha=0.7, smooth=1.0)

    optimizer_name = str(optimizer)[0:str(optimizer).find('(')]
    param_str = "Starting training:"
    for var in list(local_vars_dict.keys()):
        if var != 'device':
            var_value = local_vars_dict[var]
            param_str += "\n\t" + var + ":" + " " * (15 - len(var)) + str(var_value)

    logging.info(param_str + f'''\n\tNet Name:\t\t{net_name}\n\tInput Channel:\t{image_channels}\n\tClasses Num:\t{class_num}\n\tImages Shape:\t{img_h}*{img_w}''')

    train_log = AverageMeter()
    val_log = AverageMeter()
    lr_curve = list()
    best_dice = 0.0

    val_dice_WT = AverageMeter()
    val_dice_TC = AverageMeter()
    val_dice_ET = AverageMeter()

    for epoch in range(max_epoch):
        net.train()


        train_epoch_imgs_list = []
        train_epoch_masks_list = []
        train_epoch_loss_list = []

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{max_epoch}', unit='img', leave=True) as pbar:
            for i, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = batch['mask']
                slice_name = batch['idx']

                current_batch_size = imgs.size(0)
                if current_batch_size < batch_size:
                    break

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)


                masks_pred = net(imgs)
                masks_pred = torch.sigmoid(masks_pred)

                loss_tversky = tversky_loss_fn(masks_pred, true_masks)
                loss_imb = focal_tversky(masks_pred, true_masks, alpha=0.7, gamma=1.2, smooth=1.0)


                loss_tac = 0.0
                if prev_model_loaded:
                    if len(contrast_replay_buffer) >= batch_size:
                        buffer_imgs, buffer_masks = contrast_replay_buffer.sample(batch_size)
                        buffer_imgs = buffer_imgs.to(device=device, dtype=torch.float32)
                        buffer_masks = buffer_masks.to(device=device, dtype=torch.float32)
                        with torch.no_grad():
                            prev_outputs = prev_model(buffer_imgs)
                            prev_outputs = torch.sigmoid(prev_outputs)
                        n_prev = prev_outputs.size(0)
                        loss_tac = tac_criterion(masks_pred[:n_prev], prev_outputs, true_masks[:n_prev])
                    else:
                        loss_tac = 0.0


                loss = loss_tversky * tversky_w + loss_imb * imb_w + loss_tac * nce_weight
                train_log.add_value({"loss": loss.item()}, n=1)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                pred = (masks_pred > 0.5).float()
                dice_sum, num = batch_dice(pred.cpu(), true_masks.cpu())
                dice = dice_sum / num
                train_log.add_value({"dice": dice}, n=1)


                for idx in range(current_batch_size):
                    l_t = tversky_loss_fn(masks_pred[idx:idx + 1], true_masks[idx:idx + 1])
                    l_i = focal_tversky(masks_pred[idx:idx + 1], true_masks[idx:idx + 1],
                                        alpha=0.7, gamma=1.5, smooth=1.0)
                    if len(contrast_replay_buffer) > 0:
                        replay_img, _ = contrast_replay_buffer.sample(1)
                        replay_img = replay_img.to(device)
                        with torch.no_grad():
                            replay_output = net(replay_img)
                            replay_output = torch.sigmoid(replay_output)
                        l_tac = tac_criterion(masks_pred[idx:idx + 1], replay_output, true_masks[idx:idx + 1])
                    else:
                        l_tac = torch.tensor(0.0, device=device)
                    sample_loss = l_t * tversky_w + l_i * imb_w + l_tac * nce_weight

                    train_epoch_imgs_list.append(imgs[idx:idx + 1].detach().cpu())
                    train_epoch_masks_list.append(true_masks[idx:idx + 1].detach().cpu())
                    train_epoch_loss_list.append(sample_loss.detach().cpu())

                pbar.update(current_batch_size)

            train_log.updata_avg()
            mean_loss = train_log.res_dict["loss"][epoch]
            mean_dice = train_log.res_dict["dice"][epoch]


        net.eval()
        val_log.reset()
        val_dice_WT.reset()
        val_dice_TC.reset()
        val_dice_ET.reset()

        with tqdm(total=len(val_loader_2d), desc='Validation round', unit='batch', leave=False) as pbar:
            for j, batch_val in enumerate(val_loader_2d):
                imgs = batch_val['image']
                true_masks = batch_val['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    mask_pred = net(imgs)
                    mask_pred = torch.sigmoid(mask_pred)

                pred = (mask_pred > 0.5).float()
                dice_val_sum, nidus_num = batch_dice(pred.cpu(), true_masks.cpu())
                val_log.add_value({"dice": dice_val_sum}, n=nidus_num)

                loss_val_tversky = tversky_loss_fn(mask_pred, true_masks)
                loss_val_imb = focal_tversky(mask_pred, true_masks, alpha=0.7, gamma=1.2, smooth=1.0)
                loss_val = loss_val_tversky * tversky_w + loss_val_imb * imb_w
                val_log.add_value({"loss": loss_val.item()}, n=1)

                pred_np = pred.cpu().numpy().astype("uint8")
                true_np = true_masks.cpu().numpy().astype("uint8")
                dice_val_0, _, _ = dice_all(pred_np[:, 0, :, :], true_np[:, 0, :, :])  # WT
                val_dice_WT.update(dice_val_0, 1)
                dice_val_1, _, _ = dice_all(pred_np[:, 1, :, :], true_np[:, 1, :, :])  # TC
                val_dice_TC.update(dice_val_1, 1)
                dice_val_2, _, _ = dice_all(pred_np[:, 2, :, :], true_np[:, 2, :, :])  # ET
                val_dice_ET.update(dice_val_2, 1)
                pbar.update()

            val_log.updata_avg()
            valid_loss_mean = val_log.res_dict["loss"][epoch]
            valid_dice_mean = val_log.res_dict["dice"][epoch]
            valid_dice_WT = val_dice_WT.avg
            valid_dice_TC = val_dice_TC.avg
            valid_dice_ET = val_dice_ET.avg
            mean_Avg = (valid_dice_WT + valid_dice_TC + valid_dice_ET) / 3.0

        if lr_scheduler == 'autoReduce':
            scheduler_lr.step(valid_loss_mean)
        else:
            scheduler_lr.step()
        lr_epoch = optimizer.param_groups[0]['lr']
        lr_curve.append(lr_epoch)

        if mean_Avg > best_dice:
            best_dice = mean_Avg
            model_path = os.path.join(base_dir, f'model_{net_name}_best.pth')
            torch.save(net.state_dict(), model_path)
            logging.info(f'Best model saved with Avg: {best_dice:.4f}')


            if len(train_epoch_imgs_list) > 0:
                epoch_imgs_tensor = torch.cat(train_epoch_imgs_list, dim=0)
                epoch_masks_tensor = torch.cat(train_epoch_masks_list, dim=0)
                epoch_loss_tensor = torch.stack(train_epoch_loss_list, dim=0)
                current_replay_buffer.add_samples(
                    epoch_imgs_tensor.to(device),
                    epoch_masks_tensor.to(device),
                    epoch_loss_tensor.to(device)
                )
                if not os.path.exists(current_replay_buffer_save_dir):
                    os.makedirs(current_replay_buffer_save_dir)
                current_replay_buffer.save(current_replay_buffer_save_dir, current_replay_buffer_filename)

                del train_epoch_imgs_list[:]
                del train_epoch_masks_list[:]
                del train_epoch_loss_list[:]

    return train_log.res_dict, val_log.res_dict, lr_curve, net


def main():
    data_path = ' '
    base_dir = ' '
    img_mode = ' '  # e.g. flair, t1, t2, t1ce

    prev_img_mode = ' '
    prev_base_dir = ' '

    if set_args == True:
        args = set_argparse()
        print('WARNING!!! Using argparse for parameters')
        base_dir = args.base_dir
        img_mode = args.img_mode
        prev_base_dir = args.prev_base_dir
        prev_img_mode = args.prev_img_mode
    assert 'res-' in base_dir, f"base_dir should include string:'res-', but base_dir is '{base_dir}'."
    if img_mode == prev_img_mode:
        base_dir = base_dir.replace('res-', f'res-{img_mode}-', 1)
    else:
        base_dir = base_dir.replace('res-', f'res-{img_mode}-{prev_img_mode}-', 1)
    assert 'res-' in prev_base_dir, f"prev_base_dir should include string:'res-', but prev_base_dir is '{prev_base_dir}'."
    prev_base_dir = prev_base_dir.replace('res-', f'res-{prev_img_mode}-', 1)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    backup_code(base_dir)
    log_path = os.path.join(base_dir, 'training.log')
    sys.stdout = Logger(log_path=log_path)
    set_logging(log_path=log_path)
    set_random_seed(seed_num=1111)

    gpu_list = [0]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(
        f'Using device : {device}\n\tGPU ID is [{os.environ["CUDA_VISIBLE_DEVICES"]}], using {torch.cuda.device_count()} device\n\tdevice name:{torch.cuda.get_device_name(0)}')

    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    global img_h, img_w
    data_path = " "
    prev_data_path = " "
    img_h, img_w = 224, 224
    train_list = ' '
    val_list = ' '
    global patientID_list
    patientID_list = read_list('patient_demo.list')
    time_tic = time.time()
    global is_leave
    is_leave = True
    logging.info('============ Start train ==============')
    if set_args == True:
        is_leave = False
        train_log, val_log, lr_curve, net = train_net(
            start_time, base_dir, prev_base_dir, data_path, prev_data_path,
            args.train_list, args.prev_list, args.val_list, device,
            args.img_mode, args.prev_img_mode, args.lr_scheduler, args.max_epoch,
            args.batch_size, args.images_rate, args.base_lr, args.weight_decay,
            args.optim_name, args.loss_name, args.tversky_w, args.imb_w,
            args.nce_weight, args.sur_siml, args.pHead_sur
        )
    else:
        train_log, val_log, lr_curve, net = train_net(start_time, base_dir, data_path, train_list, val_list, device,
                                                      img_mode)

    net_name = str(net)[0:str(net).find('(')]
    mode_path_name = os.path.join(base_dir, f'model_{net_name}_last.pth')
    torch.save(net, mode_path_name)
    logging.info('Model saved!')

    time_toc = time.time()
    time_s = time_toc - time_tic
    time_end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_toc))
    logging.info("Train finished time: {}".format(time_end))
    logging.info("Time consuming: {:.2f} min in train and test".format(time_s / 60))


def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='res-demo1', help='base dir name')
    parser.add_argument('--prev_base_dir', type=str, default='res-demo1', help='prev base dir name')
    parser.add_argument('--train_list', type=str, default='patient_demo.list', help='demo list')
    parser.add_argument('--prev_list', type=str, default='randP1_slice_nidus_train.list', help='a list of prev data')
    parser.add_argument('--val_list', type=str, default='patient_demo.list', help='demo list')
    parser.add_argument('--img_mode', type=str, default='t1', help='medical images mode')
    parser.add_argument('--prev_img_mode', type=str, default='t1', help='previous medical images mode')
    parser.add_argument('--max_epoch', type=int, default=200, help='maximum epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size per gpu')
    parser.add_argument('--images_rate', type=float, default=1, help='images rate')
    parser.add_argument('--base_lr', type=float, default=0.006, help='segmentation network learning rate')
    parser.add_argument('--step_num_lr', type=int, default=4, help='step_num for warmupMultistep lr')
    parser.add_argument('--weight_decay', type=float, default=0.0004, help='weight decay(L2 Regularization)')
    parser.add_argument('--optim_name', type=str, default='adam', help='optimizer name')
    parser.add_argument('--loss_name', type=str, default='bce', help='loss name')
    parser.add_argument('--tversky_w', type=float, default=0.3, help='dice sup Weight')
    parser.add_argument('--imb_w', type=float, default=0.4, help='imb sup Weight')
    parser.add_argument('--lr_scheduler', type=str, default='warmupMultistep', help='lr scheduler')
    parser.add_argument('--nce_weight', type=float, default=1, help='contrast loss weight')
    parser.add_argument('--sur_siml', type=str, default='tversky', help='sur_siml, tversky, dice, cos')
    parser.add_argument('--pHead_sur', type=str, default='set_false', help='pHead_sur')
    parser.add_argument('--mem_size', type=str, default=4154, help='reply memory')
    args = parser.parse_args()
    return args


def set_random_seed(seed_num):
    if seed_num != '':
        logging.info(f'set random seed: {seed_num}')
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)


def backup_code(base_dir):
    code_path = os.path.join(base_dir, 'code')
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    train_name = os.path.basename(__file__)
    dataset_name = 'dataset.py'
    shutil.copy('dataloader/' + dataset_name, os.path.join(code_path, dataset_name))
    shutil.copy(train_name, os.path.join(code_path, train_name))


if __name__ == '__main__':
    set_args = True
    main()
