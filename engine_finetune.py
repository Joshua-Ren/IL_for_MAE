# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import wandb
import torch

from timm.data import Mixup
from timm.utils import accuracy
from otherutils import *
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, args=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train(True)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()   
        torch.cuda.synchronize()
        if data_iter_step%10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.data.item(), samples.size(0))
            top1.update(prec1.item(), samples.size(0))
            top5.update(prec5.item(), samples.size(0))   
            if misc.is_main_process():
                wandb.log({'loss':loss.item()})            
                wandb.log({'learn_rate':lr})

    if misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
        wandb.log({'epoch':epoch})
        wandb.log({'train_loss':losses.avg})
        wandb.log({'train_top1':top1.avg})
        wandb.log({'train_top5':top5.avg})
            
@torch.no_grad()
def evaluate(data_loader, model, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    for i, (images,target) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.data.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        
    if misc.is_main_process():
        wandb.log({'valid_loss':losses.avg})
        wandb.log({'valid_top1':top1.avg})
        wandb.log({'valid_top5':top5.avg})
    return top1.avg, top5.avg