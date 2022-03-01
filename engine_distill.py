import math
import sys
from typing import Iterable, Optional
import torch.nn.functional as F
import wandb
import torch
import copy
from timm.data import Mixup
from timm.utils import accuracy
from otherutils import *
import util.misc as misc
import util.lr_sched as lr_sched
from util.lars import LARS
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def distill_cos_loss(teach_words, stud_words, args):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    teach_part = 1-cos(stud_words, teach_words.detach()).mean()
    return teach_part


def train_one_epoch(model: torch.nn.Module, teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    args=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train(True)
    teacher.eval()
    
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            teach_words, mask, ids = teacher.module.forward_encoder(samples, args.mask_ratio)
            mask_ids = (mask, ids)
            stud_words, _, _ = model.module.forward_encoder(samples, args.mask_ratio, mask_ids)
            loss = distill_cos_loss(teach_words, stud_words, args)
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
            if misc.is_main_process():
                wandb.log({'loss':loss.item()}) 
                wandb.log({'learn_rate':lr})                
    if misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
        wandb.log({'epoch':epoch})
