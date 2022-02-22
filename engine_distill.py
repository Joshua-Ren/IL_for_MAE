import math
import sys
from typing import Iterable, Optional
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

def distill_loss(teach_logits, teach_words, logits, words, targets, args):
    # Shape of 
    words_dim = teach_words.shape[-1]
    temper = 1.
    ratio = args.dis_ratio
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    teach_part = 1-cos(words, teach_words).mean()
    label_part = torch.nn.CrossEntropyLoss()(logits, targets)
    return teach_part*ratio + (1-ratio)*label_part


def train_one_epoch(model: torch.nn.Module, teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, args=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train(True)
    teacher.eval()
    
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
            logits, words = model(samples)
            teach_logits, teach_words = teacher(samples)
            loss = distill_loss(teach_logits, teach_words, logits, words, targets, args)
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
            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
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
       
def linear_prob_evaluate(args, model, LP_data_loader_train, LP_data_loader_val,
                         device, epoch=0, teach_flag=False):
    # ------ deep copy the model, linear prob using multi-GPU, 
    # output test-acc, delete the model
    if teach_flag:
        tmp = 'D_'
    else:
        tmp = 'D_'
    v_losses = AverageMeter()
    v_top1 = AverageMeter()
    v_top5 = AverageMeter() 
    accum_iter = args.accum_iter
    # ------- Prepare the model: change head, freeze other layers
    lp_model = copy.deepcopy(model)
    num_features = lp_model.num_features
    if args.lp_dataset=='imagenet':
        num_classes = 1000
    elif args.lp_dataset=='tiny':
        num_classes = 200
    elif args.lp_dataset=='cifar100':
        num_classes = 100
    lp_model.head = torch.nn.Linear(num_features, num_classes)
    lp_model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(lp_model.head.in_features, affine=False, eps=1e-6), lp_model.head)
    #for _, p in lp_model.named_parameters():
    #    p.requires_grad = False
    #for _, p in lp_model.head.named_parameters():
    #    p.requires_grad = True
    lp_model.to(device)
    if args.distributed:
        lp_model = torch.nn.parallel.DistributedDataParallel(lp_model, device_ids=[args.gpu])
        lp_model_without_ddp = lp_model.module
    # --------- Prepare the optimizers
    #optimizer = LARS(lp_model_without_ddp.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(lp_model_without_ddp.parameters(), lr=1e-3)
    loss_scaler = NativeScaler()
    optimizer.zero_grad()
    for k in range(10):
        lp_model.train()
        # --- train linear prob
        for data_iter_step, (samples, targets) in enumerate(LP_data_loader_train):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)            
            with torch.cuda.amp.autocast():
                logits, _ = lp_model(samples)
                loss = torch.nn.CrossEntropyLoss()(logits, targets)
            loss/=accum_iter
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=lp_model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)                
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()   
            torch.cuda.synchronize()
            if data_iter_step%10 == 0 and misc.is_main_process():
                wandb.log({tmp+'T_loss':loss.item()})
        del samples
        # --- Validate linear prob
        lp_model.eval()
        with torch.no_grad():
            for i, (images,target) in enumerate(LP_data_loader_val):
                images = images.to(device, non_blocking=True)
                labels = target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    v_logits, _ = lp_model(images)
                    v_loss = torch.nn.CrossEntropyLoss()(v_logits, labels)
                v_prec1, v_prec5 = accuracy(v_logits, labels, topk=(1, 5))
                v_losses.update(v_loss.data.item(), images.size(0))
                v_top1.update(v_prec1.item(), images.size(0))
                v_top5.update(v_prec5.item(), images.size(0))
            if misc.is_main_process():
                wandb.log({tmp+'V_loss':v_losses.avg})
                wandb.log({tmp+'V_top1':v_top1.avg})
                wandb.log({tmp+'V_top5':v_top5.avg})
        del images
    del lp_model
    return (v_top1.avg, v_top5.avg)
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
            logits, _ = model(images)
            loss = criterion(logits, target)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.data.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        
    if misc.is_main_process():
        wandb.log({'valid_loss':losses.avg})
        wandb.log({'valid_top1':top1.avg})
        wandb.log({'valid_top5':top5.avg})