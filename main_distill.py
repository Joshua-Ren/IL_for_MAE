# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from otherutils import *
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
from engine_distill import train_one_epoch, evaluate, linear_prob_evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('IN1K distill', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--teachmodel', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of teachmodel to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--dis_ratio', type=float, default=1.,
                        help='1 is pure distill, 0 is pure from target')                       

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--teach_ckp', default='',
                        help='teacher from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='can be imagenet, tiny, cifar100')    
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--lp_dataset', default='cifar100', type=str,
                        help='can be imagenet, tiny, cifar100')
                        
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--proj_name',default='Distill-IN1K', type=str)
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # ================= Prepare for distributed training =====
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # ================== Prepare for the distill dataloader ===============
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # ================== Prepare for the dataloader ===============
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # ================== Prepare for the linear prob dataloader ===============
    LP_dataset_train = build_dataset(is_train=True, args=args, force_dataset=args.lp_dataset)
    LP_dataset_val = build_dataset(is_train=False, args=args, force_dataset=args.lp_dataset)

    if True:  # args.distributed:
        LP_sampler_train = torch.utils.data.DistributedSampler(
            LP_dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        LP_sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        LP_sampler_train = torch.utils.data.RandomSampler(LP_dataset_train)
        LP_sampler_val = torch.utils.data.SequentialSampler(LP_dataset_val)

    # ================== Prepare for the dataloader ===============
    LP_data_loader_train = torch.utils.data.DataLoader(
        LP_dataset_train, sampler=LP_sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    LP_data_loader_val = torch.utils.data.DataLoader(
        LP_dataset_val, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # =================== Initialize wandb ========================
    if misc.is_main_process():
        run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
        save_path = base_folder+'results/'+args.proj_name+'/'+args.model+'/'+run_name
        args.output_dir = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # ================== Create the model: ViT ==================
    model = models_vit.__dict__[args.model](num_classes=args.nb_classes,
                                            drop_path_rate=args.drop_path,
                                            global_pool=args.global_pool,
                                            distill=True,)
    teacher = models_vit.__dict__[args.teachmodel](num_classes=args.nb_classes,
                                            drop_path_rate=args.drop_path,
                                            global_pool=args.global_pool,
                                            distill=True,)
    #for param in teacher.parameters():
    #    param.requires_grad = False    
    model.to(device)
    teacher.to(device)
    model_without_ddp = model
            # ----- Load teacher from the checkpoint
    ckp_path = base_folder + 'results/' + args.teach_ckp
    #ckp_path = base_folder+'results/Interact_MAE/mae_vit_base_patch16_smallde/offi_4GPU_smallDE400/checkpoint-300.pth'
    checkpoint = torch.load(ckp_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = teacher.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(teacher, checkpoint_model)

    # load pre-trained model
    msg = teacher.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)
    trunc_normal_(teacher.head.weight, std=2e-5)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    #optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    # ------- Before distill, calculate teacher's results
    t_top1, t_top5 = linear_prob_evaluate(args, teacher, LP_data_loader_train, LP_data_loader_val, device, teach_flag=True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        if misc.is_main_process():   
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, flag_start=True)

        train_one_epoch(model, teacher, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, mixup_fn, args=args)
        s_top1, s_top5=linear_prob_evaluate(args, model_without_ddp, LP_data_loader_train, LP_data_loader_val, device)
        # -------- Linear Prob every epoch --------------
        if misc.is_main_process():
            wandb.log({'epoch':epoch})
            wandb.log({'GAP_TOP1':s_top1-t_top1})
            wandb.log({'GAP_TOP5':s_top5-t_top5})
            if epoch % 25 == 0 or epoch + 1 == args.epochs:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.dataset=='tiny':
        args.nb_classes=200
    elif args.dataset=='cifar100':
        args.nb_class=100
    main(args)