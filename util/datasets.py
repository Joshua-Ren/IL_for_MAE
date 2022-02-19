# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets
import torchvision.transforms as T
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .data_loader_lmdb import ImageFolderLMDB


def build_dataset_CIFAR100(is_train,args):
    pass


def build_dataset(is_train, args):
    
    if args.dataset=='imagenet':
        transform = build_transform(is_train, args)
        args.data_path = '/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/datasets/ImageNet/'
        root = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
        dataset = ImageFolderLMDB(root, transform=transform)
    elif args.dataset=='tiny':
        pass
    elif args.dataset=='cifar100':
        train_T=T.Compose([
                        T.RandomResizedCrop(256,scale=(0.05, 1.0)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
        val_T =T.Compose([
                        T.Resize(256),
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])                        
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        T.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(T.CenterCrop(args.input_size))

    t.append(T.ToTensor())
    t.append(T.Normalize(mean, std))
    return T.Compose(t)
