# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:00:37 2021

@author: YIREN
"""
import torch
import torchvision
import wandb
import random
import numpy as np
import os
import torch.distributed as dist
import torchvision.transforms as T

WANDB_track_figs = 6
base_folder = '/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/IL_for_MAE/'

# ------------- Helper function for NIL
'''
# ------ Old version for freeze/defreeze, when MAE model load params from ViT model
def freeze_en_mae(model, msg):
    for name, param in model.named_parameters():
        if name in msg.missing_keys:
            param.requires_grad = True
        else:
            param.requires_grad = False
'''
def freeze_en_mae(model,msg):
    for name, param in model.named_parameters():
        if name not in ['pos_embed', 'decoder_pos_embed']:
            if name.startswith('decode'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False   
        if name in ['mask_token']:
            param.requires_grad = True

def freeze_de_mae(model):
    for name, param in model.named_parameters():
        if name not in ['pos_embed', 'decoder_pos_embed']:
            if name.startswith('decode'):
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = False   
        if name in ['mask_token']:
            param.requires_grad = False

def defreeze_en_mae(model):
    for name, param in model.named_parameters():
        if name not in ['pos_embed', 'decoder_pos_embed']:
            param.requires_grad = True
        else:
            param.requires_grad = False



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def checkpoint_save_interact(mae, g, save_path):
    path = os.path.join(save_path,'checkpoint')
    if not os.path.exists(path):
        os.makedirs(path) 
    file_name = 'encoder_ep'+str(g)+'.pt'
    path = os.path.join(path, file_name)
    torch.save(mae.state_dict(), path)
    return path
    
def checkpoint_save_pretrain(encoder, g, save_path):
    path = os.path.join(save_path,'checkpoint')
    if not os.path.exists(path):
        os.makedirs(path) 
    file_name = 'encoder_ep'+str(g)+'.pt'
    path = os.path.join(path, file_name)
    torch.save(encoder.state_dict(), path)

def ckp_converter(orig_model):
    if list(orig_model.keys())[0][:6]=='module':
        from collections import OrderedDict
        new_model = OrderedDict()
        for k, v in orig_model.items():
            name = k[7:]
            new_model[name]=v
        return new_model
    else:
        return orig_model

# ================== Functions about wandb ===================================
def wandb_gen_track_x(train_loader,):
    #for i,data in enumerate(train_loader):
    #    x = data[0]['data']
    for i, (x, y) in enumerate(train_loader):
        break
    track_tx = x[:WANDB_track_figs*2]
    return track_tx

def wandb_init(proj_name='test', run_name=None, config_args=None):
    wandb.init(
        project=proj_name,
        config={})
    if config_args is not None:
        wandb.config.update(config_args)
    if run_name is not None:
        wandb.run.name=run_name
        return run_name
    else:
        return wandb.run.name

def wandb_record_results(results, epoch):
  for key in results.keys():
    wandb.log({key:results[key][-1]})
  wandb.log({'epoch':epoch})

def wandb_show16imgs(recon_imgs, origi_imgs, table_key='initial', ds_ratio=4):
    '''
        Show (n+n)*2 images in a table with table_key:
        The input should be a tensor of (n+n,3,224,224), we down sample them
            
                From trainset           From valset
        ------------------------------------------------
        Recon      (n imgs)              (n imgs)
        ------------------------------------------------
        Origin     (n imgs)              (n imgs)
        ------------------------------------------------
    '''
    recon_imgs = recon_imgs.clone()
    origi_imgs = origi_imgs.clone()
    assert recon_imgs.shape ==  origi_imgs.shape
    n = int(recon_imgs.shape[0]/2)
    _KEYS = ['train_','valid_']
    #my_table = wandb.Table()
    recon_list, origi_list, colunm_list = [],[],[]
    for i in range(2):
        for j in range(n):
            idx = i*n+j
            recon_list.append(wandb.Image(recon_imgs[idx,:,::ds_ratio,::ds_ratio]))
            origi_list.append(wandb.Image(origi_imgs[idx,:,::ds_ratio,::ds_ratio]))
            colunm_list.append(_KEYS[i]+str(j))
            
    show_data = [recon_list,origi_list]    
    show_table = wandb.Table(data=show_data, columns=colunm_list)
    wandb.log({table_key: show_table})  

# ================= CIFAR_loader for single GPU ============================
def get_cifar_loaders(args):
    train_T=T.Compose([
                        T.Resize(256),
                        #T.RandomCrop(256),
                        #T.RandomResizedCrop(256,scale=(0.05, 1.0)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
    val_T =T.Compose([
                        T.Resize(256),
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
    if args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_T),
                            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=val_T),
                            batch_size=1000, shuffle=False, drop_last=True,pin_memory=True)    
    elif args.dataset.lower()=='cifar100':
        train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=train_T),
                            batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=val_T),
                            batch_size=1000, shuffle=False, drop_last=True,pin_memory=True)
    else:
        print('args.dataset must be cifar10 or cifar100')
    return train_loader, val_loader