#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os
import sys
import torch.backends.cudnn
import torch.utils.data
from torch.cuda import amp

import utils.data_loaders
import utils.data_transforms
import utils.network_utils as net_utils

import models
from models.IGNN import IGNN
from models.VDSR import VDSR

from tensorboardX import SummaryWriter
from core.train import train, train_VDSR
from core.test import test
from core.test_woGT import test_woGT
from losses.losses import *
from datetime import datetime as dt


def bulid_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    if cfg.NETWORK.SRNETARCH == 'IGNN':
        # Set up data augmentation
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE, cfg.CONST.SCALE),
            utils.data_transforms.FlipRotate(),
            # utils.data_transforms.BGR2RGB(),
            # utils.data_transforms.RandomColorChannel(),
            utils.data_transforms.ToTensor()
        ])

        test_transforms = utils.data_transforms.Compose([
            # utils.data_transforms.BGR2RGB(),
            utils.data_transforms.ToTensor()
        ])
       
    elif cfg.NETWORK.SRNETARCH == 'VDSR':
        # Set up data augmentation
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE, cfg.CONST.SCALE),
            utils.data_transforms.Resize(),
            utils.data_transforms.FlipRotate(),
            # utils.data_transforms.BGR2RGB(),
            # utils.data_transforms.RandomColorChannel(),
            utils.data_transforms.ToTensor()
        ])

        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.Resize(),
            # utils.data_transforms.BGR2RGB(),
            utils.data_transforms.ToTensor()
        ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_TRAIN_NAME](utils.data_loaders.DatasetType.TRAIN)
    test_dataset_loader  = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_TEST_NAME](utils.data_loaders.DatasetType.TEST)
    if cfg.NETWORK.PHASE in ['train', 'resume']:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_loader.get_dataset(train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_loader.get_dataset(test_transforms),
            batch_size=cfg.CONST.VAL_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)
    elif cfg.NETWORK.PHASE in ['test']:
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_loader.get_dataset(test_transforms),
            batch_size=cfg.CONST.TEST_BATCH_SIZE,
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    # Set up networks
    net = models.__dict__[cfg.NETWORK.SRNETARCH].__dict__[cfg.NETWORK.SRNETARCH]()
    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.SRNETARCH,
                                                net_utils.count_parameters(net)))

    # Initialize weights of networks
    if cfg.NETWORK.PHASE == 'train' and cfg.NETWORK.SRNETARCH == 'IGNN':
        net_utils.initialize_weights(net, cfg.TRAIN.KAIMING_SCALE)

    # Set up solver
    if cfg.NETWORK.SRNETARCH == 'IGNN':
        solver = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN.LEARNING_RATE,
                                            betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
    elif cfg.NETWORK.SRNETARCH == 'VDSR':
        solver = torch.optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, 
                                momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY, nesterov=False)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, range(cfg.CONST.NUM_GPU)).cuda()

    # Load pretrained model if exists
    Init_Epoch   = 0
    Best_Epoch   = 0
    Best_PSNR    = 0
    if cfg.NETWORK.PHASE in ['test', 'resume']:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)

        net.load_state_dict(checkpoint['net_state_dict'])

        if cfg.NETWORK.PHASE == 'resume': Init_Epoch = checkpoint['epoch_idx']
        Best_PSNR  = checkpoint['best_PSNR']
        Best_Epoch = checkpoint['best_epoch']
        if 'solver_state_dict' in checkpoint:
            solver.load_state_dict(checkpoint['solver_state_dict'])
            
        print('[INFO] {0} Recover complete. Current Epoch #{1}, Best_PSNR = {2} at Epoch #{3}.' \
              .format(dt.now(), Init_Epoch, Best_PSNR, Best_Epoch))

    if cfg.NETWORK.PHASE in ['train', 'resume']:
        # Set up learning rate scheduler to decay learning rates dynamically
        if cfg.NETWORK.SRNETARCH == 'IGNN':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=cfg.TRAIN.LR_MILESTONES,
                                                                            gamma=cfg.TRAIN.LR_DECAY)
        elif cfg.NETWORK.SRNETARCH == 'VDSR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(solver, step_size=cfg.TRAIN.NUM_EPOCHES//4, 
                                                                            gamma=cfg.TRAIN.LR_DECAY)
        
        # Summary writer for TensorBoard 
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'tb_log', dt.now().isoformat()+'_'+cfg.NETWORK.SRNETARCH, '%s')
        log_dir      = output_dir % 'logs'
        ckpt_dir     = output_dir % 'checkpoints'
        train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        val_writer  = SummaryWriter(os.path.join(log_dir, 'val'))

        # train and val
        if cfg.NETWORK.SRNETARCH == 'IGNN':
            train(cfg, Init_Epoch, train_data_loader, val_data_loader, net, solver, lr_scheduler, ckpt_dir, 
                                                            train_writer, val_writer, Best_PSNR, Best_Epoch)
        elif cfg.NETWORK.SRNETARCH == 'VDSR':
            scaler = amp.GradScaler()
            train_VDSR(cfg, Init_Epoch, train_data_loader, val_data_loader, net, solver, lr_scheduler, ckpt_dir, 
                                                            train_writer, val_writer, Best_PSNR, Best_Epoch, scaler)
            
        return
    elif cfg.NETWORK.PHASE in ['test']:
        if cfg.DATASET.DATASET_TEST_NAME == 'Demo':
            test_woGT(cfg, test_data_loader, net)
        else:
            test(cfg, test_data_loader, net, Best_Epoch)
        return
    