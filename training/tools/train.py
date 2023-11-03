# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit
import cv2

import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import _init_paths
import models
from configs import config
from configs import update_config
from utils.function import train, validate
from utils.utils import create_logger
from utils.train_init import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)  
    parser.add_argument('--savedir',
                        help='Output dir name for logs and checkpoints',
                        default='output/lpcv/',
                        type=str,
                        )
    parser.add_argument('--seed', type=int, default=100)  
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)


    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()       

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.savedir, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer = SummaryWriter(tb_log_dir)

    set_enviroment(config, seed=args.seed)
    
    # Data
    train_dataset, test_dataset = get_data(config)
    trainloader, testloader = get_loader(config, train_dataset, test_dataset)

    # Criterion
    sem_criterion = get_sem_criterion(config, train_dataset.class_weights)
    bd_criterion = BondaryLoss()
    
    # Model
    gpus = list(config.GPUS)
    print("Load Model")
    model = models.pidnet.get_seg_model(
        model_name=config.MODEL.NAME,
        num_classes=config.DATASET.NUM_CLASSES,
        aux_heads=config.TRAIN.USE_AUX_HEADS,
        checkpoint_path=config.MODEL.PRETRAINED,
        strict_state_dict=True
    )
    print("Model Loaded")
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Optimizer
    optimizer = get_optimizer(config, model)

    flag_rm = config.TRAIN.RESUME
    if flag_rm:
        resume_training(model, final_output_dir, optimizer, logger=logger)

    start = timeit.default_timer()

    epoch_iters = int(len(train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = end_epoch * epoch_iters

    best_mIoU = 0
    best_mF1 = 0
    best_org_mF1 = 0
    last_epoch = 0

    pbar = tqdm(range(last_epoch, end_epoch))
    for epoch in pbar:
        if trainloader.sampler is not None and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        train(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            sem_criterion=sem_criterion,
            bd_criterion=bd_criterion,
            epoch=epoch,
            epoch_iters=epoch_iters,
            num_iters=num_iters,
            writer=writer,
            config=config
        )

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < end_epoch - 100) or (epoch >= end_epoch - 100):
            validate(
                model=model,
                testloader=trainloader,
                sem_criterion=sem_criterion,
                bd_criterion=bd_criterion,
                epoch=epoch,
                writer=writer,
                config=config,
                phase="train"
            )
            mean_IoU, mean_F1, org_mean_F1 = validate(
                model=model,
                testloader=testloader,
                sem_criterion=sem_criterion,
                bd_criterion=bd_criterion,
                epoch=epoch,
                writer=writer,
                config=config,
                phase="valid"
            )

        if flag_rm == 1:
            flag_rm = 0

        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth'))

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best_IoU.pt'))

        if mean_F1 > best_mF1:
            best_mF1 = mean_F1
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best_F1.pt'))

        if org_mean_F1 > best_org_mF1:
            best_org_mF1 = org_mean_F1
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best_org_F1.pt'))

        pbar.set_description(
            f"mIoU: {best_mIoU:.3f} - last: {mean_IoU:.3f}; mF1: {best_mF1:.3f} - last: {mean_F1:.3f}; org_mF1: {best_org_mF1:.3f} - last: {org_mean_F1:.3f}"
        )
            
    torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pt'))

    writer.close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
