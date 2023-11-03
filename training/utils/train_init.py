import os

import cv2

import random
import numpy as np

import torch
import torch.optim
from torch.backends import cudnn
import albumentations as A

from sam.sam import SAM
import datasets
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss


def set_enviroment(config, seed=100):
    prepare_seed(seed)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0

    torch.set_num_threads(config.TORCH_THREADS)
    cv2.setNumThreads(config.CV2_THREADS)


def resume_training(model, final_output_dir, optimizer, logger):
    model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
    if os.path.isfile(model_state_file):
        checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
        best_mIoU = checkpoint['best_mIoU']
        last_epoch = checkpoint['epoch']
        dct = checkpoint['state_dict']
        
        model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))


def prepare_seed(seed=None):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sem_criterion(config, class_weights):
    print("sem_criterion class_weights:")
    print(class_weights)

    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=class_weights)
    
    return sem_criterion


def get_data(config):
    transform = A.Compose([
            A.ColorJitter(p=0.2),
            A.SafeRotate(limit=20, border_mode=4, p=0.2, interpolation=4),
            A.HorizontalFlip(p=0.2),

            A.GridDistortion(p=0.2),    
            A.OpticalDistortion(p=0.2),        
            A.CLAHE(p=0.2),
            A.RandomBrightnessContrast(p=0.2),    
            A.RandomGamma(p=0.2)
        ])

    train_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=train_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR,
                        transform=transform
                        )


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)



    return train_dataset, test_dataset


def get_loader(config, train_dataset, test_dataset):
    gpus = list(config.GPUS)    

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=False
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False
    )

    return trainloader, testloader



def get_optimizer(config, model):
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params,
                                lr=config.TRAIN.LR,
                                weight_decay=config.TRAIN.WD,
                                )
    elif config.TRAIN.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                lr=config.TRAIN.LR,
                                weight_decay=config.TRAIN.WD,
                                )
    elif config.TRAIN.OPTIMIZER == 'radam':
        optimizer = torch.optim.RAdam(params,
                                lr=config.TRAIN.LR,
                                weight_decay=config.TRAIN.WD,
                                )
    else:
        raise ValueError('Only Support SGD, Adam, AdamW, RAdam optimizers')

    if config.TRAIN.SAM:
        print(f"SAM activated! Rho {config.TRAIN.SAM_RHO}")
        optimizer = SAM(params, optimizer, rho=config.TRAIN.SAM_RHO)

    return optimizer
