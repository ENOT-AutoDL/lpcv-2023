import logging
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F

from utils.utils import AverageMeter, Metrics, AccuracyTracker
from utils.utils import adjust_learning_rate


def train(
    model,
    trainloader,
    optimizer,
    sem_criterion,
    bd_criterion,
    epoch,
    epoch_iters,
    num_iters,
    writer,
    config
):
    align_corners = config.MODEL.ALIGN_CORNERS
    ignore_label = config.TRAIN.IGNORE_LABEL
    base_lr = config.TRAIN.LR
    sem_criterion = sem_criterion.cuda()

    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        size = labels.size()
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        def closure():
            optimizer.zero_grad()

            pred = model(images)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]

            for i in range(len(pred)):
                pred[i] = F.interpolate(pred[i], size=size[-2:], mode='bilinear', align_corners=align_corners)

            aux_sum_loss = None
            if len(pred) != 1: # if use aux heads
                aux_pred, pred, aux_bd_pred = pred
                pred = [pred]

                aux_bd_loss = bd_criterion(aux_bd_pred, bd_gts).mean()
                aux_loss = sem_criterion(aux_pred, labels).mean()

                filler = torch.ones_like(labels) * ignore_label
                bd_label = torch.where(torch.sigmoid(aux_bd_pred[:, 0, :, :]) > 0.8, labels, filler)
                aux_sem_bd_loss = sem_criterion(pred[0], bd_label).mean()

                aux_sum_loss = aux_loss + aux_bd_loss + aux_sem_bd_loss

                avg_bce_loss.update(aux_bd_loss.item())

            task_loss = sem_criterion(pred[0], labels).mean()

            loss = task_loss
            if aux_sum_loss:
                loss = task_loss + aux_sum_loss

            model.zero_grad()
            loss.backward()

            # update average loss
            ave_loss.update(loss.item())
            avg_sem_loss.update(task_loss.item())

            return loss

        optimizer.step(closure)

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        writer.add_scalar('lr/lr', lr, i_iter+cur_iters)

    writer.add_scalar('Loss/train_loss', ave_loss.average(), epoch)


def postproc(mask):
    classes = np.unique(mask)
    if 4 in classes:
        mask = np.where(mask == 9, 0, mask)

    if 7 in classes and (mask == 7).sum() < 100:
        mask = np.where(mask == 7, 0, mask)

    for cl in classes:
        if (mask == cl).sum() < 500:
            mask = np.where(mask == cl, 0, mask)
    return mask


def validate(
    model,
    testloader,
    sem_criterion,
    bd_criterion,
    epoch,
    writer,
    config,
    phase
):
    num_classes = config.DATASET.NUM_CLASSES
    align_corners = config.MODEL.ALIGN_CORNERS
    ignore_label = config.TRAIN.IGNORE_LABEL

    sem_criterion = sem_criterion.cuda()

    model.eval()
    ave_loss = AverageMeter()
    our_metrics = Metrics(num_classes, ignore_label, "cuda")
    org_metrics = AccuracyTracker(num_classes)
    org_mF1 = 0

    with torch.no_grad():
        for batch in testloader:
            images, labels, bd_gts, _, _ = batch
            size = labels.size()
            images = images.cuda()
            labels = labels.long().cuda()
            bd_gts = bd_gts.float().cuda()

            pred = model(images)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]

            for i in range(len(pred)):
                pred[i] = F.interpolate(pred[i], size=size[-2:], mode='bilinear', align_corners=align_corners)

            # Compute losses
            aux_sum_loss = None
            if len(pred) != 1: # if use aux heads
                aux_pred, pred, aux_bd_pred = pred
                pred = [pred]

                aux_bd_loss = bd_criterion(aux_bd_pred, bd_gts).mean()
                aux_loss = sem_criterion(aux_pred, labels).mean()

                filler = torch.ones_like(labels) * ignore_label
                bd_label = torch.where(torch.sigmoid(aux_bd_pred[:, 0, :, :]) > 0.8, labels, filler)
                aux_sem_bd_loss = sem_criterion(pred[0], bd_label).mean()

                aux_sum_loss = aux_loss + aux_bd_loss + aux_sem_bd_loss

            task_loss = sem_criterion(pred[0], labels).mean()

            loss = task_loss
            if aux_sum_loss:
                loss = task_loss + aux_sum_loss

            ave_loss.update(loss.item())

            # Compute metrics
            our_metrics.update(pred[0], labels)

            for single_pred, single_label in zip(pred[0], labels):
                org_metrics.reset()
                single_pred = postproc(single_pred[None, ...].cpu().argmax(dim=1).numpy())
                org_metrics.update(single_label[None, ...].cpu().numpy(), single_pred)
                single_batch_org_mF1, _ = org_metrics.get_scores()
                org_mF1 += single_batch_org_mF1 * 100

    _, mean_F1 = our_metrics.compute_f1()
    _, mean_IoU = our_metrics.compute_iou()
    org_mF1 = org_mF1 / len(testloader) / pred[0].shape[0]

    writer.add_scalar(f'mIoU/{phase}_mIoU', mean_IoU, epoch)
    writer.add_scalar(f'mF1/{phase}_F1', mean_F1, epoch)
    writer.add_scalar(f'mF1/org_{phase}_F1', org_mF1, epoch)

    if phase == 'valid':
        writer.add_scalar('Loss/valid_loss', ave_loss.average(), epoch)
    
    return mean_IoU, mean_F1, org_mF1


def test(model, validloader, device, config):
    num_classes = config.DATASET.NUM_CLASSES
    align_corners = config.MODEL.ALIGN_CORNERS

    model = model.to(device)
    model.eval()
    org_metrics = AccuracyTracker(num_classes)

    full_score = 0.0
    with torch.no_grad():
        for img, lbl, _, _, _ in tqdm(validloader):
            img, lbl = img.to(device), lbl.to(device)
            pred = model(img)

            if isinstance(pred, (list, tuple)): # if use aux heads
                pred = pred[1]
            pred = F.interpolate(pred, size=lbl.shape[-2:], mode='bilinear', align_corners=align_corners)

            org_metrics.reset()
            pred = postproc(pred.cpu().argmax(dim=1).numpy())
            org_metrics.update(lbl.cpu().numpy(), pred)
            score, _ = org_metrics.get_scores()
            full_score += score
    full_score /= len(validloader)

    org_mF1 = float(full_score) * 100

    return org_mF1