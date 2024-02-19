
"""
Train and eval functions used in train.py
"""
import math
import sys
from typing import Iterable, Optional
import argparse

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import numpy as np
import utils
import logging


def to_one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot_targets = torch.zeros(targets.size(0), num_classes, device=targets.device)
    one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)
    return one_hot_targets


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args: argparse.ArgumentParser.parse_args = None):
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1 if args.debug else 400
    if args.update_temperature:
        model.module.update_temperature()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        predicted_labels = outputs.argmax(dim=1)
        true_labels = targets.argmax(dim=1)
        correct = (predicted_labels == true_labels).float()
        acc = correct.sum() / targets.size(0)
        metric_logger.update(accuracy=acc.item())
        if not math.isfinite(loss_value):
            logging.error("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Train averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, header='Test:'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="")
    all_preds, all_targets = [], []
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 200, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        all_preds.append(output.cpu())
        all_targets.append(target.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    class_acc = []
    _, predicted_labels = torch.max(all_preds, 1)
    for i in range(5):
        class_targets = (all_targets == i)
        class_predicted = (predicted_labels == i)
        if class_targets.sum() > 0:
            class_accuracy = np.logical_and(class_targets, class_predicted).float().sum() / class_targets.float().sum()
            class_acc.append(class_accuracy.item())
        else:
            class_acc.append(0.0)

    class_names = ['Mica', 'Detritus', 'Flint', 'Quartz', 'Feldspar']
    file_path = "./class_accuracy.txt"
    with open(file_path, "a") as file:
        file.write("==== Epoch{} ====\n".format(epoch))
        for i in range(len(class_names)):
            line = "{}: {:.4f}\n".format(class_names[i], class_acc[i])
            file.write(line)
        file.write("\n")
    print('Test Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}