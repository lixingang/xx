import argparse
from Options import PredictOpt
import os
import sys
import numpy as np
sys.path.append(os.getcwd())

import logging
import subprocess

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

import Datasets
import models.cls as clsmodels
import models.seg as segmodels
from utils.tools import *
from Fucntions import EvaluateMetrics, PostProcess


logging.basicConfig(
    # filename='new.log', filemode='w',
    format='%(asctime)s  %(levelname)-10s %(processName)s  %(name)s \033[0;33m%(message)s\033[0m',
    datefmt="%Y-%m-%d-%H-%M-%S",
    level=logging.DEBUG if os.environ.get("DEBUG") is not None else logging.INFO
)

def get_dataloader(Opt):
    if Opt.is_mask:
        DATASET = Datasets.berryTrainDataset(
            images_dir="/data/ly/dl_data/changguang/sample/img_dir/train",
            masks_dir="/data/ly/dl_data/changguang/sample/ann_dir/val",
            classes=Opt.classes,
        )
    else:
        DATASET = Datasets.berryPredictDataset(
            images_dir="/data/ly/dl_data/changguang/sample/img_dir/train",
            classes=Opt.classes,
        )
    dataloader = DataLoader(
        dataset=DATASET,
        batch_size=Opt.batch_size,
        shuffle=True,
        num_workers=Opt.num_workers,
        pin_memory=False,
        drop_last=False
    )
    logging.info(f'----- data loading -----')
    logging.info(f'size={len(DATASET)}')

    return dataloader


def predict_seg(Opt):

    device = torch.device(Opt.device)
    model = segmodels.UnetPlusPlus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=len(Opt.classes),
        activation='sigmoid',
    ).to(device)
    model = torch.load_state_dict(Opt.restore_model).to(device)
    dataloader = get_dataloader(Opt)
    processor = PostProcess(Opt)
    if Opt.is_mask is False:

        step = 0
        model.eval()
        for x, names in dataloader:
            x = x.to(device).float()
            outputs = model(x)
            print(outputs[0,:,200,200])
            step += 1
            processor.seg_save(outputs.detach().cpu().numpy(), names)
            process_bar(step / (len(dataloader.dataset) // Opt.batch_size))


    elif Opt.is_mask is True:

        step = 0
        model.eval()
        Acc = EvaluateMetrics()
        criterion = segmodels.losses.FocalLoss("binary")
        epoch_loss = 0
        for x, y, names in dataloader:
            x = x.to(device).float()
            y = y.to(device).long()
            outputs = model(x)
            loss = criterion(outputs, y.long())
            epoch_loss += loss.item()
            Acc.update(outputs, y)
            step += 1
            processor.seg_save(outputs.detach().cpu().numpy(), names)
            process_bar(step / (len(dataloader.dataset) // Opt.batch_size), start_str='predict:',
                        udata=f'loss:{loss:.4f}')

        acc_epoch = Acc.compute_on_batch()
        logging.info(
            f'[predict] loss={epoch_loss / step:.4f} Acc={acc_epoch["Accuracy"]:.4f} IoU={acc_epoch["IoU"]:.4f}')

        Acc.reset()


if __name__=='__main__':
    Opt = PredictOpt()
    predict_seg(Opt)
