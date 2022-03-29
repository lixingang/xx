import argparse
from Options import Opt
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

def get_dataloader(opt):

    DATASET, DATALOADER = {},{}
    for i in ["train","valid","test"]:
        # Tip:通过"train""valid""test"三个关键词进行索引，
        #     如果需要定制其他的参数，也可以仿照dataset的做法
        DATASET[i] = dataset.xxDataset(opt,i)
        DATALOADER[i] = DataLoader(
                dataset=DATASET[i],
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_workers,
                pin_memory=False,
                drop_last=False
        )
    logging.info(f'----- data loading -----')
    for i in ["train", "valid", "test"]:
        logging.info(f'{i}size={len(DATASET[i])}')

    return DATALOADER


def predict(opt):

    device = torch.device(opt.device)
    model = segmodels.UnetPlusPlus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=len(opt.classes),
        activation='sigmoid',
    ).to(device)
    model = torch.load_state_dict(opt.restore_model).to(device)
    dataloader = get_dataloader(opt)
    processor = PostProcess(opt)
    if opt.is_mask is False:

        step = 0
        model.eval()
        for x, names in dataloader:
            x = x.to(device).float()
            outputs = model(x)
            print(outputs[0,:,200,200])
            step += 1
            processor.seg_save(outputs.detach().cpu().numpy(), names)
            process_bar(step / (len(dataloader.dataset) // opt.batch_size))


    elif opt.is_mask is True:

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
            process_bar(step / (len(dataloader.dataset) // opt.batch_size), start_str='predict:',
                        udata=f'loss:{loss:.4f}')

        acc_epoch = Acc.compute_on_batch()
        logging.info(
            f'[predict] loss={epoch_loss / step:.4f} Acc={acc_epoch["Accuracy"]:.4f} IoU={acc_epoch["IoU"]:.4f}')

        Acc.reset()


if __name__=='__main__':
    opt = Opt()
    predict_seg(opt)
