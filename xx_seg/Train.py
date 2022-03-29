import argparse
from Options import TrainOpt
import os
import sys
import numpy as np
sys.path.append(os.getcwd())

import logging
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR,LambdaLR
import torchmetrics
from tensorboardX import SummaryWriter
import torchvision.utils as tvutils
# self-defined packages
import Datasets
import models.cls as clsmodels
import models.seg as segmodels
from Fucntions import EvaluateMetrics
from utils.tools import *

writer = SummaryWriter()

logging.basicConfig(
    # filename='new.log', filemode='w',
    format='%(asctime)s  %(levelname)-10s %(processName)s  %(name)s \033[0;33m%(message)s\033[0m',
    datefmt="%Y-%m-%d-%H-%M-%S",
    level=logging.DEBUG if os.environ.get("DEBUG") is not None else logging.INFO
)

def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def get_dataloader(Opt):
    DATASET = None #定义数据读取方式
    # Dataset = datasets.segDatasets
    DATASET = {
        "train": Datasets.berryTrainDataset(
            images_dir="/data/ly/dl_data/changguang/sample/img_dir/train",
            masks_dir="/data/ly/dl_data/changguang/sample/ann_dir/train",
            # augmentation=Datasets.get_training_augmentation(),
            classes = Opt.classes,
        ),
        "valid": Datasets.berryTrainDataset(
            images_dir="/data/ly/dl_data/changguang/sample/img_dir/val",
            masks_dir="/data/ly/dl_data/changguang/sample/ann_dir/val",
            classes=Opt.classes,
        ),
        "test": Datasets.berryTrainDataset(
            images_dir="/data/ly/dl_data/changguang/sample/img_dir/val",
            masks_dir="/data/ly/dl_data/changguang/sample/ann_dir/val",
            classes=Opt.classes,
        ),
    }

    dataloaders = {}
    for i in ["train","valid","test"]:
        # Tip:通过"train""valid""test"三个关键词进行索引，
        #     如果需要定制其他的参数，也可以仿照dataset的做法
        dataloaders[i] = DataLoader(
                dataset=DATASET[i],
                batch_size=Opt.batch_size,
                shuffle=True,
                num_workers=Opt.num_workers,
                pin_memory=False,
                drop_last=False
        )
    logging.info(f'----- data loading -----')
    for i in ["train", "valid", "test"]:
        logging.info(f'{i}size={len(DATASET[i])}')

    return dataloaders

def train_seg(Opt):
    device = torch.device(Opt.device)
    # model = segmodels.FPN(
    #     encoder_name='se_resnext50_32x4d',
    #     encoder_weights=None,
    #     in_channels = 3,
    #     # encoder_weights='imagenet',
    #     classes=len(Opt.classes),
    #     activation= 'sigmoid',
    # ).to(device)
    model = segmodels.UnetPlusPlus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes = len(Opt.classes),
        activation='softmax',
    ).to(device)

    criterion = segmodels.losses.DiceLoss("multiclass",) #Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    dataloader = get_dataloader(Opt)

    logging.info('----- start train -----')

    Acc = {
        "train":EvaluateMetrics().to_gpu(Opt),
        "valid":EvaluateMetrics().to_gpu(Opt),
        "test":EvaluateMetrics().to_gpu(Opt)
    }

    for epoch in range(Opt.num_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        step = 0
        for x, y, name in dataloader["train"]:
            optimizer.zero_grad()
            x = x.to(device).float()
            y = y.to(device).float()

            y = torch.argmax(y, dim=1)

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            acc_on_batch = Acc['train'].update(outputs,y)



            step += 1
            process_bar(step / (len(dataloader["train"].dataset) // Opt.batch_size), start_str='train:',
                        udata=f'loss:{loss:.4f} acc:{acc_on_batch["Accuracy"]:.4f} iou:{acc_on_batch["IoU"]:.4f}')

        acc_epoch = Acc['train'].compute_on_batch()
        logging.info(
            f'[train] {epoch + 1}/{Opt.num_epochs} loss={epoch_loss / step:.4f} Acc={acc_epoch["Accuracy"]:.4f} IoU={acc_epoch["IoU"]:.4f}')


        # 验证
        step = 0
        model.eval()
        epoch_loss = 0
        for x, y, name in dataloader["valid"]:
            x = x.to(device).float()
            y = y.to(device).float()
            y = y.argmax(1)
            outputs = model(x)

            Acc['valid'].update(outputs,y)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()
            acc_on_batch = Acc['valid'].update(outputs,y)

            writer.add_image('x', tvutils.make_grid(x), step)
            writer.add_image('y', tvutils.make_grid(y), step)
            writer.add_image('out', tvutils.make_grid(outputs), step)

            step += 1
            process_bar(step / (len(dataloader["valid"].dataset) // Opt.batch_size), start_str='valid:',
                        udata=f'loss:{loss:.4f} acc:{acc_on_batch["Accuracy"]:.4f} iou:{acc_on_batch["IoU"]:.4f}')
        acc_epoch = Acc['valid'].compute_on_batch()
        logging.info(
            f'[valid] {epoch + 1}/{Opt.num_epochs} loss={epoch_loss / step:.4f} Acc={acc_epoch["Accuracy"]:.4f} IoU={acc_epoch["IoU"]:.4f}')

        Acc["train"].reset()
        Acc["valid"].reset()

        torch.save(model.state_dict(), f"logs/unetplus_{epoch}.pkl")

    #test
    step = 0
    model.eval()
    epoch_loss = 0
    for x, y, name in dataloader["test"]:
        x = x.to(device).float()
        y = y.to(device).long()
        y = y.argmax(1)
        outputs = model(x)
        loss = criterion(outputs, y.long())
        epoch_loss += loss.item()
        Acc['test'].update(outputs,y)
        step += 1
        process_bar(step / (len(dataloader["test"].dataset) // Opt.batch_size), start_str='test:', udata=f'loss:{loss:.4f}')

    acc_epoch = Acc['test'].compute_on_batch()
    logging.info(
        f'[test] loss={epoch_loss / step:.4f} Acc={acc_epoch["Accuracy"]:.4f} IoU={acc_epoch["IoU"]:.4f}')

    Acc["test"].reset()





if __name__=='__main__':
    # parser = argparse.ArgumentParser(description="Demo of argparse")
    # parser.add_argument('-n','--name', default=' Li ')
    # parser.add_argument('-y','--year', default='20')
    # args = parser.parse_args()
    Opt = TrainOpt()
    train_seg(Opt)


