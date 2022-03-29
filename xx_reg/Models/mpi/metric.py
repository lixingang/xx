import torch
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import os,sys
import numpy as np
from sklearn.metrics import r2_score

class EvaluateMetrics():
    def __init__(self):
        # 定义metric的字典集合
        self.y = []
        self.y_ = []
        self.step = 0
        self.res = {
            'R2': 0,
            'RMSE': 0,
            'MAPE': 0
        }
        self.func = {
            'R2':self.R2,
            'RMSE':self.RMSE,
            'MAPE':self.MAPE
        }

    def update(self, y_hat, y):
        # 更新 on batch
        # y_hat: numpy.array
        # y: numpy.array
        self.step += 1
        self.y_.append(y_hat.detach().cpu().numpy())
        self.y.append(y.detach().cpu().numpy())

    def compute(self):
        # 返回每个epoch的精度(基于batch求均值）
        for key in self.res.keys():
            self.res[key] = self.func[key](self.y,self.y_)
        return self.res

    def reset(self):
        # 重置sum，step
        self.step = 0
        for key in self.res.keys():
            self.res[key] = 0
        self.y = []
        self.y_ = []

    def RMSE(self, y, y_, mv=None):
        # y = y.view(-1).cpu().numpy()
        # y_ = y_.view(-1).cpu().numpy()
        y = y[0]
        y_ = y_[0]
        if mv is not None:
            y = y * mv["std"] + mv["mean"]
            y_ = y_ * mv["std"] + mv["mean"]
        return np.sqrt(((y - y_) ** 2).mean())

    def MAPE(self, y, y_, mv=None):
        y = y[0]
        y_ = y_[0]
        # y = y.view(-1).cpu().numpy()
        # y_ = y_.view(-1).cpu().numpy()

        # if mv is not None:
        #     y = y * mv["std"] + mv["mean"]
        #     y_ = y_ * mv["std"] + mv["mean"]
        diff = np.abs(np.array(y) - np.array(y_))
        return np.mean(diff / y)

    def R2(self, y, y_, mv=None):
        y = y[0]
        y_ = y_[0]
        # print(len(y),len(y_))
        # print("y:",y,"y_",y_)
        # y = y.view(-1).cpu().numpy()
        # y_ = y_.view(-1).cpu().numpy()
        #
        # if mv is not None:
        #     y = y * mv["std"] + mv["mean"]
        #     y_ = y_ * mv["std"] + mv["mean"]
        return r2_score(y, y_)





class BasicEvaluateMetrics():
    def __init__(self):
        # 定义metric的字典集合
        self.acc_dict = {
            "Accuracy": torchmetrics.Accuracy(),
            "IoU": torchmetrics.IoU(2),
        }
        self.step = 0
        self.sum_acc_dict = {key:0 for key in self.acc_dict.keys()}

    def update(self, y_hat, y):
        # 更新 on batch
        # y_hat: tensor
        # y: tensor
        self.step += 1
        res = {key: None for key in self.acc_dict.keys()}
        for key in self.acc_dict.keys():
            res[key] = self.acc_dict[key](y_hat, y)
        for key in self.acc_dict.keys():
            self.sum_acc_dict[key]+=res[key]
        return res

    def compute_on_batch(self):
        # 返回每个batch最终的精度
        res = {key: 0 for key in self.acc_dict.keys()}
        for key in self.acc_dict.keys():
            res[key] = self.sum_acc_dict[key] * 1.0 / self.step
        return res

    def compute(self):
        # 返回每个epoch的精度(基于batch求均值）
        self.res = {key:None for key in self.acc_dict.keys()}
        for key in self.acc_dict.keys():
            self.res[key] = self.acc_dict[key].compute()
        return self.res

    def reset(self):
        # 重置sum，step
        self.step = 0
        for key in self.acc_dict.keys():
            self.acc_dict[key].reset()
        for key in self.acc_dict.keys():
            self.sum_acc_dict[key] = 0

    def get_acc_list(self):
        print(self.acc_dict.keys())
        
    def to_gpu(self, device):
        for key in self.acc_dict.keys():
            self.acc_dict[key].to(torch.device(device))
        return self
