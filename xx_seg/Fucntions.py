import torch
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import os,sys
import numpy as np
## import self-defined packages
import Datasets
from torch.utils.data import DataLoader

class EvaluateMetrics():
    def __init__(self):
        self.acc_dict = {
            "Accuracy": torchmetrics.Accuracy(),
            "IoU": torchmetrics.IoU(2),
        }
        self.step = 0
        self.sum_acc_dict = {key:0 for key in self.acc_dict.keys()}
    def update(self, y_hat, y): # on batch
        self.step += 1
        res = {key: None for key in self.acc_dict.keys()}
        for key in self.acc_dict.keys():
            res[key] = self.acc_dict[key](y_hat, y)
        for key in self.acc_dict.keys():
            self.sum_acc_dict[key]+=res[key]
        return res
    def compute_on_batch(self):
        res = {key: 0 for key in self.acc_dict.keys()}
        for key in self.acc_dict.keys():
            res[key] = self.sum_acc_dict[key] * 1.0 / self.step
        return res
    def compute(self): # on epoch
        self.res = {key:None for key in self.acc_dict.keys()}
        for key in self.acc_dict.keys():
            self.res[key] = self.acc_dict[key].compute()
        return self.res
    def reset(self): #reset in the end of epoch
        self.step = 0
        for key in self.acc_dict.keys():
            self.acc_dict[key].reset()
        for key in self.acc_dict.keys():
            self.sum_acc_dict[key] = 0
    def get_acc_list(self):
        print(self.acc_dict.keys())
    def to_gpu(self, Opt):
        for key in self.acc_dict.keys():
            self.acc_dict[key].to(torch.device(Opt.device))
        return self


class PostProcess():
    def __init__(self, Opt):
        self.log_dir = Opt.log_dir
    def seg_save(self, outputs, names):
        # outputs: batch*channels*size1*size2 ; numpy格式
        if not os.path.exists(os.path.join(self.log_dir,"predict")):
            os.makedirs(os.path.join(self.log_dir,"predict"))
        if len(outputs.shape) == 3:
            out = grey2rgb(np.argmax(outputs, axis=0))
            cv2.imwrite( os.path.join(self.log_dir,"predict",names+".png"), out)
        elif len(outputs.shape)==4:
            for i in range(outputs.shape[0]):
                out = grey2rgb(np.argmax(outputs[i], axis=0))
                cv2.imwrite( os.path.join(self.log_dir, "predict", names[i]+".png"), out)


def grey2rgb(image, color_dict={0:[1,1,1], 1:[255,255,255]}):
    color = np.ones([image.shape[0], image.shape[1], 3])
    for key in color_dict.keys():
        color[image == key] = color_dict[key]

    return color


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()