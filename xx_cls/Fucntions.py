import torch
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import os,sys
import numpy as np
## import self-defined packages
import Datasets
from torch.utils.data import DataLoader


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