import torch
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import os,sys
import numpy as np
## import self-defined packages
from torch.utils.data import DataLoader
import random

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

def randnorepeat(m,n):
    p=list(range(n))
    d=random.sample(p,m)
    return d

def randsplit(ratio,n):
    assert len(ratio)==2 or len(ratio)==3
    assert sum(ratio) == 1.0
    # 累加求和，例如 [0.7,0.15,0.15] 累计求和为 [0.7, 0.85, 1]
    # 然后剔除最后一个值，即结果为 [0.7, 0.85]
    split_index = np.cumsum(ratio).tolist()[:-1]

    # # 是否洗牌
    # if shuffle:
    #     # 不放回的从原始数据中随机取数据，直到到达frac设置的比例
    #     n = n.sample(frac=1.)
    # 切分
    splits = np.split(n, [round(x * len(n)) for x in split_index])

    res = []
    # 增加一个属于第几个切分部分的索引
    for i in range(len(ratio)):
        res.append(list(splits[i]))
    return res

if __name__=='__main__':
    from Options import Opt
    opt = Opt()
    res = get_list(opt)
    print(res)