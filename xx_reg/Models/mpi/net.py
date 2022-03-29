# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:32:17 2020

@author: sjtcwdsj
"""
import torch.nn as nn
import torch
import numpy as np
import torch.distributions as tdist

import math

cfg = {
        'model':{'name':'Fea','loop':1},

        #数据参数
        #'locvar':[5,6,7,8,9,10,11,12,13,14,15,16,17,18],

        #模型参数
        'Fea':{
            'CNet1':[[64,True,6,12,36,3],],
            'CNet2':[[64,True,3,12,36,3],],
            'CNet3':[[64,True,1,12,36,3],],
            'LNet1':[[1,16],{'bias':True}],
            # 'LNet3':[[49,64],{'bias':True}],
            },
        'MLP1':{'config':[64,64,32]},#第一部分的MLP参数
        'MLP2':{'config':[128,128,128]},#第二部分的MLP参数
        'RNN':{'hidden_size':128,'num_layers':3,'GRU':True},
    }
# conv+gap

def build_net():
    fea = Fea(cfg['Fea'])
    mlp = MLP(fea.fea_len, *cfg["MLP1"]["config"])
    model = Fea_E(fea, mlp)
    return model

class CNet(nn.Module):
    def __init__(self, out, bias, in_ch, out_ch, out_1, kernel, **kwargs):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, out_1, 1, padding=0, **kwargs)
        self.conv1 = nn.Conv2d(out_1, out_1, kernel, padding=int((kernel - 1) / 2), **kwargs)
        self.conv2_1 = nn.Conv2d(out_1, out_1, 1, padding=int((1 - 1) / 2), **kwargs)
        self.conv2_2 = nn.Conv2d(out_1, out_1, 3, padding=int((3 - 1) / 2), **kwargs)
        self.conv2_3 = nn.Conv2d(out_1, out_1, 5, padding=int((5 - 1) / 2), **kwargs)
        self.conv3 = nn.Conv2d(out_1*3, out_ch, kernel, padding=int((kernel - 1) / 2), **kwargs)
        self.tanh = nn.Tanh()
        self.L = nn.Linear(out_ch, out, bias=bias)

        self.fea_len = out

    def forward(self, x):
        x = self.conv0(x)
        x = self.tanh(x)
        x = self.conv1(x)
        x = self.tanh(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        # print(x1.shape,x2.shape,x3.shape )
        x = torch.cat((x1, x2, x3), 1)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.tanh(x)

        x = x.mean(-1).mean(-1)
        x = self.L(x)

        return x


class LNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = nn.Linear(*args, **kwargs)
        self.fea_len = self.L.out_features

    def forward(self, x):
        return self.L(x)


class Fea(nn.Module):
    # 定义搜索空间
    __M_SEARCH_SET__ = [nn, ]

    def __init__(self, par: str):
        super().__init__()
        self._n_modules = len(par)
        keys = par.keys()
        # self.dictpart = dictpart
        # 添加一层激活函数
        # self.m = nn.ReLU(inplace=True)
        # if dictpart:
        #     self.embeding = nn.Embedding(dictnum, dictsize)

        for i, c in enumerate(keys):

            var = par[c]

            c = c[:-1]
            # 初始化搜索空间
            _search_set = [globals()] + Fea.__M_SEARCH_SET__

            if (len(var) == 1):
                var.append({})

            for s in _search_set:
                if hasattr(s, c):
                    setattr(self, f"f_{i}", getattr(s, c)(*var[0], **var[1]))
                    break
                elif c in s:
                    setattr(self, f"f_{i}", s[c](*var[0], **var[1]))
                    break
            else:
                raise ValueError(f"未知的类型：{c}")
        self.fea_len = 0
        for i in range(self._n_modules):
            self.fea_len += getattr(self, f"f_{i}").fea_len
        # if dictpart:
        #     self.fea_len += dictsize

    def forward(self, *args):
        # 输入数据按顺序进入网络，并拼接起来

        # y=None
        for i, p in enumerate(args):
            # if torch.sum(p!=p)!=0:
            #     print(i,p.shape,torch.sum(p!=p))
            if i == 0:
                y = getattr(self, f"f_{(i) % self._n_modules}")(p)
            else:
                # if len(y.shape) == 3:
                #     y = torch.squeeze(y, 1)
                if len(p.shape)==3:
                    p = torch.squeeze(p,1)
                y = torch.cat((y, getattr(self, f"f_{(i) % self._n_modules}")(p)), 1)
        # 添加一层激活函数
        # y=self.m(y)

        # if torch.sum(y!=y)!=0:
        #     print(y.shape,torch.sum(y!=y))

        return y


# MLP模块
# 参数介绍:
# *embed_chs:多层感知机每层的输入
# 在模型的两个部分的两种使用方式的输入层数不同
class MLP(nn.Module):
    def __init__(self, fea_len, *embed_chs):
        super().__init__()
        # MLP模块
        a_ch = fea_len

        layers = []
        for b_ch in embed_chs:
            fc = nn.Linear(a_ch, b_ch)
            self.init_fc(fc)
            layers.append(nn.Sequential(fc, nn.Tanh(), ))
            a_ch = b_ch
        self.layers = nn.Sequential(*layers)
        
        # 结果输出层
        self.out = nn.Linear(a_ch, 1)
        # self.zdist = tdist.Normal(0, 1)
        # self.mean = nn.Linear(a_ch, 1)
        # self.std = nn.Linear(a_ch, 1)

    @staticmethod
    def init_fc(fc):
        """
        Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
        fc.weight.data.uniform_(-r, r)
        fc.bias.data.fill_(0)

    def forward(self, x):
        y = self.layers(x)

        # m = self.mean(y)
        # s = self.std(y)
        # z = self.zdist.sample((y.shape[0], 1)).to(y.device)
        out = self.out(y)
        return out, 0, 0, y


# 单年特征提取模块：feature_extraction
# 参数介绍：
# fea:特征模块
# MLP:多层感知机
class Fea_E(nn.Module):

    def __init__(self, Fea, MLP):
        super().__init__()

        self.Fea = Fea
        self.MLP = MLP

    @staticmethod
    def init_fc(fc):
        """
        Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
        fc.weight.data.uniform_(-r, r)
        fc.bias.data.fill_(0)

    def forward(self, *args):
        # 特征模块
        fea = self.Fea(*args)

        # MLP
        y, m, s, f = self.MLP(fea)

        return y, m, s, f


# RNN
# 参数介绍:
# hidden_size:隐藏层数\输出层数
# num_layers:循环层数
# GRU:为True时，采用GRU网络，为False时，采用LSTM网络
# class RNN(nn.Module):
#     def __init__(self, fea_len, hidden_size, num_layers, GRU=True):
#         super().__init__()
#
#         self.fea_all_len = hidden_size
#
#         # 输入的特征长度
#         input_size = fea_len
#
#         self.GRU = GRU
#         if self.GRU == True:
#             self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
#         else:
#             self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
#
#         self.dense = nn.Linear(hidden_size, hidden_size)
#
#     # x__:前年数据(batch,Fea.outsize)
#     # x_:去年数据(batch,Fea.outsize)
#     # x:当年数据(batch,Fea.outsize)
#     def forward(self, x__, x_, x):
#
#         # 增加一维，表示不同年份
#         x__ = x__.unsqueeze(1)
#         x_ = x_.unsqueeze(1)
#         x = x.unsqueeze(1)
#
#         # 在新的维度上拼接
#         y = torch.cat((x__, x_, x), 1)
#
#         # y.shape():(batch,seq_len,input_size)
#         y, _ = self.rnn(y)
#
#         y = y[:, -1, :]
#         y = self.dense(y)
#         return y


# 产量预估模块，yield prediction
# 参数介绍:
# Fea:特征模块
# RNN:RNN(GRU或LSTM)
# MLP:多层感知机
# fixFea:为True时,在训练时将Fea模块的参数固定
# class Yield_P(nn.Module):
#     def __init__(self, Fea, RNN, MLP, fixFea=True):
#         super().__init__()
#
#         self.Fea = Fea
#         # 固定Fea层的参数
#         if fixFea == True:
#             for p in self.parameters():
#                 p.requires_grad = False
#
#         self.RNN = RNN
#         self.MLP = MLP
#         self.lenvar = self.Fea._n_modules
#
#     # A~F:当年数据
#     # A_~F_:去年数据
#     # A__~F__:前年数据
#     def forward(self, *args):
#         # 特征提取模块1
#
#         x__ = self.Fea(args[-1], *args[0:self.lenvar])
#         x_ = self.Fea(args[-1], *args[self.lenvar:2 * self.lenvar])
#         x = self.Fea(args[-1], *args[2 * self.lenvar:3 * self.lenvar])
#
#         # RNN模块(LSTM或GRU)
#         y = self.RNN(x__, x_, x)
#
#         # MLP模块
#         y, m, s, f = self.MLP(y)
#
#         return y, m, s, f


# # 构建SPP层(空间金字塔池化层)
# class SPPLayer(torch.nn.Module):

#     def __init__(self, num_levels, pool_type='max_pool'):
#         super(SPPLayer, self).__init__()

#         self.num_levels = num_levels
#         self.pool_type = pool_type

#     def forward(self, x):
#         num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
#         for i in range(self.num_levels):
#             level = i+1
#             kernel_size = (math.ceil(h / level), math.ceil(w / level))
#             stride = (math.ceil(h / level), math.ceil(w / level))
#             pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

#             # 选择池化方式
#             if self.pool_type == 'max_pool':
#                 tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
#             else:
#                 tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

#             # 展开、拼接
#             if (i == 0):
#                 x_flatten = tensor.view(num, -1)
#             else:
#                 x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)


#         return x_flatten


if __name__ == "__main__":
    # __M_SEARCH_SET__=[nn,]
    cfg = {
        'model':{'name':'Fea','loop':1},

        #数据参数
        #'locvar':[5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        'locvar':[5,6,7,8,9,10,11,12,13,14,15,16,17],
        'locy':4,
        'locpath':4,
        'quality':11,

        #模型参数
        'Fea':{
            'CNet1':[[64,True,21,7,7,3],{'groups':7}],
            'LNet2':[[28,64],{'bias':True}],
            'LNet3':[[49,64],{'bias':True}],
            'CNet2':[[8,True,1,8,8,3],{'groups':1}],
            'LNet4':[[210,8],{'bias':True}],
            'LNet5':[[210,8],{'bias':True}],
            'LNet6':[[210,8],{'bias':True}],
            'LNetA':[[2,8],{'bias':True}],
            'LNetB':[[4,8],{'bias':True}],
            },
        'MLP1':{'config':[128,128,128]},#第一部分的MLP参数
        'MLP2':{'config':[128,128,128]},#第二部分的MLP参数
        'RNN':{'hidden_size':128,'num_layers':3,'GRU':True},
    }
    fea = Fea(cfg['Fea'])
    mlp = MLP(fea.fea_len, *cfg["MLP1"]["config"])
    model = Fea_E(fea, mlp)
    print(model)
