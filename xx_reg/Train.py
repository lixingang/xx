# built-in packages
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
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR,LambdaLR
import torchmetrics
from tensorboardX import SummaryWriter
import torchvision.utils as tvutils
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
# self-defined packages
import Models.mpi.dataset as dataset
import Models.mpi.loss as losses
import Models.mpi.metric as metric
import Models.mpi.net as net
from Utils.tools import *
from Functions import *
from Models.mpi.gp import GaussianProcess
def R2(y, y_, mv=None):
    y = y.view(-1).cpu().numpy()
    y_ = y_.view(-1).cpu().numpy()
    # print(y,y_)
    return r2_score(y, y_)


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

def train(opt):
    device = torch.device(opt.device)
    model = net.build_net().to(device)
    print(model)
    if opt.restore_model is not None:
        model = torch.load(opt.restore_model)
    # print(model)
    criterion = losses.HEMLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    dataloader = get_dataloader(opt)
    best_metric = -99
    logging.info('----- start train -----')

    Acc = {
        "train":metric.EvaluateMetrics(),
        "valid":metric.EvaluateMetrics(),
        "test":metric.EvaluateMetrics()
    }

    GP = GaussianProcess(sigma=1, r_loc=1.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01)
    for epoch in range(opt.num_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        step = 0
        train_loc = []
        train_fea = []
        train_y = []
        model_information = {}
        for x, y, geo in dataloader["train"]:
            optimizer.zero_grad()
            # for i in x:
            #     print(i.shape)
            x = [i.to(device).float() for i in x]
            # print([torch.sum(torch.isnan(i)) for i in x])
            # print([torch.sum(torch.isinf(i)) for i in x])
            y = y.to(device).float()
            y_,m,s,f = model(*x)
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            Acc['train'].update(y_,y)

            train_loc.append(geo)
            train_fea.append(f.detach().cpu())
            train_y.append(y.detach().cpu())
            step += 1
            process_bar(step / (len(dataloader["train"].dataset) // opt.batch_size), start_str='train:',
                        udata=f'loss:{loss:.4f}')

        train_loc = torch.cat(train_loc)
        train_fea = torch.cat(train_fea)
        train_y = torch.cat(train_y)
        # print(train_y)
        res = Acc['train'].compute()
        logging.info(
            f'[train] {epoch + 1}/{opt.num_epochs} loss={epoch_loss / step:.4f} R2:{res["R2"]:.4f} RMSE:{res["RMSE"]:.4f} MAPE:{res["MAPE"]:.4f}')

        writer.add_scalars(f"train", {
            'R2': res["R2"], 'RMSE': res["RMSE"], 'MAPE': res["MAPE"],
        }, epoch)

        # 验证
        step = 0
        model.eval()
        epoch_loss = 0
        valid_loc = []
        valid_fea = []
        valid_y = []
        valid_y_ = []
        for x, y, geo in dataloader["valid"]:
            x = [i.to(device).float() for i in x]
            y = y.to(device).float()
            y_,m,s,f = model(*x)
            valid_fea.append(f.detach().cpu())
            valid_loc.append(geo)
            valid_y.append(y.detach().cpu())
            valid_y_.append(y_.detach().cpu())
            loss = criterion(y_, y)
            epoch_loss += loss.item()
            Acc['valid'].update(y_,y)
            step += 1
            process_bar(step / (len(dataloader["valid"].dataset) // opt.batch_size), start_str='valid:',
                        udata=f'loss:{loss:.4f}')
        valid_fea = torch.cat(valid_fea)
        valid_loc = torch.cat(valid_loc)
        valid_y = torch.cat(valid_y)
        valid_y_ = torch.cat(valid_y_)
        res = Acc['valid'].compute()
        logging.info(
            f'[valid] {epoch + 1}/{opt.num_epochs} loss={epoch_loss / step:.4f} R2:{res["R2"]:.4f} RMSE:{res["RMSE"]:.4f} MAPE:{res["MAPE"]:.4f}')

        writer.add_scalars(f"valid",{
            'R2': res["R2"], 'RMSE': res["RMSE"], 'MAPE': res["MAPE"],
        },epoch)

        torch.save(model.state_dict(), f"Logs/mpi_{epoch}.pkl")
        if res["R2"]>best_metric:
            best_metric = res['R2']
            logging.info(f"Save Best Model in Epoch {epoch}, Current/best r2 is {res['R2']:.3f}/{best_metric:.3f} ...")
            torch.save(model, f"Logs/best_model.pkl")

        Acc["train"].reset()
        Acc["valid"].reset()

        # GP
        # model_information['train_feat'] = train_fea.numpy()
        # model_information['test_feat'] = valid_fea.numpy()
        # model_information['train_loc'] = train_loc.detach().cpu().numpy()
        # model_information['test_loc'] = valid_loc.detach().cpu().numpy()
        # # model_information['train_years'] = train_year.detach().cpu().numpy()
        # # model_information['test_years'] = test_year.detach().cpu().numpy()
        # model_information['train_real'] = train_y.numpy().squeeze()
        # model_information['model_weight'] = model.state_dict()['MLP.out.weight'].detach().cpu().numpy()
        # model_information['model_bias'] = model.state_dict()['MLP.out.bias'].detach().cpu().numpy()
        # # for key in model_information.keys():
        # #     print(key,model_information[key].shape)
        
        # if epoch%1000==0:
        #     print("RUN GP")
        #     gp_pred = GP.run(  model_information['train_feat'],
        #                         model_information['test_feat'],
        #                         model_information['train_loc'],
        #                         model_information['test_loc'],
        #                         np.random.randint(2017,2020, size=len(train_loc)),
        #                         np.random.randint(2017,2020, size=len(valid_loc)),
        #                         model_information['train_real'],
        #                         model_information['model_weight'],
        #                         model_information['model_bias'])
        #     gp_pred = torch.tensor(gp_pred)
        # else:
        #     gp_pred = valid_y_
        # r2=R2(valid_y, gp_pred,)
        # logging.info(
        #      f'[valid-GP] {epoch + 1}/{opt.num_epochs} loss={epoch_loss / step:.4f} R2:{r2:.4f}')


        if epoch%4==0:
            #test
            step = 0
            model = torch.load("Logs/best_model.pkl")
            model.eval()
            epoch_loss = 0
            test_loc = []
            test_fea = []
            test_y = []
            test_y_ = []
            for x, y, geo in dataloader["test"]:
                x = [i.to(device).float() for i in x]
                y = y.to(device).float()
                y_,m,s,f  = model(*x)
                loss = criterion(y_, y)
                epoch_loss += loss.item()
                Acc['test'].update(y_,y)
                step += 1
                process_bar(step / (len(dataloader["test"].dataset) // opt.batch_size), start_str='test:', udata=f'loss:{loss:.4f}')
                test_fea.append(f.detach().cpu())
                test_loc.append(geo)
                test_y.append(y.detach().cpu())
                test_y_.append(y_.detach().cpu())

            test_fea = torch.cat(test_fea)
            test_loc = torch.cat(test_loc)
            test_y = torch.cat(test_y)
            res = Acc['test'].compute()
            logging.info(
                f'[test] loss={epoch_loss / step:.4f} R2:{res["R2"]:.4f} RMSE:{res["RMSE"]:.4f} MAPE:{res["MAPE"]:.4f}')

            # Acc["test"].reset()
            

            # GP
            model_information['train_feat'] = train_fea.numpy()
            model_information['test_feat'] = test_fea.numpy()
            model_information['train_loc'] = train_loc.detach().cpu().numpy()
            model_information['test_loc'] = test_loc.detach().cpu().numpy()
            # model_information['train_years'] = train_year.detach().cpu().numpy()
            # model_information['test_years'] = test_year.detach().cpu().numpy()
            model_information['train_real'] = train_y.numpy().squeeze()
            model_information['model_weight'] = model.state_dict()['MLP.out.weight'].detach().cpu().numpy()
            model_information['model_bias'] = model.state_dict()['MLP.out.bias'].detach().cpu().numpy()
            # for key in model_information.keys():
            #     print(key,model_information[key].shape)
            
            print("RUN GP")
            gp_pred = GP.run(  model_information['train_feat'],
                                model_information['test_feat'],
                                model_information['train_loc'],
                                model_information['test_loc'],
                                np.random.randint(2017,2020, size=len(train_loc)),
                                np.random.randint(2017,2020, size=len(test_loc)),
                                model_information['train_real'],
                                model_information['model_weight'],
                                model_information['model_bias'])
            gp_pred = torch.tensor(gp_pred)
            r2=R2(test_y, gp_pred,)
            logging.info(
                    f'[test-GP] {epoch + 1}/{opt.num_epochs} loss={epoch_loss / step:.4f} R2:{r2:.4f}')
    
    
    return res
def get_list(Opt):
    file_list = os.listdir(Opt.data_dir)
    file_list = [os.path.join(Opt.data_dir, i) for i in file_list if "npy" in i]
    # random_list = list(range(len(file_list)))
    # random.shuffle(indexs)
    # res = randsplit(Opt.ratio, file_list)
    return np.array(file_list)

if __name__=='__main__':
    # parser = argparse.ArgumentParser(description="Demo of argparse")
    # parser.add_argument('-n','--name', default=' Li ')
    # parser.add_argument('-y','--year', default='20')
    # args = parser.parse_args()

    setup_seed(0)
    opt = Opt()
    kf = KFold(n_splits=4, shuffle=True, random_state=0)
    file_list = get_list(opt)

    i = 1
    all_res = open("all_res.txt","w")
    for train_index, test_index in kf.split(file_list):
        print("-"*10, i, "-"*10)
        X_tv, X_test = file_list[train_index], file_list[test_index]
        X_train, X_valid = randsplit([0.7,0.3], X_tv)
        opt.train_list = X_train
        opt.valid_list = X_valid
        opt.test_list = X_test
        res = train(opt)
        all_res.write(f"R2:{res['R2']:.4f} RMSE:{res['RMSE']:.4f} MAPE:{res['MAPE']:.4f} \n")
        i+=1
        break
    all_res.close()
    # train(opt)


