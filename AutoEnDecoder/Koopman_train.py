import numpy as np
import torch
import torch.nn as nn
from model import AutoEncoderModel
from data import getdata
from utils import lossfunction
from utils import view
import time
from tqdm import tqdm
import argparse
import pandas as pd
starttime = time.time()

y1 = [] # 可视化loss list
y2 = [] # 精度person_cof list
loss_min = 10 # 定位最小误差
early_stop_flag = False
torch.manual_seed(1)
GPU_Flag = torch.cuda.is_available() # 检测GPU是否有效

# 参数配置
parser = argparse.ArgumentParser(description='parser example')
parser.add_argument('--epoch', default=100, type=int, help='Train Epoch')
parser.add_argument('--lr', default=0.0025, type=float, help='learning rate')
parser.add_argument('--batch_size', default=25, type=int, help='batch size')
parser.add_argument('--train_flag', default=True, type=bool, help='whehter train or not ')
args = parser.parse_args()

# 查看GPU使用
if GPU_Flag:
    Coder = AutoEncoderModel.AutoEncoder().cuda()
    print("model training on GPU")
else:
    Coder = AutoEncoderModel.AutoEncoder()
    print("model training on CPU")



def train(Coder,args):
    '''
    训练自编码器模型成pkl
    Cober 自编码器模型
    args 配置参数
    '''
    # 读取数据
    data_dict = getdata.Get_data('Koopman_20000.csv')
    print(len(data_dict))
    # 读取配置参数
    EPOCH = args.epoch
    LR = args.lr
    batch_size = args.batch_size
    Train_Flag = args.train_flag
    # 优化器选择
    optimizer = torch.optim.Adamax(Coder.parameters(), lr=LR)
    # 自编码器训练过程
    if Train_Flag:
        for epoch in range(EPOCH):
            loss = 0
            cache_data = 0
            loss_mean = 0
            epoch_start_time = time.time()
            for column in tqdm(range(len(data_dict))): # column表示数据第几列
                loss += lossfunction.loss_compute(data_dict[column], Coder, GPU_Flag) * 5 # 存一个batch的数据
                cache_data += 1
                # print("column",column)
                # print("cache_data:%d"%cache_data)
                # print("loss:%d", loss)
                if cache_data >= batch_size: # 一个batch size 反向传播一次wa
                    # if torch.any(torch.isnan(loss)):
                    #     loss = 0
                    #     continue
                    loss_mean += loss.item() / len(data_dict) # 会导致内存一直增加，直到最后allocate error
                    # print("column", column)
                    # print(loss_mean)
                    optimizer.zero_grad() # 梯度清零
                    loss.backward(retain_graph=True) # 反向传播
                    optimizer.step() # 优化器迭代
                    loss = 0 # 清空loss缓存
                    cache_data = 0 # 缓存计数
            if epoch >= 0:
                epoch_end_time = time.time()
                print('Epoch :', epoch,'|','train_loss:%.4f'%loss_mean,'|','take_time:%.2f'%(epoch_end_time - epoch_start_time))
            y1.append(loss_mean) # 维系误差loss list 用于可视化训练误差
            y2.append(loss_mean) # 维系相关系数cof list 用于可视化训练误差
        # 可视化误差
        view.perform_loss_cof(y1,y2)
        print(len(y1))

        if early_stop_flag :
            print("------early stopping ----------")

        # 保存Cober模型
        torch.save(Coder,'checkpoint/AutoEncoder.pkl')
        print('________________finish training___________________')
        endtime = time.time()
        print('训练耗时：',(endtime - starttime))
    # 实例化模型
    Coder = Coder
    # 加载模型参数
    Coder = torch.load('checkpoint/AutoEncoder12.pkl')
    # 得到解码器信息
    print("编码器内容")
    print(list(Coder.encoder.parameters()))
    print("解码器内容")
    print(list(Coder.decoder.parameters()))
    # 得到MATA MATB
    test_data_dict = getdata.Get_data('Koopman_500.csv')
    loss_mean = 0
    for column in tqdm(range(len(test_data_dict))):  # column表示数据第几列
        loss = lossfunction.loss_compute(test_data_dict[column], Coder, GPU_Flag)  # 存一个batch的数据
        loss_mean += loss.item() / len(test_data_dict)  # 会导致内存一直增加，直到最后allocate error

    print("test_lose_mean",loss_mean)
    l = [i for i in range(21)]
    if GPU_Flag:
        X_state =torch.tensor(l).to(torch.float32).cuda()
        U = torch.tensor(1).to(torch.float32).cuda()
    else:
        X_state =torch.tensor(l).to(torch.float32)
        U = torch.tensor(1).to(torch.float32)
    _, _, MATA ,MATB = Coder(X_state,U,True) # 用一个列子引出MATA MATB

    # 返回所要的两个矩阵
    return MATA ,MATB

if __name__ == "__main__":
    MATA , MATB =train(Coder, args)
    print(MATA.shape,MATB.shape) # 返回A B 矩阵 A 是150*150 B是1*150

    data = pd.DataFrame(MATA.detach().numpy())
    writer = pd.ExcelWriter('checkpoint/excel/MATA.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()

    data = pd.DataFrame(MATB.detach().numpy())
    writer = pd.ExcelWriter('checkpoint/excel/MATB.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
