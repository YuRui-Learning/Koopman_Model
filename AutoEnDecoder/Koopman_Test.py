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
import pickle
import scipy.io

starttime = time.time()
y1 = [] # 可视化loss list
y2 = [] # 精度person_cof list
loss_min = 10 # 定位最小误差
early_stop_flag = False
torch.manual_seed(1)
GPU_Flag = torch.cuda.is_available() # 检测GPU是否有效


# 模型示例化，自编码器模型
if GPU_Flag:
    Coder = AutoEncoderModel.AutoEncoder().cuda()
    print("model training on GPU")
else:
    Coder = AutoEncoderModel.AutoEncoder()
    print("model training on CPU")



def train(Coder):
    '''
    训练自编码器模型成pkl
    Cober 自编码器模型
    '''
    # 实例化模型
    Coder = Coder
    # 加载模型参数
    Coder = torch.load('checkpoint/AutoEncoder12.pkl')
    # 得到解码器信息
    print("编码器内容")
    print(list(Coder.encoder.parameters()))
    print("解码器内容")
    print(list(Coder.decoder.parameters()))
    # pkl转mat
    file = open(r'checkpoint/AutoEncoder12.pkl','rb')

    # 得到MATA MATB
    test_data_dict = getdata.Get_data('Koopman_500.csv') # 加载所有数据 21个为一列数据，共11组为一个list，共20000组左右
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
    MATA , MATB =train(Coder)
    print(MATA.shape,MATB.shape) # 返回A B 矩阵 A 是150*150 B是1*150

    data = pd.DataFrame(MATA.detach().numpy())
    writer = pd.ExcelWriter('checkpoint/excel/MATA.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()

    data = pd.DataFrame(MATB.detach().numpy())
    writer = pd.ExcelWriter('checkpoint/excel/MATB.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
