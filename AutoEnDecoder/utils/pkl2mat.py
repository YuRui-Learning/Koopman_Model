import torch
import numpy as np
import scipy.io as sio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__,"..","..")))


def transform(pkl_name):
    '''
    pkt 转 mat用于matlab部分
    pkl_name : pkl文件名称
    '''
    data_path = os.path.join('../checkpoint', pkl_name)
    # file = open(r'../checkpoint/AutoEncoder12.pkl','rb')
    file = open(data_path,'rb')
    model = torch.load(file)
    encoder= model._modules['encoder']
    print(encoder)
    decoder= model._modules['decoder']
    print(decoder)
    weight = dict()
    print(list(model.encoder.parameters())[0])
    weight['Linera1'] = list(model.encoder.parameters())[0].data.cpu().numpy()
    weight['Relu1'] = list(model.encoder.parameters())[1].data.cpu().numpy()
    weight['Linera2'] = list(model.encoder.parameters())[2].data.cpu().numpy()
    weight['Relu2'] = list(model.encoder.parameters())[3].data.cpu().numpy()
    weight['Linera3'] = list(model.encoder.parameters())[4].data.cpu().numpy()
    sio.savemat('../checkpoint/mat/encoder.mat', mdict=weight)

    weight = dict()
    weight['Linera1'] = list(model.decoder.parameters())[0].data.cpu().numpy()
    weight['Relu1'] = list(model.decoder.parameters())[1].data.cpu().numpy()
    weight['Linera2'] = list(model.decoder.parameters())[2].data.cpu().numpy()
    weight['Relu2'] = list(model.decoder.parameters())[3].data.cpu().numpy()
    weight['Linera3'] = list(model.decoder.parameters())[4].data.cpu().numpy()
    weight['Relu3'] = list(model.decoder.parameters())[5].data.cpu().numpy()
    weight['Linera4'] = list(model.decoder.parameters())[6].data.cpu().numpy()
    weight['Sigmoid'] = list(model.decoder.parameters())[7].data.cpu().numpy()
    sio.savemat('../checkpoint/mat/decoder.mat', mdict=weight)

    weight = dict()
    a = model.matrix_A
    weight['matrixA'] = model.matrix_A.data.cpu().numpy()
    weight['matrixB'] = model.matrix_B.data.cpu().numpy()
    sio.savemat('../checkpoint/mat/matrix.mat', mdict=weight)

if __name__ == "__main__":
    pkl_name = 'AutoEncoder12.pkl'
    transform(pkl_name)