import torch.nn as nn
import torch
import math

global input_callback_predict
global input_calback_recon


def loss_compute(data_list, Coder, GPU_Flag):
    """Define the (unregularized) loss functions for the training.
        Arguments:
            data_list -- 41 0-21 2-23，.... ,,20-41
            Cober 模型实例
            GPU_Flag 是否有GPU

        Attention:
            flag -- 自编码器计算两个不同步骤，True的时候经计算预测和线性误差，False时候计算重构误差

        Returns:
            predictloss -- 预测误差：下一状态的预测误差，通过给定u去预估下时刻的数据
            reconloss -- 重构误差：本书状态列表重现能力
            matrixloss -- 矩阵误差：在前向传播时候计算的一个矩阵损失，量化乘A B 前和乘A B 后差距
            infloss -- 无穷误差: 为predictloss和reconloss的无穷范数
            loss_value 加权综合
    """
    lose = {'predict_loss': 0, 'linear_loss': 0, 'recon_loss': 0, 'inf_loss': 0}

    for step in range(len(data_list) - 1):  # step表示预测第k步
        flag = True  # 计算predict_loss和linear_loss
        # 读取数据，根据是否有GPU
        if GPU_Flag:
            X_state = torch.from_numpy(data_list[step]).to(torch.float32).cuda()
            X_state_1 = torch.from_numpy(data_list[step + 1]).to(torch.float32).cuda()
        else:
            X_state = torch.from_numpy(data_list[step]).to(torch.float32)
            X_state_1 = torch.from_numpy(data_list[step + 1]).to(torch.float32)
        U = X_state_1[-2] # 得到输入U
        # 因为步数大于一时候，会将上一回调周期的输出做为该时刻输入，所以需要给input区分赋值
        if step == 0:
            input = X_state
        elif step > 0:
            input = input_callback_predict # 上一回调周期数据
        decoded, matrixloss, MATA, MATB = Coder(input, U, flag)
        input_callback_predict = decoded # 获得数据
        # 编码器输入和解码器输出的差距这个是缓存所有的
        lose['predict_loss'] += torch.sqrt(torch.mean(torch.pow((decoded - X_state_1), 2)))  # decoded,X_state_1 状态变量之间的差距
        lose['inf_loss'] += torch.norm(decoded - X_state_1, p=1)
        # 量化出 A , B 的作用，乘以A，B前，和乘以A,B后的误差 A * flpa(x) + B * U - flapa(x) -u
        lose['linear_loss'] += matrixloss  # 量化matrix作用
        flag = False
        if step == 0:
            input = X_state
        elif step > 0:
            input = input_calback_recon
        decoded = Coder(input, U , flag)
        input_calback_recon = decoded # 上一回调周期数据
        lose['recon_loss'] += torch.sqrt(torch.mean(torch.pow((decoded - X_state), 2)))  # decoded,X_state reconsturct loss 是自身预测自身的损失
        lose['inf_loss'] += torch.norm(decoded - X_state, p=1)  # L∞ inf loss
    # 所有误差都要除p步求平均
    lose['recon_loss'] = lose['recon_loss'] / step
    lose['inf_loss'] = lose['inf_loss'] / step
    lose['predict_loss'] = lose['recon_loss'] / step
    lose['linear_loss'] = lose['inf_loss'] / step



    loss_value = lose['predict_loss'] + lose['linear_loss'] + lose['recon_loss'] * 0.3 + lose['inf_loss'] * 1e-9

    return loss_value

# def loss_compute(data_list , Coder):
#     """Define the (unregularized) loss functions for the training.
#         Arguments:
#             data_list -- 41 0-20 2-22，.... ,,
#             X_state -- 输入原状态
#             X_state_1 -- 实际的下一时刻状态
#             matrixloss -- 在前向传播时候计算的一个矩阵损失，量化乘A B 前和乘A B 后差距
#
#         Returns:
#             loss_value 加权综合
#     """
#     lose = {'predict_loss': 0, 'linear_loss': 0, 'recon_loss': 0, 'inf_loss': 0}
#
#     # flag = True # 计算predict_loss和linear_loss
#     # X_state = torch.from_numpy(data_list[0]).to(torch.float32)
#     # X_state_1 = torch.from_numpy(predict_list[0]).to(torch.float32)
#     # decoded , matrixloss , _ , _  = Coder(X_state, X_state_1, flag)
#     # # 编码器输入和解码器输出的差距这个是缓存所有的
#     # lose['predict_loss'] += torch.sqrt(torch.mean(torch.pow((decoded - X_state_1), 2)))# Lx,x 状态变量之间的差距
#     # # 量化出 A , B 的作用，乘以A，B前，和乘以A,B后的误差 A * flpa(x) + B * U - flapa(x) -u
#     # lose['linear_loss'] = matrixloss # Lo,x，encoder_loss
#
#     for step in range(len(data_list) - 1):  # step表示预测第k步
#         flag = True  # 计算predict_loss和linear_loss
#         X_state = torch.from_numpy(data_list[step]).to(torch.float32)
#         X_state_1 = torch.from_numpy(data_list[step + 1]).to(torch.float32)
#         U = X_state_1[-2] # 得到输入U
#         decoded, matrixloss, MATA, MATB = Coder(X_state, U, flag)
#         # 编码器输入和解码器输出的差距这个是缓存所有的
#         lose['predict_loss'] += torch.sqrt(torch.mean(torch.pow((decoded - X_state_1), 2)))  # Lx,x 状态变量之间的差距
#         lose['inf_loss'] += torch.norm(decoded - X_state_1, p=1)
#         # 量化出 A , B 的作用，乘以A，B前，和乘以A,B后的误差 A * flpa(x) + B * U - flapa(x) -u
#         lose['linear_loss'] += matrixloss  # Lo,x，encoder_loss
#         flag = False
#         decoded = Coder(X_state, U , flag)
#         lose['recon_loss'] += torch.sqrt(torch.mean(torch.pow((decoded - X_state), 2)))  # Lo,x reconsturct loss，因为
#         lose['inf_loss'] += torch.norm(decoded - X_state_1, p=1)  # L∞ inf loss
#     # 所有误差都要除p步求平均
#     lose['recon_loss'] = lose['recon_loss'] / step
#     lose['inf_loss'] = lose['inf_loss'] / step
#     lose['predict_loss'] = lose['recon_loss'] / step
#     lose['linear_loss'] = lose['inf_loss'] / step
#
#
#
#     loss_value = lose['predict_loss'] + lose['linear_loss'] + lose['recon_loss'] * 0.3 + lose['inf_loss'] * 1e-9
#
#     return loss_value