import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self):
        """Define  the AutoEncoder.
            matrix_A 150 * 150
            matrix_B 150 * 1
            encoder 21 * 32 *64 *129
            decoder 150 * 128 * 64 *32 * 21
        """

        super(AutoEncoder, self).__init__()
        matrix_B = torch.randn(1,150, requires_grad=True)
        matrix_A = torch.randn(150, 150, requires_grad=True)
        self.matrix_B = nn.Parameter(matrix_B)
        self.matrix_A = nn.Parameter(matrix_A)

        self.encoder = nn.Sequential(
            nn.Linear(21, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 129),

        )
        self.decoder = nn.Sequential(
            nn.Linear(150, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 21),
            nn.Sigmoid()

        )

    def forward(self, x ,u, flag): # flag 用于区分是计算重构误差还是预测误差
        """Define the forward
            Arguments:
                self -- Encoder Decoer MatrixA MatrixB
                X -- state dict
                u -- input control
                flag divide recon or predict

            Attention:
                flag -- 自编码器计算两个不同步骤，True的时候经计算预测和线性误差，False时候计算重构误差

            Returns:
                flag is False -- decoder
                flag is True -- decoded, Matrix_Loss, self.matrix_A, self.matrix_B
        """
        U = u
        encoded = self.encoder(x) # 21维度向量 经过编码器变成129维
        glist = torch.cat((x, encoded), 0)  # tensor拼接 150维 glist
        if flag is False: # flag是False时候计算重构误差
            decoded = self.decoder(glist) # 不做拼接直接decoer分析
            return decoded

        elif flag is True:# flag is True 的时候计算预测误差和线性误差
            U_Convert = torch.mm(U.reshape(1,-1),self.matrix_B).reshape(150,-1) # tensor内积:U和matrixB
            flpha_Convert = torch.mm(glist.reshape(1,-1) , self.matrix_A).reshape(150,1) # tensor内积：glist和matrixA
            ylist = (flpha_Convert + U_Convert) # 150维向量相加:即A，B矩阵相乘后做相加
            Base_Value = (glist + U)# 因为要计算一下A，B矩阵的影响，该值为一个基准值
            Matrix_Loss = torch.sqrt(torch.mean(torch.pow((Base_Value - ylist.view(-1)), 2))) # 线性误差也是矩阵误差
            Encoder_Input = ylist.view(-1) # 压缩为1维向量，不然是高位数组
            decoded = self.decoder(Encoder_Input) ## 解码的过程
            return decoded, Matrix_Loss, self.matrix_A, self.matrix_B



