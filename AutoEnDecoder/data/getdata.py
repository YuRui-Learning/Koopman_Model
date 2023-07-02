import pandas as pd
import numpy as np
import random
import os


def Get_data(koopman = 'Koopman_20000.csv'):
    """Define the data dict.
        Arguments:
            koopman -- file name

        Returns:
            data_dict 21个数据为一组，一个list 有11组数据，0-21，2-23，...., 20-41,共20000组书记
    """
    koopman = os.path.join('data', koopman)
    init_lenght = 21 # 初始长度为21
    df = pd.read_csv(koopman,header=None).T
    data = np.array(df)
    data_dict =dict()
    for i in range(data.shape[1]): # 20000维 20000维不同数据
        data_dict[i] = []
        if abs(data.T[i].mean()) > 1e2: # 筛选一些异常数据
            print(i)
        for j in range((data.shape[0] - init_lenght)//2 + 1): # 原先就有一维数据，所以要多读一次
            data_dict[i].append(data.T[i][ j * 2 : init_lenght + j * 2])
    print()
    return data_dict

if __name__ == "__main__":
    Get_data(koopman = 'Koopman_20000.csv')