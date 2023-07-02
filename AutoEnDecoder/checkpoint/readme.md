# 训练结果汇总

| 网络模型          | 优化器 | epoch | batachsize | lr     | 硬件 | 训练耗时   | 训练误差 | 可视化 |
| ----------------- | ------ | ----- | ---------- | ------ | ---- | ---------- | -------- | ------ |
| AutoEncoder1.pkl  | Adam   | 20    | 100        | 0.001  | CPU  | 24471.87s  | 7.8416   | 1.svg  |
| AutoEncoder2.pkl  | Adamax | 20    | 100        | 0.0005 | GPU  | 5641.66s   | 7.9478   | 2.svg  |
| AutoEncoder3.pkl  | Adamax | 50    | 100        | 0.0008 | GPU  | 14149.10s  | 7.7437   | 3.svg  |
| AutoEncoder4.pkl  | Adamax | 20    | 50         | 0.001  | GPU  | 5605.99s   | 7.8019   | 4.svg  |
| AutoEncoder5.pkl  | Adamax | 20    | 50         | 0.002  | GPU  | 5553       | 8.5270   | 5.svg  |
| AutoEncoder6.pkl  | Adamax | 20    | 50         | 0.0008 | CPU  | 89599.19S  | 7.7508   | 6.svg  |
| AutoEncoder7.pkl  | Adamax | 30    | 50         | 0.0015 | GPU  | 8307       | 7.6808   | 7.svg  |
| AutoEncoder8.pkl  | Adamax | 30    | 25         | 0.0015 | CPU  | 71198.35s  | 7.5401   | 8.svg  |
| AutoEncoder9.pkl  | Adamax | 50    | 50         | 0.002  | GPU  | 8307.35s   | 7.3776   | 9.svg  |
| AutoEncoder10.pkl | Adamax | 100   | 25         | 0.0025 | GPU  | 28392.52zs | 7.3281   | 10.svg |
| AutoEncoder11.pkl | Adamax | 20    | 25         | 0.003  | GPU  | 5907.65zs  | 7.6552   | 11.svg |
| AutoEncoder11.pkl | Adamax | 20    | 15         | 0.003  | GPU  | 5844.65zs  | 7.5947   | 11.svg |
| AutoEncoder12.pkl | Adamax | 100   | 25         | 0.0025 | CPU  | 329101.28S | 7.375    | 12.svg |
|                   |        |       |            |        |      |            |          |        |
|                   |        |       |            |        |      |            |          |        |
|                   |        |       |            |        |      |            |          |        |
|                   |        |       |            |        |      |            |          |        |
|                   |        |       |            |        |      |            |          |        |
|                   |        |       |            |        |      |            |          |        |

/excel目录里面是matrixA和matrixB以矩阵形式写入

/mat是编码器解码器和矩阵分别以mat形式存储的结果，分别是encoder.mat,decoder.mat和matrix.mat

/svg下是网络训练时候误差可视化的结果
