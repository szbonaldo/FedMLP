import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
# ChestXray14
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
# 读取CSV文件
data = pd.read_csv(r'E:\ICH_stage2\ICH_stage2\data_png185k_512.csv')

# 按比例划分数据集
train_ratio = 0.7
test_ratio = 0.3

# 划分训练集和剩余数据
train_data, test_data = train_test_split(data, test_size=(1 - train_ratio))


# 输出数据集的大小
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

# 可以保存这些数据集到不同的CSV文件中
train_data.to_csv('train_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)
