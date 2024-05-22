import csv
from collections import Counter

import pandas as pd
import numpy as np

import os
from tqdm import tqdm
data = pd.read_csv(r"E:\ICH_stage2\ICH_stage2\stage_2_train.csv")
data = np.array(data)
patient_num = len(data) // 6
ID_all = []
label_all = []
label_add = [0]*5
for i in range(patient_num):
    ID = data[6*i][0].split('_epidural')[0]
    label_add = [data[6*i][1], data[6*i+1][1], data[6*i+2][1], data[6*i+3][1], data[6*i+4][1]]
    ID_all.append(ID)
    label_all.append(label_add)

ID_have = []
label_have = []
for i in tqdm(range(patient_num)):
    name = ID_all[i] + ".png"
    path = os.path.join(r"E:\ICH_stage2\ICH_stage2\png185k_512", name)
    if os.path.exists(path):
        ID_have.append(name)
        label_have.append(label_all[i])


csv_path = r'E:\ICH_stage2\ICH_stage2\data_png185k_512.csv'
count = 1
title = ['Image Index', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
with open(csv_path, 'w', newline='') as new_csv_file:
    csv_writer = csv.writer(new_csv_file)
    # 遍历每一行数据
    for i in tqdm(range(len(ID_have)+1)):
        # 如果行满足特定条件，则将其写入新文件
        if count == 1:
            csv_writer.writerow(title)
            count += 1
        else:
            count += 1
            csv_writer.writerow([ID_have[i-1]] + label_have[i-1])
label_have_total = np.sum(label_have, axis=0)
label_have_class = np.sum(label_have, axis=1)
print(label_have_total)  # [2761 32564 23766 32122 42496]
print(Counter(label_have_class))    #  Counter({0: 87948, 1: 67969, 2: 22587, 3: 5642, 4: 885, 5: 20})