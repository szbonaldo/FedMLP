import csv
# ChestXray14
import numpy as np
import pandas as pd

csv_path = '../onehot-label.csv'
count = 1
pre_patient = ''
total_disease = [0]*14
new_disease = [0]*14
patients = 0
with open(csv_path, 'r') as csv_file:
    # 读取csv文件
    csv_reader = csv.reader(csv_file)
    # 遍历每一行数据
    for row in csv_reader:
        # 如果行满足特定条件，则将其写入新文件
        if count == 1:
            count += 1
            print(count)
        else:
            count += 1
            print(count)
            if row[0].split('_')[0] != pre_patient:
                patients += 1
                pre_patient = row[0].split('_')[0]
                total_disease = [i + j for i, j in zip(total_disease, new_disease)]
                new_disease = list(map(int, row[1:]))
            else:
                new_disease = [int(new_disease[i]) or int(row[i+1]) for i in range(len(new_disease))]
    total_disease = [i + j for i, j in zip(total_disease, new_disease)]
    print(np.array(total_disease)/patients)

