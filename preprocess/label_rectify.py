import csv
# ChestXray14
import numpy as np
import pandas as pd

# df = pd.read_csv('Data_Entry_2017_v2020.csv')
# print(df)
csv_path = './Data_Entry_2017_v2020.csv'
count = 1
title = ['Image Index', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
with open(csv_path, 'r') as csv_file:
    # 读取csv文件
    csv_reader = csv.reader(csv_file)

    # 创建一个新的csv文件，用于存储筛选后的结果
    with open('../onehot-label-PA.csv', 'w', newline='') as new_csv_file:
        csv_writer = csv.writer(new_csv_file)
        # 遍历每一行数据
        for row in csv_reader:
            # 如果行满足特定条件，则将其写入新文件
            if count == 1:
                csv_writer.writerow(title)
                count += 1
                print(count)
            else:
                count += 1
                print(count)
                if row[6] == 'PA':
                    label_row = [row[0]] + [0]*14
                    label = row[1]  # str
                    if label == 'No Finding':
                        csv_writer.writerow(label_row)
                    else:
                        label_list = label.split('|')
                        for i in label_list:
                            label_row[title.index(i)] = 1
                        csv_writer.writerow(label_row)
                else:
                    pass

