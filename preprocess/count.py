import csv
# ChestXray14
import pandas as pd
import os


def file_name(file_dir):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        file_list.append(files)
    return file_list

image_4_path = r'E:\szb\Hust\2022~2023\lab\博一\images_004.tar\images_004\images'
image_5_path = 'E:/szb/Hust/2022~2023/lab/博一/images_005.tar/images_005/images'
file_5_list = file_name(image_5_path)[0]
file_4_list = file_name(image_4_path)[0]
csv_path = '../Data_Entry_2017_v2020.csv'

with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    with open('filtered_data_4.csv', 'w', newline='') as new_csv_file:
        csv_writer = csv.writer(new_csv_file)
        for row in csv_reader:
            if row[0] in file_4_list:
                csv_writer.writerow(row)
