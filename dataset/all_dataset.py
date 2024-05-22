import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset


class ChestXray14(Dataset):
    def __init__(self, datapath, mode, transform=None):
        self.datapath = datapath
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join("/home/szb/multilabel/", self.mode + "_dataset_8class.csv")
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["Image Index"].values
        self.targets = self.file.iloc[0:, 1:].values.astype(np.float32)

    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)

        if self.transform is not None:
            if isinstance(self.transform, tuple):
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)
                return {"image_aug_1": image1,
                        "image_aug_2": image2,
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                image = self.transform(image)
                return {"image": image,
                        "target": target,
                        "index": index,
                        "image_id": image_id}

    def __len__(self):
        return len(self.targets)

    def read_image(self, image_id):
        image_path = os.path.join("/home/szb/ChestXray14/images/image/", image_id)
        image = Image.open(image_path).convert("RGB")
        return image


class ICH(Dataset):
    def __init__(self, datapath, mode, transform=None):
        self.datapath = datapath
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join("/home/szb/ICH_stage2/ICH_stage2/", self.mode + "_dataset_ICH.csv")
        # csv_file = os.path.join("/home/szb/ICH_stage2/ICH_stage2/", self.mode + '_demo.csv')  # demo exp(5000 samples)
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["Image Index"].values
        self.targets = self.file.iloc[0:, 1:].values.astype(np.float32)

    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)
        if self.transform is not None:
            if isinstance(self.transform, tuple):
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)
                return {"image_aug_1": image1,
                        "image_aug_2": image2,
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                image = self.transform(image)
                return {"image": image,
                        "target": target,
                        "index": index,
                        "image_id": image_id}

    def __len__(self):
        return len(self.targets)

    def read_image(self, image_id):
        image_path = os.path.join("/home/szb/ICH_stage2/ICH_stage2/png185k_512/", image_id)
        image = Image.open(image_path).convert("RGB")
        return image
