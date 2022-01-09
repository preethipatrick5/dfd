import itertools
import os
import random

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from torch.utils.data import DistributedSampler


class Datamodule(pl.LightningDataModule):

    def __init__(self, root, transforms, train_data, test_data, val_data, data_type=1, batch_size=128, num_workers=10,
                 shuffle=True, clip_duration=2, max_frames=120, return_path=False):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.clip_duration = clip_duration
        self.max_frames = max_frames
        self.train_data = self._make(train_data) if data_type == 1 else train_data
        self.test_data = self._make(test_data) if data_type == 1 else train_data
        self.val_data = self._make(val_data) if data_type == 1 else train_data
        self.return_path = return_path

    def _make(self, data):
        images = []
        if type(data) is list:
            images = data
        else:
            print("Loading image data")
            for index, row in tqdm.tqdm(data.iterrows(), total=len(data)):
                images += self._get(row)
        random.shuffle(images)
        return images

    def _get(self, row):
        image_folder = row['relative_path'].split(".")[0]
        full_image_folder = os.path.join(self.root, image_folder)
        images = os.listdir(full_image_folder)
        label = self._get_label(row)
        data = [(os.path.join(image_folder, image), label) for image in images]
        return data

    @staticmethod
    def _get_label(row):
        label = 1 if row['label'] == 'real' else 0
        return label

    def train_dataloader(self):
        if self.data_type == 1:
            dataset = Image(self.train_data, self.transforms, self.root)
        elif self.data_type == 2:
            dataset = Video(data=self.train_data, transforms=self.transforms, root=self.root,
                            max_frames=self.max_frames)
        else:
            raise Exception("Invalid data_type")
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                  num_workers=self.num_workers)
        return data_loader

    def val_dataloader(self):
        if self.data_type == 1:
            dataset = Image(self.val_data, self.transforms, self.root, return_path=self.return_path)
        elif self.data_type == 2:
            dataset = Video(self.val_data, self.transforms, self.root, return_path=self.return_path)
        else:
            raise Exception("Invalid data_type")
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                  num_workers=self.num_workers)
        return data_loader

    def test_dataloader(self):
        if self.data_type == 1:
            dataset = Image(self.test_data, self.transforms, self.root)
        elif self.data_type == 2:
            dataset = Video(self.test_data, self.transforms, self.root)
        else:
            raise Exception("Invalid data_type")
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                  num_workers=self.num_workers)
        return data_loader


class Image(torch.utils.data.Dataset):
    def __init__(self, data, transforms, root, return_path=False):
        self.data = data
        self.transforms = transforms
        self.root = root
        self.return_path = return_path

    def __getitem__(self, index):
        x, y = self.data[index]
        image_folder = os.path.join(*x.split("/")[:-1])
        x = os.path.join(self.root, x)
        x = cv2.imread(x)
        if x is None:
            print(self.data[index])
        x = self.transforms(x)
        if self.return_path:
            return x, y, image_folder
        return x, y

    def __len__(self):
        return len(self.data)


class Video(torch.utils.data.Dataset):
    def __init__(self, data, transforms, root, max_frames=20, return_path=False):
        self.data = data
        self.transforms = transforms
        self.root = root
        self.max_frames = max_frames
        self.return_path = return_path

    def __getitem__(self, index):
        item = self.data.iloc[index, :]
        video_folder_name = item['relative_path'].split(".")[0]
        label = item["label"]
        frames = self.load_frames(video_folder_name)
        frames = list(map(self.transforms, frames))
        frames = torch.stack(frames, dim=0)
        if self.return_path:
            return frames, label, video_folder_name
        return frames, label

    def __len__(self):
        return len(self.data)

    def load_frames(self, video_folder_name):
        def load_image(image_path):
            path = os.path.join(self.root, video_folder_name, image_path)
            return cv2.imread(path)

        path = os.path.join(self.root, video_folder_name)
        frames = sorted(os.listdir(path), key=lambda x: int(x.split(".")[0]))[:self.max_frames]
        frames = list(map(load_image, frames))
        return frames
