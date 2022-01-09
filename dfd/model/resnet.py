import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dfd.model.base_model import BaseModel


class Resnet(BaseModel):

    def __init__(self, model_depth=50, freeze_pretrained=True):
        super(Resnet, self).__init__()
        self.model = self._get_model(model_depth)(pretrained=True)
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        result = self.model(x)
        result = torch.sigmoid(result)
        return result.view(-1)

    def _training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def _test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    @staticmethod
    def _get_model(model_depth):
        model_depth_map = {
            101: torchvision.models.resnet101,
            50: torchvision.models.resnet50
        }
        return model_depth_map.get(model_depth)
