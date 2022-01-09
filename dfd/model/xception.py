from dfd.model.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import cv2
import pretrainedmodels


class Xception(BaseModel):

    def __init__(self):
        super(Xception, self).__init__()
        self.model = pretrainedmodels.__dict__["xception"](num_classes=1000, pretrained='imagenet')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.last_linear = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        result = self.model(x)
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
