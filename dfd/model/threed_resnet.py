import numpy
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from dfd.model.base_model import BaseModel


class Resnet3D(BaseModel):

    def __init__(self, freeze_pretrained=True):
        super(Resnet3D, self).__init__()
        self.model = torchvision.models.video.r3d_18(pretrained=True)
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1)
        )
        self.loss_function = F.binary_cross_entropy

    def forward(self, x):
        device = x.device
        x = x.cpu().detach().numpy()
        x = numpy.transpose(x, (0, 2, 1, 3, 4))
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
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

