from concurrent.futures import ThreadPoolExecutor

import cv2
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import functional as FM
import numpy as np

import torchmetrics
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from torch.nn.functional import binary_cross_entropy

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad
}


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.loss_function = binary_cross_entropy
        self.transforms = kwargs.get("transforms")
        self.train_losses = []
        self.train_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.mean_calculator = lambda x: sum(x) / len(x)
        self.executor = ThreadPoolExecutor(max_workers=kwargs.get("max_workers", 10))
        self.target_layer = None
        self.name = None

    def _init(self):
        self.name = self.__class__.__name__

    def set_target_layer(self, target_layer):
        self.target_layer = target_layer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._training_step(batch, batch_idx)
        y = y.type(torch.float32)
        loss = self.loss_function(y_hat, y)
        y_pred = torch.round(y_hat.clone())
        y_pred = y_pred.type(torch.int32)
        y = y.type(torch.int32)
        acc = torchmetrics.functional.accuracy(y_pred, y)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}

    def _training_step(self, batch, batch_idx):
        raise Exception("_training_step not implemented")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._test_step(batch, batch_idx)
        y = y.type(torch.float32)
        loss = self.loss_function(y_hat, y)
        y_hat = torch.round(y_hat)
        y_hat = y_hat.type(torch.int32)
        y = y.type(torch.int32)
        acc = FM.accuracy(y_hat, y)
        self.test_losses.append(loss)
        self.test_accuracies.append(acc)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}

    def _test_step(self, batch, batch_idx):
        raise Exception("_test_step not implemented")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._validation_step(batch, batch_idx)
        y = y.type(torch.float32)
        loss = self.loss_function(y_hat, y)
        y_hat = torch.round(y_hat)
        y_hat = y_hat.type(torch.int32)
        y = y.type(torch.int32)
        acc = FM.accuracy(y_hat, y)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}

    def _validation_step(self, batch, batch_idx):
        raise Exception("_validation_step not implemented")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
        return [optimizer], [scheduler]

    def grad_cam(self, image_path, method_name, use_cuda):
        cam = methods[method_name](model=self,
                                   target_layers=[self.target_layer],
                                   use_cuda=use_cuda, )
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        target_category = None
        cam.batch_size = 2

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            eigen_smooth=False,
                            aug_smooth=False)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        cv2.imwrite(f'{self.name}_{method_name}_cam.jpg', cam_image)
