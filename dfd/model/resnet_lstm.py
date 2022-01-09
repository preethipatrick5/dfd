from typing import Any

import pretrainedmodels
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.metrics import functional as FM
from torch import nn

from dfd.model.base_model import BaseModel


# from detector_model.encoder import ResNet
# https://www.kaggle.com/c/deepfake-detection-challenge/discussion/134366
# from detector_model.sequence_classifier import LSTM


class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model.get("num_classes", 1)
        dr_rate = params_model.get("dr_rate", 0.1)
        pretrained = params_model.get("pretrained", True)
        rnn_hidden_size = params_model.get("rnn_hidden_size", 1)
        rnn_num_layers = params_model.get("rnn_num_layers", 100)

        baseModel = torchvision.models.resnet101(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Sequential(
            nn.Linear(rnn_hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:, ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:, ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = torchvision.models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(self.h_FC_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        # self.o = nn.Linear(in_f, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class XceptionCNNEncoder(nn.Module):
    def __init__(self, in_f, out_f):
        super(XceptionCNNEncoder, self).__init__()
        self.base = pretrainedmodels.__dict__["xception"](num_classes=1000, pretrained='imagenet')
        self.h1 = Head(in_f, out_f)

    def forward(self, x_3d):
        cnn_embed_seq = []

        for i in range(x_3d.size(1)):
            x = self.base(x_3d[:, i, :, :, :])
            x = self.h1(x)
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class XCNNEncoder(nn.Module):
    def __init__(self, in_f, out_f):
        super(XCNNEncoder, self).__init__()
        self.base = pretrainedmodels.__dict__["xception"](num_classes=1000, pretrained='imagenet')
        print(self.base)
        print("#" * 50)
        modules = list(self.base.children())[:-1]
        self.base = nn.Sequential(*modules)
        print(self.base)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc3 = nn.Linear(512, 300)

    def forward(self, x_3d):
        cnn_embed_seq = []

        for i in range(x_3d.size(1)):
            x = self.base(x_3d[:, i, :, :, :])
            x = self.pooling(x)
            x = x.view(x.size(0), -1)
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=0.4, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class RNNDecoder(nn.Module):
    def __init__(self, in_f, out_f):
        super(RNNDecoder, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=in_f,
            hidden_size=256,
            num_layers=3,
            batch_first=True
        )

        self.f1 = nn.Linear(256, 128)
        self.r = nn.ReLU()
        self.d = nn.Dropout(0.3)
        self.f2 = nn.Sequential(nn.Linear(128, out_f), nn.Sigmoid())

    def forward(self, x):
        self.LSTM.flatten_parameters()
        x, (hn, hc) = self.LSTM(x)
        x = self.d(self.r(self.f1(x[:, -1, :])))
        x = self.f2(x)
        return x


class XceptionLRCN(nn.Module):
    def __init__(self, cnn_features, out_f):
        super(XceptionLRCN, self).__init__()
        self.cnn = XceptionCNNEncoder(1000, cnn_features)
        self.rnn = RNNDecoder(cnn_features, out_f)

    def forward(self, x_3d):
        encoded = self.cnn(x_3d)
        predicted = self.rnn(encoded)
        return predicted


class LitXceptionLRCN(pl.LightningModule):

    def __init__(self, cnn_features, out_f):
        super(LitXceptionLRCN, self).__init__()
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        CNN_embed_dim = 512  # latent dim extracted by 2D CNN
        res_size = 224  # ResNet image size
        dropout_p = 0.0  # dropout probability
        # self.encoder = CNN(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
        #                       CNN_embed_dim=CNN_embed_dim)
        # self.sequence_classifier = LSTM(input_shape=512, hidden_size=512, num_layers=3)
        self.model = XceptionLRCN(cnn_features, out_f)
        self.loss_function = F.binary_cross_entropy

    def forward(self, x) -> Any:
        y_predicted = self.model(x)
        return y_predicted

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x
        y_predicted = self(x)
        y_predicted = y_predicted.view(-1, )
        loss = self.loss_function(y_predicted.view(-1, ), y.type(torch.float32))
        # Logging to TensorBoard by default
        # y_predicted_numpy = y_predicted.cpu().detach().numpy()
        # y_predicted_numpy[y_predicted_numpy < 0.5] = 0
        # y_predicted_numpy[y_predicted_numpy > 0.5] = 1
        # acc = FM.accuracy(y_predicted_numpy, y.cpu().detach().numpy())
        y_predicted_numpy = y_predicted.detach().clone()
        y_predicted_numpy[y_predicted_numpy < 0.5] = 0
        y_predicted_numpy[y_predicted_numpy >= 0.5] = 1
        acc = FM.accuracy(y_predicted_numpy.type(torch.int), y.type(torch.int))
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class ResnetLSTM(BaseModel):
    def __init__(self):
        super(ResnetLSTM, self).__init__()
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        CNN_embed_dim = 512
        self.encoder = ResCNNEncoder()
        self.sequence_classifier = DecoderRNN()

    def forward(self, x):
        y_intermediate = self.encoder(x)
        y_predicted = self.sequence_classifier(y_intermediate)
        y_predicted = torch.sigmoid(y_predicted)
        return y_predicted.view(-1)

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

# class ResnetLSTM(pl.LightningModule):
#
#     def __init__(self):
#         super(ResnetLSTM, self).__init__()
#         CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
#         CNN_embed_dim = 512  # latent dim extracted by 2D CNN
#         res_size = 224  # ResNet image size
#         dropout_p = 0.0  # dropout probability
#         # self.encoder = CNN(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
#         #                       CNN_embed_dim=CNN_embed_dim)
#         # self.sequence_classifier = LSTM(input_shape=512, hidden_size=512, num_layers=3)
#         self.encoder = ResCNNEncoder()
#         self.sequence_classifier = DecoderRNN()
#         self.loss_function = F.binary_cross_entropy
#
#     def forward(self, x) -> Any:
#         y_intermediate = self.encoder(x)
#         y_predicted = self.sequence_classifier(y_intermediate)
#         return y_predicted
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         x = x
#         y_predicted = self(x)
#         y_predicted = y_predicted.view(-1, )
#         loss = self.loss_function(y_predicted.view(-1, ), y.type(torch.float32))
#         # Logging to TensorBoard by default
#         # y_predicted_numpy = y_predicted.cpu().detach().numpy()
#         # y_predicted_numpy[y_predicted_numpy < 0.5] = 0
#         # y_predicted_numpy[y_predicted_numpy > 0.5] = 1
#         # acc = FM.accuracy(y_predicted_numpy, y.cpu().detach().numpy())
#         y_predicted_numpy = y_predicted.detach().clone()
#         y_predicted_numpy[y_predicted_numpy < 0.5] = 0
#         y_predicted_numpy[y_predicted_numpy >= 0.5] = 1
#         acc = FM.accuracy(y_predicted_numpy.type(torch.int), y.type(torch.int))
#         self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return {'loss': loss, 'acc': acc}
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
#         return [optimizer], [scheduler]
