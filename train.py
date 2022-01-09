import argparse
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

import dfd


def get_transforms(train_type):
    if train_type == "clean":
        transforms = dfd.utils.transforms.basic_cnn_transforms()
    elif train_type == "noise":
        transforms = dfd.utils.transforms.noise_cnn_transforms()
    else:
        raise Exception("Invalid train_type")
    return transforms


def get_trainable(model_type, for_data, root, train_type, return_path=False, transforms=None, from_checkpoint=False):
    transforms = transforms if transforms else get_transforms(train_type)
    if model_type == "r":
        model, data_module = dfd.get_resnet_and_data_module(
            for_data=for_data, root=root, transforms=transforms, return_path=return_path,
            from_checkpoint=from_checkpoint, train_type=train_type
        )
    elif model_type == "x":
        model, data_module = dfd.get_xception_and_data_module(
            for_data=for_data, root=root, transforms=transforms, return_path=return_path,
            from_checkpoint=from_checkpoint, train_type=train_type
        )
    elif model_type == "rl":
        model, data_module = dfd.get_resnet_lstm_and_data_module(
            for_data=for_data, root=root, transforms=transforms, return_path=return_path,
            from_checkpoint=from_checkpoint, train_type=train_type
        )
    elif model_type == "3r":
        model, data_module = dfd.get_3dresnet_and_data_module(
            for_data=for_data, root=root, transforms=transforms, return_path=return_path,
            from_checkpoint=from_checkpoint, train_type=train_type
        )
    else:
        raise Exception("Invalid modeltype")
    return model, data_module


def get_gpu(gpu):
    if gpu:
        gpu = gpu.split(",")
        gpu = list(map(lambda x: int(x), gpu))
    return gpu


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--root", required=True)
    arg_parser.add_argument("--for_data", required=True)
    arg_parser.add_argument("--epochs", required=True)
    arg_parser.add_argument("--model", required=True)
    arg_parser.add_argument("--gpu", required=False, default=None)
    arg_parser.add_argument("--dev_run", required=False, default='0')
    arg_parser.add_argument("--train_type", required=False, default='clean')
    args = arg_parser.parse_args()
    model, data_module = get_trainable(args.model, args.for_data, args.root, args.train_type)
    gpus = get_gpu(args.gpu)
    dev_run = args.dev_run == '1'
    tb_logger = pl_loggers.TensorBoardLogger(f"lightning_logs/{args.for_data}/{args.train_type}/{args.model}")
    trainer = pl.Trainer(fast_dev_run=dev_run, gpus=gpus, max_epochs=int(args.epochs), logger=tb_logger)
    trainer.fit(model, datamodule=data_module)
