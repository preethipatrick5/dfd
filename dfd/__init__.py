import os
import pickle

from dfd.model.resnet import Resnet
from dfd.model.xception import Xception
from dfd.model.resnet_lstm import ResnetLSTM
from dfd.model.threed_resnet import Resnet3D
from dfd.data.dfdc import Datamodule
from dfd.utils import config
import pandas as pd


def _get_train_test_val(train_file, test_file, val_file):
    if train_file.split(".")[-1] == "csv":
        train, test, val = pd.read_csv(train_file), pd.read_csv(test_file), pd.read_csv(val_file)
    elif train_file.split(".")[-1] in ["data", "pickle"]:
        train, test, val = None, None, None
        with open(train_file, "rb") as file:
            train = pickle.load(file)
        with open(test_file, "rb") as file:
            test = pickle.load(file)
        with open(val_file, "rb") as file:
            val = pickle.load(file)
    else:
        raise Exception("only csv and pickle allowed")
    return train, test, val


def _get_train_test_val_path(for_data, root):
    if for_data == "dfdc":
        data_config = config.get_dfdc_metadata_config(root)
    elif for_data == "ff":
        data_config = config.get_ff_metadata_config(root)
    else:
        raise Exception("for_data can only be one [dfdc, ff]")
    return data_config


def get_latest(folder, version=None):
    if version is None:
        versions = os.listdir(folder)
        if versions:
            if len(versions) > 1:
                version = sorted(versions)
                version = version[-1]
            else:
                version = versions[0]
            checkpoints = os.listdir(os.path.join(folder, version))
            if checkpoints:
                return os.path.join(folder, version, checkpoints[0])
        else:
            raise Exception("No version found")
    else:
        if not os.path.exists(os.path.join(folder, version)):
            raise Exception("No version found")
        checkpoints = os.listdir(os.path.join(folder, version))
        if checkpoints:
            return os.path.join(folder, version, checkpoints[0])
        else:
            raise Exception("No checkpoints found")


def get_resnet_and_data_module(for_data, root, transforms, batch_size=256, train_type="clean",
                               num_workers=10, shuffle=True, return_path=False, from_checkpoint=False, version=None):
    if from_checkpoint:
        checkpoint_folder = f"lightning_logs/{for_data}/{train_type}/r"
        latest_version = get_latest(checkpoint_folder, version)
        model = Resnet.load_from_checkpoint(checkpoint_path=latest_version)
    else:
        model = Resnet()
    data_config = _get_train_test_val_path(for_data, root)
    train, test, val = _get_train_test_val(**data_config)
    data_loader = Datamodule(
        root=root, transforms=transforms,
        train_data=train, test_data=test, val_data=val,
        data_type=1, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle, return_path=return_path
    )
    return model, data_loader


def get_xception_and_data_module(for_data, root, transforms, batch_size=256, train_type="clean",
                                 num_workers=10, shuffle=True, return_path=False, from_checkpoint=False, version=None):
    if from_checkpoint:
        checkpoint_folder = f"lightning_logs/{for_data}/{train_type}/x"
        latest_version = get_latest(checkpoint_folder, version)
        model = Xception.load_from_checkpoint(checkpoint_path=latest_version)
    else:
        model = Xception()
    data_config = _get_train_test_val_path(for_data, root)
    train, test, val = _get_train_test_val(**data_config)
    data_loader = Datamodule(
        root=root, transforms=transforms,
        train_data=train, test_data=test, val_data=val,
        data_type=1, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle, return_path=return_path
    )
    return model, data_loader


def get_resnet_lstm_and_data_module(for_data, root, transforms, batch_size=2, train_type="clean",
                                    num_workers=1, shuffle=True, max_frames=120, return_path=False,
                                    from_checkpoint=False, version=None):
    if from_checkpoint:
        checkpoint_folder = f"lightning_logs/{for_data}/{train_type}/rl"
        latest_version = get_latest(checkpoint_folder, version)
        model = ResnetLSTM.load_from_checkpoint(checkpoint_path=latest_version)
    else:
        model = ResnetLSTM()
    data_config = _get_train_test_val_path("ff", root)
    train, test, val = _get_train_test_val(**data_config)
    train = train[train['frame_count'] >= max_frames]
    test = test[test['frame_count'] >= max_frames]
    val = val[val['frame_count'] >= max_frames]
    data_loader = Datamodule(
        root=root, transforms=transforms,
        train_data=train, test_data=test, val_data=val,
        data_type=2, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle, max_frames=max_frames, return_path=return_path
    )
    return model, data_loader


def get_3dresnet_and_data_module(for_data, root, transforms, batch_size=3, train_type="clean",
                                 num_workers=1, shuffle=True, max_frames=60, clip_duration=2, return_path=False,
                                 from_checkpoint=False, version=None):
    if from_checkpoint:
        checkpoint_folder = f"lightning_logs/{for_data}/{train_type}/r3"
        latest_version = get_latest(checkpoint_folder, version)
        model = Resnet3D.load_from_checkpoint(checkpoint_path=latest_version)
    else:
        model = Resnet3D()
    data_config = _get_train_test_val_path("ff", root)
    train, test, val = _get_train_test_val(**data_config)
    train = train[train['frame_count'] >= max_frames]
    test = test[test['frame_count'] >= max_frames]
    val = val[val['frame_count'] >= max_frames]
    data_loader = Datamodule(
        root=root, transforms=transforms,
        train_data=train, test_data=test, val_data=val,
        data_type=2, batch_size=batch_size,
        num_workers=num_workers, shuffle=shuffle, max_frames=max_frames, clip_duration=clip_duration,
        return_path=return_path
    )
    return model, data_loader
