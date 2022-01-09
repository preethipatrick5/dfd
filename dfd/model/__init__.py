from dfd.model.resnet import Resnet


class ModelType:
    RESNET = 1
    types = [RESNET]


def get_model(type: int):
    return Resnet()
