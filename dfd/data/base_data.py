import torch
from abc import ABC


class DFData(torch.utils.data.Dataset, ABC):
    def __init__(self, *, data_folder, meta_data, transforms, every_nth_frame=48, **kwargs):
        self.data_folder = data_folder
        self.meta_data = meta_data
        self.every_nth_frame = every_nth_frame
        self.transforms = transforms
