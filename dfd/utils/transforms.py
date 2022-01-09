import torch
from PIL import Image
from torchvision import transforms

import image as imaugs


class AuglyTransform:
    def __init__(self, transform, *args, **kwargs):
        self.transform = transform
        self.args = args
        self.kwargs = kwargs

    def __call__(self, image):
        return self.transform(image=image, **self.kwargs)


class ConvertToRGB:
    def __call__(self, image, *args, **kwargs):
        test = Image.new("RGB", image.size, (255, 255, 255))
        test.paste(image, mask=image.split()[3])
        return test


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomApply(torch.nn.Module):
    def __init__(self, transform_list, max_num=4):
        super().__init__()
        self.transform_list = transform_list
        self.max_num = min(max_num, len(transform_list))

    def forward(self, img):
        perm = torch.randperm(len(self.transform_list))
        k = torch.randint(1, self.max_num + 1, [1])
        idx = perm[:k]
        for ind in idx:
            img = self.transform_list[ind](img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transform_list:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


def basic_cnn_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def noise_cnn_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            RandomApply([
                imaugs.saturation,
                imaugs.saturation,
                imaugs.pixelization,
                imaugs.perspective_transform,
                imaugs.RandomNoise()
            ], max_num=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
