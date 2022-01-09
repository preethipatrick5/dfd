from torchvision import transforms

from dfd import Datamodule, Resnet, Xception, Resnet3D, ResnetLSTM
from dfd.utils.transforms import AuglyTransform, AddGaussianNoise, RandomApply, ConvertToRGB
import image as imaugs


def get_baseline():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_random_noise():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            imaugs.RandomNoise(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_padding():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([150, 150]),
            imaugs.pad,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_perspective_transform():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            imaugs.perspective_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_shuffle_pixels(factor=0.8):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            AuglyTransform(imaugs.shuffle_pixels, factor=factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_saturation(factor=0.8):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            AuglyTransform(imaugs.saturation, factor=factor),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_pixelization(ratio=0.8):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([150, 150]),
            AuglyTransform(imaugs.pixelization, ratio=ratio),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_text_overlay():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            AuglyTransform(imaugs.overlay_text),
            ConvertToRGB(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_gaussian_noise():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            transforms.ToTensor(),
            AddGaussianNoise(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_gaussian_blur(kernal_size=15):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            transforms.GaussianBlur(kernel_size=kernal_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_random_augs():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([250, 250]),
            RandomApply([
                imaugs.saturation,
                imaugs.saturation,
                imaugs.pixelization,
                imaugs.perspective_transform,
                # imaugs.pad,
                imaugs.RandomNoise()
            ], max_num=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )


def get_all_experiments():
    base_experiments = {
        "baseline": get_baseline(),
        "saturation": get_saturation(),
        "pixelization": get_pixelization(),
        "gaussian_blur": get_gaussian_blur(),
        "gaussian_noise": get_gaussian_noise(),
        "random_noise": get_random_noise(),
        "padding": get_padding(),
        "perspective_transform": get_perspective_transform(),
        "random_augmentation": get_random_augs(),
        "shuffle_pixels": get_shuffle_pixels(),
        #         "text_overlay": get_text_overlay()
    }

    gaussian_blur_experiments = {
        "get_gaussian_blur_kernal_15": get_gaussian_blur(5),
        "get_gaussian_blur_kernal_50": get_gaussian_blur(25),
        "get_gaussian_blur_kernal_100": get_gaussian_blur(75),
        "get_gaussian_blur_kernal_200": get_gaussian_blur(125),
    }

    pixelization_experiments = {
        "pixelization_0.3": get_pixelization(ratio=0.3),
        "pixelization_0.5": get_pixelization(ratio=0.5),
        "pixelization_0.7": get_pixelization(ratio=0.7),
        "pixelization_0.9": get_pixelization(ratio=0.9),
        "pixelization_0.1": get_pixelization(ratio=1.0),
    }

    saturation_experiments = {
        "saturation_0.3": get_saturation(factor=0.3),
        "saturation_0.5": get_saturation(factor=0.5),
        "saturation_0.7": get_saturation(factor=0.7),
        "saturation_0.9": get_saturation(factor=0.9),
        "saturation_0.1": get_saturation(factor=1.0),
    }

    shuffle_pixels_experiments = {
        "shuffle_pixels_0.3": get_shuffle_pixels(factor=0.3),
        "shuffle_pixels_0.5": get_shuffle_pixels(factor=0.5),
        "shuffle_pixels_0.7": get_shuffle_pixels(factor=0.7),
        "shuffle_pixels_0.9": get_shuffle_pixels(factor=0.9),
        "shuffle_pixels_0.1": get_shuffle_pixels(factor=1.0),
    }

    experiments = [base_experiments, gaussian_blur_experiments, pixelization_experiments, saturation_experiments,
                   shuffle_pixels_experiments]
    return experiments


def get_models_and_loaders(root, transforms, train_data, test_data, val_data):
    models = [Resnet3D(), ResnetLSTM(), Resnet(), Xception()]
    loaders = [
        Datamodule(
            root=root, transforms=transforms, train_data=train_data, test_data=test_data, val_data=val_data,
            data_type=2,
            batch_size=128, num_workers=1, shuffle=True, clip_duration=2, max_frames=120, return_path=True
        ),
        Datamodule(
            root=root, transforms=transforms, train_data=train_data, test_data=test_data, val_data=val_data,
            data_type=2,
            batch_size=128, num_workers=1, shuffle=True, clip_duration=2, max_frames=120, return_path=True
        ),
        Datamodule(
            root=root, transforms=transforms, train_data=train_data, test_data=test_data, val_data=val_data,
            data_type=1,
            batch_size=128, num_workers=1, shuffle=True, clip_duration=2, max_frames=120, return_path=True
        ),
        Datamodule(
            root=root, transforms=transforms, train_data=train_data, test_data=test_data, val_data=val_data,
            data_type=1,
            batch_size=128, num_workers=1, shuffle=True, clip_duration=2, max_frames=120, return_path=True
        ),
    ]
    return models, loaders
