import torch
from PIL import Image
from torchvision.transforms import transforms
from utils.image import default_interpolation_mode
from .random_erase import RandomErasing

__all__ = ['get_train_transform', 'get_test_transform', 'get_low_level_transform']


# ---------------------------
# Data Transform
# ---------------------------
def get_train_transform(config):
    transform_list = [transforms.Resize(size=(256, 128), interpolation=default_interpolation_mode)]
    if config['use_crop']:
        transform_list += [
            transforms.Pad(10),
            transforms.RandomCrop((256, 128))
        ]
    if config['use_flip']:
        transform_list += [transforms.RandomHorizontalFlip()]
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if config['use_erase']:
        transform_list += [RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])]
    return transforms.Compose(transform_list)


def get_test_transform():
    transform_list = [
        transforms.Resize(size=(256, 128), interpolation=default_interpolation_mode),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


# -----------------------------------------
# Data Transform for Image Enhancement
# -----------------------------------------
def get_low_level_transform(input_size, input_mode=None):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    if input_mode == 'crop':
        print('[WARNING] Please check if all inputs get the same cropped window!')
        transform_list = [transforms.RandomCrop(size=input_size)]
    elif input_mode == 'resize':
        transform_list = [transforms.Resize(size=input_size, interpolation=default_interpolation_mode)]
    elif input_mode == 'resize_and_crop':
        from .random_crop import ResizeAndCrop
        transform_list = [ResizeAndCrop(size=input_size, interpolation=default_interpolation_mode)]
    elif input_mode == 'none' or input_mode is None:
        transform_list = []
    else:
        assert 0, 'Invalid transform input mode: {}'.format(input_mode)
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':
    pass
