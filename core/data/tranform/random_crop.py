import PIL.Image
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as functional
from .transform import default_interpolation_mode


def get_img_size(img):
    if isinstance(img, Image.Image):
        width, height = img.size
    else:
        assert img.ndim >= 2
        width, height = img.shape[-1], img.shape[-2]
    return width, height


class SyncRandomCrop(transforms.RandomCrop):

    def __init__(self, size, resize_if_needed=True, interpolation=default_interpolation_mode, **kwargs):
        if isinstance(size, int):
            size = (size, size)
        super().__init__(size, **kwargs)
        self.interpolation = interpolation
        self.resize_if_needed = resize_if_needed

    def _pad(self, img):
        if self.padding is not None:
            img = functional.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = get_img_size(img)

        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = functional.pad(img, padding, self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = functional.pad(img, padding, self.fill, self.padding_mode)

        return img

    def forward(self, *img_list):

        cropped_list, crop_params = [], None

        if isinstance(img_list, torch.Tensor) or isinstance(img_list, Image.Image):
            img_list = [img_list]

        for img in img_list:
            if self.resize_if_needed:
                width, height = get_img_size(img)
                if height < self.size[0] or width < self.size[1]:
                    img = functional.resize(img, max(self.size), interpolation=self.interpolation)
            else:
                img = self._pad(img)

            if crop_params is None:
                crop_params = self.get_params(img, self.size)

            cropped_list.append(functional.crop(img, *crop_params))

        if len(cropped_list) == 1:
            return cropped_list[0]

        return cropped_list


class ResizeAndCrop(torch.nn.Module):
    def __init__(self, size: tuple, interpolation=default_interpolation_mode):
        super().__init__()
        self.target_size = size
        self.interpolation = interpolation
        self.min_size = min(size)
        self.rand_crop = transforms.RandomCrop(size)

    def forward(self, img):
        # width, height = img.size
        if img.size[0] < self.target_size[1] or img.size[1] < self.target_size[0]:
            img = functional.resize(img, max(self.target_size), interpolation=self.interpolation)
        img_crop = self.rand_crop(img)
        return img_crop
