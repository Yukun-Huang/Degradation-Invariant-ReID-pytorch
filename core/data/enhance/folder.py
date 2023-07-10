import os
import os.path as osp
import random
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from ..tranform import get_low_level_transform, get_degrade_function
from ..tranform.random_crop import SyncRandomCrop
from ..utils import make_dataset


class PairedFolder(Dataset):
    def __init__(self, root, gt_root, train_or_test, config):
        self.imgs = make_dataset(root)
        self.gt_imgs = make_dataset(gt_root)
        self.img_num = len(self.imgs)
        self.gt_img_num = len(self.gt_imgs)
        self.root = root
        self.gt_root = gt_root
        if train_or_test in ['test', 'val']:
            self.rand_crop, input_mode = None, 'resize'
        else:
            self.rand_crop, input_mode = SyncRandomCrop(config['input_size']), 'none'
        self.transform = get_low_level_transform(config['input_size'], input_mode=input_mode)
        if config['degrade_type'] is not None:
            self.degrade_transform = get_degrade_function(config['degrade_type'])
        print('{}:  num_img={}, num_gt={}'.format(self.__class__.__name__, self.img_num, self.gt_img_num))

    def __getitem__(self, index):
        path = self.imgs[index]
        gt_path = self.gt_imgs[index]
        img, gt = default_loader(path), default_loader(gt_path)
        if self.rand_crop is not None:
            img, gt = self.rand_crop(img, gt)
        deg = self.degrade_transform(gt) if self.degrade_transform else None
        img, gt = self.transform(img), self.transform(gt)
        if deg:
            return gt, self.transform(deg), img, 1, 0
        else:
            return gt, img

    def __len__(self):
        return len(self.imgs)


class UnPairedFolder(PairedFolder):
    def __init__(self, root, gt_root, train_or_test, config):
        super(UnPairedFolder, self).__init__(root, gt_root, train_or_test, config)
        if train_or_test != 'test':
            self.reset()

    def reset(self):
        random.shuffle(self.imgs)
        random.shuffle(self.gt_imgs)


class TestFolder(Dataset):
    def __init__(self, root, train_or_test, config):
        self.imgs = make_dataset(root)
        self.img_num = len(self.imgs)
        self.root = root
        self.train_or_test = train_or_test
        self.transform = get_low_level_transform(config['input_size_test'], input_mode='resize')
        print('{}:  num_img={}'.format(self.__class__.__name__, self.img_num))

    def __getitem__(self, index):
        path = self.imgs[index]
        img = default_loader(path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
