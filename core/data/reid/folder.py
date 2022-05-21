# -*- coding: utf-8 -*-
import numpy as np
from random import randint, choice
from functools import partial
from torchvision.datasets import ImageFolder
from .cam_parser import get_cam_and_label
from ..tranform import get_train_transform, get_test_transform, get_degrade_function, get_low_level_transform
from ..utils import make_dataset


__all__ = 'get_data_folder'


# ---------------------------
# Basic Folder
# ---------------------------
class BasicFolder(ImageFolder):
    def __init__(self, root):
        super(BasicFolder, self).__init__(root)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)
        print('[{}] {}, {}'.format(self.__class__.__name__, root, self.img_num))

    def __getitem__(self, index):
        raise NotImplementedError

    def __get_pos_sample(self, target, index, path):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        if len(pos_index) == 0:  # in the query set, only one sample
            return path
        else:
            rand = randint(0, len(pos_index)-1)
        return self.samples[pos_index[rand]][0]

    def __get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = randint(0, len(neg_index)-1)
        return self.samples[neg_index[rand]]


# ---------------------------
# Stage I
# 1.退化类型：ill: gamma, res: down-sample, mix:
# 2.返回格式：
#   2a.无监督：Img1_U + Img1_U_deg + Img2_U, 其中Img1_U和Img2_U的图像质量未知，分别来自不同摄像头
#   2b.半监督：Img1_H + Img1_H_deg + Img2_L, 其中Img1_H为高质量图像，Img2_L为低质量图像
# ---------------------------
class SemiTripletFolder(BasicFolder):
    def __init__(self, root, train_or_test, config):
        super(SemiTripletFolder, self).__init__(root)
        self.train_or_test = train_or_test
        self.lr_samples, self.hr_samples, self.deg_labels = [], [], []
        for path, pid in self.samples:
            if 'c0' in path.split('/')[-1]:
                self.lr_samples.append((path, pid))
                self.deg_labels.append(0)
            else:
                self.hr_samples.append((path, pid))
                self.deg_labels.append(1)
        self.transform = get_test_transform()
        deg_func = get_degrade_function(config['degrade_type'])
        if config['degrade_type'] == 'res':
            ratio_range = eval(config['sync_ds_ratio'])
            if ratio_range[0] == ratio_range[1]:
                ratio_range = (ratio_range[0], ratio_range[1] + 1e-6)
            deg_func = partial(deg_func, ratio_range=ratio_range)
            print('down-sampling ratio range: {}'.format(ratio_range))
        self.degrade_transform = deg_func

    def __getitem__(self, index):
        path, pid = self.samples[index]
        if not self.deg_labels[index]:
            lr_path, lr_label = path, pid
            hr_path, hr_label = choice(self.hr_samples)
        else:
            lr_path, lr_label = choice(self.lr_samples)
            hr_path, hr_label = path, pid
        hr_img, lr_img = self.loader(hr_path), self.loader(lr_path)
        deg_img = self.degrade_transform(hr_img)
        if self.transform is not None:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
            deg_img = self.transform(deg_img)
        return hr_img, deg_img, lr_img, hr_label, lr_label


class UnTripletFolder(BasicFolder):
    def get_sample_with_different_index(self, index):
        while 1:
            new_index = randint(0, self.img_num-1)
            if new_index != index:
                break
        return self.samples[new_index]

    def get_sample_with_different_camera(self, cam_id):
        index = np.argwhere(self.cameras != cam_id).flatten()
        return self.samples[index[randint(0, len(index)-1)]]

    def __init__(self, root, train_or_test, config):
        super(UnTripletFolder, self).__init__(root)
        self.train_or_test = train_or_test
        # get cameras
        self.cameras = np.array(get_cam_and_label(config['dataset'], self.imgs)[0])
        # transform
        self.transform = get_test_transform()
        self.degrade_transform = get_degrade_function(config['degrade_type'])

    def __getitem__(self, index):
        path1, label1 = self.samples[index]
        path2, label2 = self.get_sample_with_different_camera(self.cameras[index])
        img1, img2 = self.loader(path1), self.loader(path2)
        deg1 = self.degrade_transform(img1)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            deg1 = self.transform(deg1)
        return img1, deg1, img2, label1, label2


class PairedFolder(BasicFolder):
    def __init__(self, root, deg_root, train_or_test, config):
        super(PairedFolder, self).__init__(root)
        self.train_or_test = train_or_test
        if train_or_test == 'val':
            self.deg_samples = make_dataset(deg_root, limit=500)
        else:
            self.deg_samples = make_dataset(deg_root)
        print('[{}] {}, {}'.format(self.__class__.__name__, deg_root, len(self.deg_samples)))
        # transform
        self.transform = get_test_transform()
        self.deg_transform = get_low_level_transform(input_size=(256, 128), input_mode='resize_and_crop')

    def __getitem__(self, index):
        path1, label1 = self.samples[index]
        path2 = choice(self.deg_samples)
        img1, img2 = self.loader(path1), self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.deg_transform(img2)
        return img1, img2, img2, label1, -1


# ---------------------------
# Stage II
# ---------------------------
class ReIDFolder(BasicFolder):
    def __init__(self, root, train_or_test, config):
        super(ReIDFolder, self).__init__(root)
        if train_or_test == 'train':
            self.transform = get_train_transform(config['TRANSFORM'])
        else:
            self.transform = get_test_transform()

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index


class ReIDFolderWithDegradation(ReIDFolder):
    def __init__(self, root, train_or_test, config):
        super(ReIDFolderWithDegradation, self).__init__(root, train_or_test, config)
        self.degrade_transform = get_degrade_function(config['degrade_type'])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_deg = self.degrade_transform(sample)
        if self.transform is not None:
            sample = self.transform(sample)
            sample_deg = self.transform(sample_deg)
        return sample, sample_deg, target


# ---------------------------
# API
# ---------------------------
def get_data_folder(folder_type):
    if folder_type == 'semi_triplet':
        return SemiTripletFolder
    elif folder_type == 'un_triplet':
        return UnTripletFolder
    elif folder_type == 'paired':
        return PairedFolder
    elif folder_type == 'reid_deg':
        return ReIDFolderWithDegradation
    elif folder_type == 'reid':
        return ReIDFolder
    assert 0, 'Invalid folder type: {}'.format(folder_type)
