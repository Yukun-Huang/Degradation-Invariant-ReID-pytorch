# -*- coding: utf-8 -*-
import os
import re
import torch
from torch.nn import functional
from PIL import Image
import numpy as np
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt
from .image import to_recover


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) if not callable(getattr(trainer, attr)) and
               not attr.startswith("__") and ('loss' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations)


def get_loss_info(loss_dict, phase):
    loss_info = f'[{phase}]  '
    for k, v in sorted(loss_dict.items()):
        if k.endswith('_total'):
            continue
        k = k.replace('loss_', 'L_')
        if isinstance(v, torch.Tensor):
            v = torch.mean(v).item()
        loss_info += f'{k}={v:.3f}  '
    return loss_info


def write_loss_by_dict(iterations, loss_dict, train_writer):
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            v = torch.mean(v).item()
        train_writer.add_scalar(k, float(v), iterations)


def write_loss_by_str(iterations, loss_info, train_writer):
    float_pattern = r'[-+]?[0-9]*\.?[0-9]+'
    loss_pattern = r'L_[a-zA-Z0-9_-]+'
    for res in re.compile(loss_pattern + r'=' + float_pattern).findall(loss_info):
        name, value = res.split('=')
        train_writer.add_scalar(name, float(value), iterations)


def write_images(image_outputs, image_directory, file_name=None, max_display_size=None, recover=True):
    if max_display_size is None:
        display_size = image_outputs[0].size(0)
    else:
        display_size = min(image_outputs[0].size(0), max_display_size)
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    image_tensor = torch.cat([images[:display_size] for images in image_outputs], dim=0)
    if recover:
        image_grid = make_grid(to_recover(image_tensor), nrow=display_size, padding=0, normalize=False)
    else:
        image_grid = make_grid(image_tensor, nrow=display_size, padding=0, normalize=True, scale_each=True)
    save_path = image_directory if file_name is None else os.path.join(image_directory, file_name)
    save_image(image_grid, save_path, nrow=1)


def get_display_images(loader, display_size):
    rand_perm = np.random.permutation(loader.dataset.img_num)[0:display_size]
    list_a, list_b, list_c = [], [], []
    for i in rand_perm:
        one_batch = loader.dataset[i]
        img_a, img_b, img_c = one_batch[0], one_batch[1], one_batch[2]
        list_a.append(img_a)
        list_b.append(img_b)
        list_c.append(img_c)
    display_images_a = torch.stack(list_a).cuda()
    display_images_b = torch.stack(list_b).cuda()
    display_images_c = torch.stack(list_c).cuda()
    return display_images_a, display_images_b, display_images_c


class LossVisualizer:
    def __init__(self):
        fig = plt.figure()
        self.ax0 = fig.add_subplot(121, title="loss")
        self.ax1 = fig.add_subplot(122, title="err")
        self.fig = fig

        self.x_epoch = []
        self.y_epoch = {
            'train_loss': [], 'train_err': [], 'val_loss':   [], 'val_err':   [],
        }

    def update(self, epoch, train_loss, train_err, val_loss, val_err):
        self.x_epoch.append(epoch)
        self.y_epoch['train_loss'].append(train_loss)
        self.y_epoch['train_err'].append(train_err)
        self.y_epoch['val_loss'].append(val_loss)
        self.y_epoch['val_err'].append(val_err)

    def draw_curve(self, save_dir):
        self.ax0.plot(self.x_epoch, self.y_epoch['train_loss'], 'bo-', label='train')
        self.ax0.plot(self.x_epoch, self.y_epoch['val_loss'], 'ro-', label='val')
        self.ax1.plot(self.x_epoch, self.y_epoch['train_err'], 'bo-', label='train')
        self.ax1.plot(self.x_epoch, self.y_epoch['val_err'], 'ro-', label='val')
        if len(self.x_epoch) == 1:
            self.ax0.legend()
            self.ax1.legend()
        self.fig.savefig(os.path.join(save_dir, 'train.jpg'))


class FeatureVisualizer:
    def __init__(self, size=(256, 128), norm='spatial', color_map='jet', interpolate='bilinear'):
        self.color_map = plt.get_cmap(color_map)
        self.norm = norm
        self.size = size
        self.interpolate = interpolate
        self.height = size[0]
        self.width = size[1]
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.t_mean = torch.Tensor([[[[0.485]], [[0.456]], [[0.406]]]])
        self.t_std = torch.Tensor([[[[0.229]], [[0.224]], [[0.225]]]])

    @staticmethod
    def _check_dimension(x):
        assert x.dim() == 3 or x.dim() == 4, 'Input should be 3D or 4D tensor.'

    def _normalize(self, x, p=2):
        assert x.size(0) == 1
        if self.norm == 'minmax':
            max_val, min_val = torch.max(x), torch.min(x)
            x = (x - min_val) / (max_val - min_val)
        elif self.norm == 'spatial':
            x = x.div(torch.norm(x.flatten(), p=p, dim=0))
        elif self.norm == 'abs':
            x = torch.abs(x)
        return x

    def _recover_numpy(self, x):
        x = self.std * x + self.mean
        x = x * 255.0
        x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
        return x

    def _recover_torch(self, x):
        if x.is_cuda:
            x = x.detach().cpu()
        return x * self.t_std + self.t_mean

    def _transform_image(self, x, recover):
        """
        Transform image from torch.tensor to numpy.array.
        """
        # reduce by batch, choose the first sample
        x = x[0].unsqueeze(dim=0) if x.dim() == 4 else x.unsqueeze(dim=0)
        x = functional.interpolate(x, size=self.size, mode=self.interpolate, align_corners=False)
        x = x.squeeze(dim=0).detach().cpu().numpy().transpose((1, 2, 0))
        return self._recover_numpy(x) if recover else x

    def _transform_feature(self, f, reduce_type='sum'):
        """
        Transform feature from torch.tensor to numpy.array.
        """
        # reduce by batch, choose the first sample
        f = f[0].unsqueeze(dim=0) if f.dim() == 4 else f.unsqueeze(dim=0)
        f = functional.interpolate(f, size=self.size, mode=self.interpolate, align_corners=False)
        if reduce_type == 'mean':
            f = f.mean(dim=1, keepdim=True)  # reduce by channel
        elif reduce_type == 'sum':
            f = f.sum(dim=1, keepdim=True)  # reduce by channel
        f = self._normalize(f)
        return f.squeeze().detach().cpu().numpy()

    def _draw(self, x, ca, color_map):
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        cmp = color_map if color_map is not None else self.color_map
        ca.imshow(x, cmap=cmp)

    def save_feature(self, f, save_path, color_map=None, show=False):
        self._check_dimension(f)
        f = self._transform_feature(f.cpu())
        self._draw(f, plt.gca(), color_map)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        if show:
            plt.show()
        plt.close()

    def save_image(self, img, save_path, n_row=8, recover=True):
        self._check_dimension(img)
        if recover:
            img = self._recover_torch(img)
        save_image(img, save_path, nrow=n_row)

    @staticmethod
    def save_mixed_image(img_path, feat_path, save_path, alpha=0.5, beta=None):
        if beta is None:
            beta = 1 - alpha
        img1 = Image.open(img_path).convert('RGB')
        img2 = Image.open(feat_path).convert('RGB').resize(img1.size)
        img_mix = np.array(img1) * alpha + np.array(img2) * beta
        Image.fromarray(img_mix.clip(0.0, 255.0).astype(np.uint8)).save(save_path)

    def save_both(self, img, f, save_path, recover=True, show=False):
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="image")
        ax1 = fig.add_subplot(122, title="feature")
        self._draw(self._transform_image(img, recover), ax0)
        self._draw(self._transform_feature(f), ax1)
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()
