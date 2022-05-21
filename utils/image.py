import torch
import cv2
import math
from PIL import Image
import numpy as np
from torchvision.transforms import functional, RandomCrop
try:
    from torchvision.transforms import InterpolationMode
    default_interpolation_mode = InterpolationMode.BICUBIC
except Exception as e:
    default_interpolation_mode = Image.BICUBIC
    print(e)


def load_image_as_tensor(path, normalize=False, size=None, pre_process=None, crop=False):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    if pre_process is not None:
        img = pre_process(img)
    if size is not None:
        if crop:
            img = RandomCrop(size)(img)
        else:
            img = functional.resize(img, size, interpolation=default_interpolation_mode)
    img = functional.to_tensor(img)
    if normalize:
        img = functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img


def save_image_from_tensor(tensor, path, recover=False):
    if recover:
        tensor = to_recover(tensor)
    if len(tensor.shape) == 4:
        assert tensor.size(0) == 1
        tensor = tensor[0]
    img = to_cv2(tensor, to_BGR=True)
    cv2.imwrite(path, img)
    # tensor = tensor.cpu().numpy().transpose((1, 2, 0))
    # img = Image.fromarray(tensor, mode='RGB')
    # img.save(path)


def to_gray(x, recover=True, simple=False, normalize=False):
    if recover:
        x = to_recover(x)
    if simple:
        x = torch.mean(x, dim=1, keepdim=True)
    else:
        x = to_recover(x, mean=(0.0, 0.0, 0.0), std=(0.299, 0.587, 0.114))
        x = torch.mean(x, dim=1, keepdim=True)
    if normalize:
        min_x, max_x = torch.min(x), torch.max(x)
        x = (x - min_x) / (max_x - min_x)
    return x


def to_brightness(x, recover=True, normalize=False):
    if recover:
        x = to_recover(x)
    x, _ = torch.max(x, dim=1, keepdim=True)
    if normalize:
        min_x, max_x = torch.min(x), torch.max(x)
        x = (x - min_x) / (max_x - min_x)
    return x


def to_edge(x, recover=True, add_noise=False):
    if recover:
        x = to_recover(x)
    out = torch.zeros(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = cv2.cvtColor(to_cv2(x, to_uint8=True), cv2.COLOR_RGB2GRAY)  # 256x128x1
        xx = cv2.Canny(xx, 10, 200)                                      # 256x128
        xx = xx / 255.0 - 0.5  # {-0.5, 0.5}
        if add_noise:
            xx += np.random.randn(xx.shape[0], xx.shape[1]) * 0.1  # add random noise
        out[i, :, :] = torch.from_numpy(xx.astype(np.float32))
    out = out.unsqueeze(1)
    return out.to(x.device)


def to_cv2(x, to_uint8=True, to_BGR=False):
    assert (len(x.shape) == 3) or (len(x.shape) == 4 and x.size(0) == 1)
    if len(x.shape) == 4:
        x = x[0]
    x = x.cpu().numpy().transpose((1, 2, 0))
    if to_uint8:
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    if to_BGR:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x


def to_recover(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    assert len(x.shape) == 4 and x.size(1) == 3
    t_mean = torch.tensor(mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device)
    t_std = torch.tensor(std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device)
    return x * t_std + t_mean


def calc_psnr(x1, x2):
    assert len(x1.shape) == 4 and x1.shape == x2.shape
    if x1.size(0) == 1:
        mse = ((x1 - x2) ** 2).mean()
        psnr = 10 * math.log10(1 / mse)
    else:
        psnr = 0.
        for xx1, xx2 in zip(x1, x2):
            mse = ((xx1 - xx2) ** 2).mean()
            psnr += 10 * math.log10(1 / mse)
    return psnr


def calc_ssim(x1, x2):
    from .ssim import ssim
    assert len(x1.shape) == 4 and x1.shape == x2.shape
    if x1.size(0) == 1:
        score = ssim(x1, x2).item()
    else:
        score = 0.
        for xx1, xx2 in zip(x1, x2):
            xx1 = xx1.unsqueeze(0)
            xx2 = xx2.unsqueeze(0)
            score += ssim(xx1, xx2).item()
    return score


def metric_with_reference(x1, x2, method, use_recover=True):
    assert len(x1.shape) == 4 and x1.shape == x2.shape
    if use_recover:
        x1, x2 = to_recover(x1), to_recover(x2)
    if method == 'psnr':
        return calc_psnr(x1, x2)
    if method == 'ssim':
        return calc_ssim(x1, x2)
    if method == 'all':
        return calc_psnr(x1, x2), calc_ssim(x1, x2)
    assert 0, 'invalid method: {}'.format(method)
