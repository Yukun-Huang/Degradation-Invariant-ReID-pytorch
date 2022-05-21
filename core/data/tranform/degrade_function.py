import cv2
import numpy as np
from random import random, shuffle
from PIL import Image, ImageFilter
from functools import partial
from torchvision.transforms.functional import adjust_gamma


# ---------------------------
# Degradation Transform: PIL.Image.Image -> PIL.Image.Image
# ---------------------------
def gamma_degradation(src, gamma_range=(2.0, 3.5), gain_range=(0.75, 1.0)):
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    gain = np.random.uniform(gain_range[0], gain_range[1])
    return adjust_gamma(img=src, gamma=gamma, gain=gain)


def down_sample_degradation(src: Image.Image, ratio_range=(3.0, 5.0)):
    factor = np.random.uniform(*ratio_range)
    target_size = [round(src.size[0] // factor), round(src.size[1] // factor)]
    dst = src.resize(size=target_size, resample=Image.BILINEAR)
    return dst


def jpg_degradation(src):
    params = (cv2.IMWRITE_JPEG_QUALITY, np.random.uniform(10.0, 20.0))
    _, buff = cv2.imencode('.jpg', np.array(src), params)
    deg = cv2.imdecode(buff, cv2.IMREAD_UNCHANGED)
    return Image.fromarray(deg)


def gaussian_blur_degradation(src: Image.Image):
    original_size, fixed_size = src.size, (128, 256)
    radius = np.random.randint(low=3, high=7)  # 5
    src = src.resize(size=fixed_size, resample=Image.BILINEAR)
    dst = src.filter(ImageFilter.GaussianBlur(radius=radius))
    dst = dst.resize(size=original_size, resample=Image.BILINEAR)
    return dst


def motion_blur_degradation(src):
    src = np.array(src)
    degree = np.random.randint(5, 15)
    angle = np.random.uniform(0.0, 180.0)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    rotation_matrix = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, rotation_matrix, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(src, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, np.min(src), np.max(src), norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    # blurred = np.array(blurred, dtype=np.uint8)
    blurred = Image.fromarray(blurred)
    return blurred


def add_noise(img, noise_type, noise_param=10.):
    """Adds Gaussian or Poisson noise to image."""

    w, h = img.size
    c = len(img.getbands())

    # Poisson distribution
    # It is unclear how the paper handles this. Poisson noise is not additive,
    # it is data dependent, meaning that adding sampled valued from a Poisson
    # will change the image intensity...
    if noise_type == 'poisson':
        noise = np.random.poisson(img)
        noise_img = img + noise
        noise_img = 255 * (noise_img / np.amax(noise_img))
    # Normal distribution (default)
    elif noise_type == 'gaussian':
        if isinstance(noise_param, float):
            std = noise_param
        elif isinstance(noise_param, tuple) or isinstance(noise_param, list):
            std = np.random.uniform(noise_param[0], noise_param[1])
        noise = np.random.normal(0, std, (h, w, c))
        # Add noise and clip
        noise_img = np.array(img) + noise

    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noise_img)


def dark_degradation(src):
    if random() < 0.5:
        return add_noise(gamma_degradation(src), 'poisson')
    else:
        return gamma_degradation(add_noise(src, 'poisson'))


deg_func_dict = {
    'ill': gamma_degradation,
    'res': down_sample_degradation,
    'jpg': jpg_degradation,
    'motion': motion_blur_degradation,
    'blur': gaussian_blur_degradation,
    'noise': partial(add_noise, noise_type='gaussian'),
    'poisson': partial(add_noise, noise_type='poisson'),
    'dark': dark_degradation,
}


def get_binomial_distribution(n, q):
    from scipy.special import comb
    p_list = [comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(n + 1)]
    return p_list


def mixed_degradation(image, degrade_list, prob: float):
    for i, deg_type in enumerate(degrade_list):
        if random() < prob:
            image = deg_func_dict[deg_type](image)
    return image


def conditional_mixed_degradation(image, degrade_list, prob: list):
    shuffle(degrade_list)
    for i, deg_type in enumerate(degrade_list):
        image = deg_func_dict[deg_type](image)
        if random() < prob[i]:
            break
    return image


def get_degrade_function(degrade_type):
    if '+' in degrade_type:
        items = degrade_type.split('_')
        degrade_list, prob = items[0].split('+'), float(items[1])
        if len(items) == 2:
            bi_dist = get_binomial_distribution(len(degrade_list), prob)   # len = n + 1
            prob = [bi_dist[i] / sum(bi_dist[i:]) for i in range(1, len(bi_dist))]
            return partial(conditional_mixed_degradation, degrade_list=degrade_list, prob=prob)
        else:
            assert items[-1] == 'clear'
            return partial(mixed_degradation, degrade_list=degrade_list, prob=prob)
    else:
        return deg_func_dict[degrade_type]


# if __name__ == '__main__':
#     from PIL import Image
#     src = Image.open('src.jpg')
#     for i in range(10):
#         dst = gamma_degradation(src).save(f'dst_{i}.jpg')
