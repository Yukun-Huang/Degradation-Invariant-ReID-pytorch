import torch
import random
import math

__all__ = ['RandomErasing', 'get_random_fill']


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, img_sync=None):

        if random.uniform(0, 1) > self.probability:
            if img_sync is not None:
                return img, img_sync
            else:
                return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    if img_sync is not None:
                        img_sync[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img_sync[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img_sync[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    if img_sync is not None:
                        img_sync[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                if img_sync is not None:
                    return img, img_sync
                else:
                    return img

        if img_sync is not None:
            return img, img_sync
        else:
            return img


class RandomFilling(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image. 0.02
         sh: Maximum proportion of erased area against input image. 0.4
         r1: Minimum aspect ratio of erased area. 0.3
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        random.seed(7)

    def fill_block(self, img, img_of_block):
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = img_of_block[0, x1:x1 + h, y1:y1 + w]
                    img[1, x1:x1 + h, y1:y1 + w] = img_of_block[1, x1:x1 + h, y1:y1 + w]
                    img[2, x1:x1 + h, y1:y1 + w] = img_of_block[2, x1:x1 + h, y1:y1 + w]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = img_of_block[0, x1:x1 + h, y1:y1 + w]

                return img.detach()

        return img

    def __call__(self, img, img_of_block):
        if random.uniform(0, 1) > self.probability:
            return img
        else:
            return self.fill_block(img, img_of_block)


class RandomMixedFilling(object):
    def __init__(self, probability=0.5, mixed_probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self._rand_fill = RandomFilling(probability, sl, sh, r1)
        self.probability = probability
        self.mixed_probability = mixed_probability

    def __call__(self, img, img_of_block):
        if random.uniform(0, 1) > self.probability:
            return img
        else:
            if random.uniform(0, 1) > self.mixed_probability:
                return self._rand_fill.fill_block(img, img_of_block)
            else:
                return self._rand_fill.fill_block(img_of_block, img)


class RandomGaussianFilling(object):
    @staticmethod
    def get_gaussian_kernel(kernel_size=49, sigma=15, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          groups=channels, bias=False, padding=kernel_size // 2,
                                          padding_mode='replicate')

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.blur_layer = self.get_gaussian_kernel().cuda()

    def get_weights(self, img):
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    torch.zero_(img[0, x1:x1 + h, y1:y1 + w])
                    torch.zero_(img[1, x1:x1 + h, y1:y1 + w])
                    torch.zero_(img[2, x1:x1 + h, y1:y1 + w])
                else:
                    torch.zero_(img[0, x1:x1 + h, y1:y1 + w])

                if len(img.shape) == 3:
                    img = self.blur_layer(img.unsqueeze(0)).squeeze(0)
                else:
                    img = self.blur_layer(img)

                return img.detach()

        return img

    def __call__(self, img, img_of_block):
        if random.uniform(0, 1) > self.probability:
            return img
        else:
            with torch.no_grad():
                w = self.get_weights(torch.ones(1, img.size(1), img.size(2)).cuda())
                return w * img + (1 - w) * img_of_block


def get_random_fill(fill_type):
    # 0.5, 0.5_0.5, gauss_0.5
    use_blur = 'gauss' in fill_type
    fill_type = fill_type.replace('gauss_', '').replace('_gauss', '')
    items = fill_type.split('_')
    if len(items) == 1:
        if use_blur:
            return RandomGaussianFilling(probability=float(items[0]))
        else:
            return RandomFilling(probability=float(items[0]))
    if len(items) == 2:
        return RandomMixedFilling(probability=float(items[0]), mixed_probability=float(items[1]))


if __name__ == '__main__':
    from torchvision.utils import save_image

    inp = torch.zeros(1, 128, 128)
    inp_gt = torch.ones(1, 128, 128)

    out = get_random_fill('1.0_gauss')(inp, inp_gt)


