import os
import os.path as osp
from random import sample
from torchvision.datasets.folder import is_image_file


def make_dataset(data_dir, limit=None):
    images = []
    if not osp.isdir(data_dir):
        print(data_dir)
        raise Exception('Check data dir')
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if is_image_file(filename):
                images.append(osp.join(root, filename))
    if limit is not None:
        images = sample(images, k=limit)
    images = sorted(images)
    return images
