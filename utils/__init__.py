import os
import torch
import time


def set_random_seed(seed=0):
    import random as random_python
    import numpy.random as random_numpy
    random_python.seed(seed)
    random_numpy.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_sub_folder(output_directory, directory_names):
    directory_paths = []
    for directory_name in directory_names:
        directory_path = os.path.join(output_directory, directory_name)
        if not os.path.exists(directory_path):
            print("Creating directory: {}".format(directory_path))
            os.makedirs(directory_path)
        directory_paths.append(directory_path)
    return tuple(directory_paths)


def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        seconds = time.time() - self.start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        during = "{:.0f}h {:.0f}m {:.0f}s".format(h, m, s)
        print(self.msg.format(during))

