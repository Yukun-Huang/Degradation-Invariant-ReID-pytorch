import re
import numpy as np
import os.path as osp


__all__ = ['get_cam_and_label']


def _get_id_market_duke(img_paths):
    camera_id = []
    labels = []
    for path, v in img_paths:
        filename = osp.split(path)[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def _get_id_cuhk03(img_paths):
    camera_id = []
    labels = []
    for path, v in img_paths:
        filename = osp.split(path)[-1]
        camera = 2 * (int(filename.split('_')[0]) - 1) + int(filename.split('_')[2])
        label = path.split('/')[-2]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


def _get_id_msmt17(img_paths):
    camera_id = []
    labels = []
    for path, v in img_paths:
        filename = osp.split(path)[-1]
        label = filename[0:4]
        camera = filename.split('_')[2]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


def _get_id_resolution(img_paths):
    camera_id = []
    labels = []
    for path, v in img_paths:
        filename = osp.split(path)[-1]
        label = filename[0:5]
        camera = filename.split('c')[1][0]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels


def _get_id_vehicle(img_paths, ordered_label=False):
    camera_ids, labels = [], []
    pattern = re.compile(r'([-\d]+)_c([-\d]+)')
    for path, pid_ordered in img_paths:
        pid, cam_id = map(int, pattern.search(path).groups())
        if pid == -1:  # junk images are just ignored
            continue
        camera_ids.append(cam_id - 1)  # camera index starts from 0
        if ordered_label:
            labels.append(pid_ordered)
        else:
            labels.append(pid)
    return camera_ids, labels


def get_cam_and_label(dataset, image_paths) -> (list, list):
    if dataset in ['msmt17']:
        return _get_id_msmt17(image_paths)
    elif dataset in ['cuhk03d', 'cuhk03l']:
        return _get_id_cuhk03(image_paths)
    elif dataset in ['caviar', 'mlr_cuhk03', 'mlr_viper', 'mlr_market']:
        return _get_id_resolution(image_paths)
    elif dataset in ['market', 'duke', 'mde_market']:
        return _get_id_market_duke(image_paths)
    elif dataset in ['veri', 'vric']:
        return _get_id_vehicle(image_paths)
    assert 0, 'invalid camera parse function'

