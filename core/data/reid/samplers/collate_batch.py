# encoding: utf-8
import torch


def train_collate_fn(batch):
    outputs = []
    for item in zip(*batch):
        if isinstance(item[0], torch.Tensor):
            outputs.append(torch.stack(item, dim=0))
        else:
            outputs.append(torch.tensor(item, dtype=torch.int64))
    return tuple(outputs)


def val_collate_fn(batch):
    outputs = []
    for item in zip(*batch):
        if isinstance(item[0], torch.Tensor):
            outputs.append(torch.stack(item, dim=0))
        else:
            outputs.append(torch.tensor(item, dtype=torch.int64))
    return tuple(outputs)
