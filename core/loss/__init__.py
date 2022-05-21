# encoding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .triplet_loss import TripletLoss
from .label_smooth import CrossEntropyLabelSmooth
from functools import partial


def pixel_wise_criterion(inputs, targets, loss_func, reduction: str = 'mean'):
    if isinstance(inputs, tuple):
        _loss = 0.0
        for (_input, _target) in zip(inputs, targets):
            _loss += loss_func(_input, _target, reduction=reduction)
        return _loss / len(inputs)
    else:
        return loss_func(inputs, targets, reduction=reduction)


l1_criterion = partial(pixel_wise_criterion, loss_func=F.l1_loss)
l2_criterion = mse_criterion = partial(pixel_wise_criterion, loss_func=F.mse_loss)
smooth_l1_criterion = partial(pixel_wise_criterion, loss_func=F.smooth_l1_loss)


def make_reid_loss(loss_type, num_classes, margin=0.3, use_smooth=False):
    triplet = TripletLoss(margin)  # triplet loss
    if use_smooth:
        cross_entropy = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, num_classes:", num_classes)
    if loss_type == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif loss_type == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif loss_type == 'softmax_triplet':
        def loss_func(score, feat, target):
            if use_smooth:
                return cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                return F.cross_entropy(score, target) + triplet(feat, target)[0]
    else:
        raise NotImplementedError
    return loss_func


# Kullback-Leibler Divergence
def kl_criterion(_input: Tensor, _target: Tensor, reduction='batchmean', log_input=True, log_target=False):
    """
    :param _input: log-probability (default) if log_input=True else probability
    :param _target: log-probability if log_target=True else probability (default)
    :param reduction: ['batchmean', 'mean', 'none', 'sum']
    :param log_input: A flag indicating whether '_input' is passed in the log space.
    :param log_target: A flag indicating whether '_target' is passed in the log space.
    :return:
        F.kl_div(log_input, target) = target * (log(target) - log_input)
                                    = target * log(target) - target * log_input
    """
    if not log_input:
        _input = torch.log(_input)
    return F.kl_div(_input, _target, reduction=reduction, log_target=log_target)


# Jensen-Shannon Divergence
def js_criterion(*inputs: Tensor, reduction='batchmean'):
    """
    :param inputs: tuple of probabilities
    :param reduction: ['batchmean', 'mean', 'none', 'sum']
    :return:
        js_divergence = mean( KL(P1//Pm) + KL(P2//Pm) + ... )
    """
    js_div = 0.0
    log_prob_mean = torch.clamp(torch.stack(inputs, dim=0).mean(dim=0), min=1e-7, max=1.).log()
    for _input in inputs:
        js_div += F.kl_div(log_prob_mean, _input, reduction=reduction)
    return js_div / len(inputs)


if __name__ == '__main__':
    p_x = torch.tensor([[0.1, 0.3, 3.5, 0.3, 0.1], [0.1, 0.3, 3.5, 0.3, 0.1]])
    p_y = torch.tensor([[2.7, 0.3, 0.1, 0.3, 0.1], [2.7, 0.3, 0.1, 0.3, 0.1]])

    p_x = torch.softmax(p_x, dim=1)
    p_y = torch.softmax(p_y, dim=1)
    p_x_log = torch.log(p_x)
    p_y_log = torch.log(p_y)
