import torch
import numpy as np


# --------------------- Numpy --------------------- #
def normalize_numpy(x: np.ndarray, order=2, axis=1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(x, ord=order, axis=axis, keepdims=True)
    return x / (norm + np.finfo(np.float32).eps)


def compute_dist_numpy(array1, array2, metric='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array with shape [m1, n]
        array2: numpy array with shape [m2, n]
        metric: one of ['cosine', 'euclidean']
    Returns:
        numpy array with shape [m1, m2]
    """
    assert metric in ['cosine', 'euclidean']
    if metric == 'cosine':
        array1 = normalize_numpy(array1, axis=1)
        array2 = normalize_numpy(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


# --------------------- PyTorch --------------------- #
def normalize(x: torch.Tensor, p=2, dim=-1):
    return x.div(torch.norm(x, p=p, dim=dim, keepdim=True).expand_as(x))


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x.float(), y.t().float(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def batch_euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [N, m, d]
        y: pytorch Variable, with shape [N, n, d]
    Returns:
        dist: pytorch Variable, with shape [N, m, n]
    """
    assert len(x.size()) == 3
    assert len(y.size()) == 3
    assert x.size(0) == y.size(0)
    assert x.size(-1) == y.size(-1)

    N, m, d = x.size()
    N, n, d = y.size()

    # shape [N, m, n]
    xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
    dist = xx + yy
    dist.baddbmm_(x.float(), y.permute(0, 2, 1).float(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def weighted_euclidean_dist(x, y, w):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
      y: pytorch Variable, with shape [n, d]
      w: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [n, n]
    """
    m, n = x.size(0), y.size(0)
    assert m == n

    xx = torch.pow(x, 2)            # [n, d]
    xx = xx * w                     # [n, d]
    xx = xx.sum(1, keepdim=True)    # [n, 1]
    xx = xx.expand(m, n)            # [n, n]

    yy = torch.pow(x, 2)            # [n, d]
    yy = yy * w                     # [n, d]
    yy = yy.sum(1, keepdim=True)    # [n, 1]
    yy = yy.expand(n, m).t()        # [n, n]

    dist = xx + yy  # [n, n]
    '''dist = 1 * dist - 2 * (x @ y_t)'''
    dist.addmm_(w * x.float(), y.t().float(), beta=1, alpha=-2)
    return dist.clamp(min=1e-12).sqrt()  # for numerical stability


def compute_dist(x: torch.Tensor, y: torch.Tensor, metric='euclidean', use_norm=False):
    """
    Args:
        x: pytorch Tensor, with shape [m, d]
        y: pytorch Tensor, with shape [n, d]
        use_norm: L2-normalized inputs
        metric: distance metric
    Returns:
        dist: pytorch Tensor, with shape [m, n]
    """
    assert metric in ['inner', 'cosine', 'euclidean']
    if use_norm:
        x = normalize(x)
        y = normalize(y)
    if metric == 'inner':
        return torch.mm(x, y.T)
    elif metric == 'euclidean':
        return euclidean_dist(x, y)
    elif metric == 'cosine':
        return torch.cosine_similarity(x[:, :, None], y.T[None, :, :])
