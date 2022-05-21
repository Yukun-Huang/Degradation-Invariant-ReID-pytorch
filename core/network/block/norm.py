import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor': # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# SPADE
class SpatiallyAdaptiveNorm2d(nn.Module):
    def __init__(self, norm_nc, label_nc, param_free_norm_type, kernel_size):
        super().__init__()

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        n_hidden = 128

        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, n_hidden, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(n_hidden, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(n_hidden, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x, style):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        if x.size()[2:] != style.size()[2:]:
            # 'nearest' for seg map
            style = F.interpolate(style, size=x.size()[2:], mode='bilinear', align_corners=True)
        style = self.mlp_shared(style)
        gamma = self.mlp_gamma(style)
        beta = self.mlp_beta(style)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SpatiallyAdaptiveInstanceNorm2d(SpatiallyAdaptiveNorm2d):
    def __init__(self, norm_dim, label_dim, kernel_size=3):
        super().__init__(norm_dim, label_dim, 'instance', kernel_size)


class SpatiallyAdaptiveBatchNorm2d(SpatiallyAdaptiveNorm2d):
    def __init__(self, norm_dim, label_dim, kernel_size=3):
        super().__init__(norm_dim, label_dim, 'batch', kernel_size)
