import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import LayerNorm, AdaptiveInstanceNorm2d


def get_padding(padding, pad_type):
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        return nn.ZeroPad2d(padding)
    assert 0, "Unsupported padding type: {}".format(pad_type)


def get_norm(norm_dim, norm_type):
    if norm_type == 'bn':
        return nn.BatchNorm2d(norm_dim)
    elif norm_type == 'in':
        return nn.InstanceNorm2d(norm_dim)
    elif norm_type == 'ln':
        return LayerNorm(norm_dim)
    elif norm_type == 'adain':
        return AdaptiveInstanceNorm2d(norm_dim)
    elif norm_type == 'none':
        return None
    assert 0, "Unsupported normalization: {}".format(norm_type)


def get_norm_1d(norm_dim, norm_type):
    if norm_type == 'bn':
        return nn.BatchNorm1d(norm_dim)
    elif norm_type == 'in':
        return nn.InstanceNorm1d(norm_dim)
    elif norm_type == 'ln':
        return nn.LayerNorm(norm_dim)
    elif norm_type == 'none':
        return None
    assert 0, "Unsupported normalization: {}".format(norm_type)


def get_activation(activ_type):
    if activ_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activ_type == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activ_type == 'prelu':
        return nn.PReLU()
    elif activ_type == 'selu':
        return nn.SELU(inplace=True)
    elif activ_type == 'tanh':
        return nn.Tanh()
    elif activ_type == 'none':
        return lambda x: x
    assert 0, "Unsupported activation: {}".format(activ_type)


##################################################################################
# My Block
##################################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0,
                 norm='none', activation='relu', pad_type='zero', dilation=1, bias=True):
        super(Conv2dBlock, self).__init__()
        # initialize padding
        self.pad = get_padding(padding, pad_type)
        # initialize normalization
        self.norm = get_norm(output_dim, norm)
        # initialize activation
        self.activation = get_activation(activation)
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        # initialize activation
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class LiteConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1, padding=0, pad_type='zero',
                 norm='none', activation='relu', dilation=1, bias=True):
        super(LiteConv2dBlock, self).__init__()
        self.conv_1x1 = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        self.conv_dw = nn.Conv2d(output_dim, output_dim, kernel_size, stride=stride, bias=bias,
                                 dilation=dilation, groups=output_dim)
        # initialize padding
        self.pad = get_padding(padding, pad_type)
        # initialize normalization
        self.norm = get_norm(output_dim, norm)
        # initialize activation
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv_1x1(x)
        x = self.conv_dw(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, conv, scale_factor, mode='bilinear', upsample_first=True):
        super(UpSampleBlock, self).__init__()
        self.conv = conv
        self.scale_factor = scale_factor
        self.upsample_first = upsample_first
        self.mode = mode
        self.use_scale = scale_factor not in (None, 'none', 1, 1.0)

    def forward(self, x):
        if self.use_scale and self.upsample_first:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        x = self.conv(x)
        if self.use_scale and not self.upsample_first:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


class NormLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::
        >>> m = nn.Linear(20, 30)
        >>> output = m(torch.randn(128, 20))
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, F.normalize(self.weight), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
