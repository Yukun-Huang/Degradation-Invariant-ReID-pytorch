import torch
import torch.nn as nn
from .basic import LinearBlock, Conv2dBlock, LiteConv2dBlock, get_activation
from .norm import SpatiallyAdaptiveInstanceNorm2d as SPAdaIN


##################################################################################
# Sequential Models
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()
        model = []
        if res_type == 'basic' or res_type == 'nonlocal':
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'slim':
            dim_half = dim // 2
            model += [Conv2dBlock(dim, dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type)]
        else:
            assert 0
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type == 'nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out


class LiteResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='bn', activation='relu', pad_type='reflect', res_type='basic', residual=True):
        super(LiteResBlock, self).__init__()
        model = []
        if res_type == 'basic':
            model += [LiteConv2dBlock(in_dim, in_dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [LiteConv2dBlock(in_dim, out_dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            assert 0
        self.res_type = res_type
        self.residual = residual
        self.model = nn.Sequential(*model)
        self.output = get_activation(activation)

    def forward(self, x):
        residual = x
        out = self.model(x)
        if self.residual:
            out += residual
        out = self.output(out)
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Specific Blocks
##################################################################################
class NonlocalBlock(nn.Module):
    def __init__(self, in_dim):
        super(NonlocalBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self-attention value + input feature
                attention: B * N * N (N is W * H)
        """
        batch_size, n_channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B * C * N
        attention = self.softmax(torch.bmm(proj_query, proj_key))  # B * N * N

        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B * C * N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, n_channels, width, height)
        out = self.gamma * out + x
        return out


class ASPP(nn.Module):
    def __init__(self, input_dim, output_dim, norm='in', activation='relu', pad_type='zero'):
        super(ASPP, self).__init__()
        dim_part = input_dim // 2
        self.conv1 = Conv2dBlock(input_dim, dim_part, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

        self.conv6 = []
        self.conv6 += [Conv2dBlock(input_dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv6 += [Conv2dBlock(dim_part, dim_part, 3, 1, 3, norm=norm, activation='none', pad_type=pad_type, dilation=3)]
        self.conv6 = nn.Sequential(*self.conv6)

        self.conv12 = []
        self.conv12 += [Conv2dBlock(input_dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv12 += [Conv2dBlock(dim_part, dim_part, 3, 1, 6, norm=norm, activation='none', pad_type=pad_type, dilation=6)]
        self.conv12 = nn.Sequential(*self.conv12)

        self.conv18 = []
        self.conv18 += [Conv2dBlock(input_dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv18 += [Conv2dBlock(dim_part, dim_part, 3, 1, 9, norm=norm, activation='none', pad_type=pad_type, dilation=9)]
        self.conv18 = nn.Sequential(*self.conv18)

        self.fuse = Conv2dBlock(4*dim_part, output_dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        out = torch.cat((conv1,conv6,conv12, conv18), dim=1)
        out = self.fuse(out)
        return out


class SPAdaINResBlock(nn.Module):
    def __init__(self, input_nc, style_nc, activ_type):
        super(SPAdaINResBlock, self).__init__()
        self.conv1 = Conv2dBlock(input_nc, input_nc, 3, 1, 1, norm='none', activation='none', pad_type='reflect')
        self.spain1 = SPAdaIN(input_nc, style_nc)
        self.conv2 = Conv2dBlock(input_nc, input_nc, 3, 1, 1, norm='none', activation='none', pad_type='reflect')
        self.spain2 = SPAdaIN(input_nc, style_nc)
        self.activation = get_activation(activ_type)

    def forward(self, x, style):
        residual = x
        x = self.spain1(self.conv1(x), style)
        x = self.activation(x)
        x = self.spain2(self.conv2(x), style)
        x = self.activation(x)
        x = x + residual
        return x
