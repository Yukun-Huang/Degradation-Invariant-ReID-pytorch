import torch
import torch.nn as nn
from torch.nn import functional
from core.network.block.basic import Conv2dBlock, UpSampleBlock
from core.network.block.sequential import MLP, ResBlocks, SPAdaINResBlock
from core.network.block.init import weights_init


##################################################################################
# Decoder
##################################################################################
class Decoder(nn.Module):
    def __init__(self, input_dim=128, n_upsample=2, n_res=4, activ='lrelu', pad_type='reflect'):
        super(Decoder, self).__init__()
        dim = input_dim
        # res
        self.n_upsample = n_upsample
        self.res = ResBlocks(n_res, dim, 'adain', activ, pad_type=pad_type)
        # up-sample
        self.up = []
        for i in range(n_upsample):
            self.up += [UpSampleBlock(
                conv=Conv2dBlock(dim, dim//2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type),
                scale_factor=2,
                upsample_first=True,
            )]
            dim //= 2
        # use reflection padding in the last conv layer
        self.convs = []
        self.convs += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.convs += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.convs += [Conv2dBlock(dim, 3,   1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        # seq
        self.up = nn.Sequential(*self.up)
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.res(x)
        x = self.up(x)
        x = self.convs(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, input_dim=128, n_res=4, activ='lrelu', pad_type='reflect'):
        super(UnetDecoder, self).__init__()
        # res
        self.res = ResBlocks(n_res, input_dim, 'adain', activ, pad_type=pad_type)
        # up-sample
        self.up1 = UpSampleBlock(
            conv=Conv2dBlock(128, 64, 5, 1, 2, norm='ln', activation='lrelu', pad_type='reflect'),
            scale_factor=2,
            upsample_first=True,
        )
        self.up2 = UpSampleBlock(
            conv=Conv2dBlock(64+32, 32, 5, 1, 2, norm='ln', activation='lrelu', pad_type='reflect'),
            scale_factor=2,
            upsample_first=True,
        )
        # use reflection padding in the last conv layer
        self.convs = []
        self.convs += [Conv2dBlock(32+16, 32, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.convs += [Conv2dBlock(32, 32, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.convs += [Conv2dBlock(32, 3,  1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.convs = nn.Sequential(*self.convs)

    def forward(self, content):
        c3, c2, c1 = content   # c=64x32x32, c1=16x128x128, c2=32x64x64, c3=64x32x32
        x = self.res(c3)    # 128, 32x32
        x = self.up1(x)    # 64, 64x64
        x = torch.cat((x, c2), dim=1)   # 64+32, 64x64
        x = self.up2(x)    # 32, 128x128
        x = torch.cat((x, c1), dim=1)   # 32+16, 128x128
        x = self.convs(x)       # 3,  128x128
        return x


class SpAdaINDecoder(UnetDecoder):
    def __init__(self, input_dim=128, style_dim=128, n_res=4, activ='lrelu', pad_type='reflect'):
        super(SpAdaINDecoder, self).__init__(input_dim, n_res, activ, pad_type)
        self.res = nn.ModuleList()
        for _ in range(n_res):
            self.res.append(SPAdaINResBlock(input_dim, style_dim, activ))

    def forward(self, content, style):
        c3, c2, c1 = content
        for res_block in self.res:
            c3 = res_block(c3, style)
        c3 = torch.cat((self.up1(c3), c2), dim=1)
        c3 = torch.cat((self.up2(c3), c1), dim=1)
        return self.convs(c3)


##################################################################################
# Generator
##################################################################################
class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, config, content_dim=128, style_dim=128):
        super(AdaINGen, self).__init__()
        # Params
        activ = 'lrelu'
        pad_type = 'reflect'
        self.content_dim = content_dim
        self.style_dim = style_dim
        # build network
        self.dec = Decoder(input_dim=content_dim, activ=activ, pad_type=pad_type)
        self.build_MLP_layers(style_dim, content_dim)
        # init
        self.apply(weights_init('kaiming'))

    def decode(self, content, style):
        if isinstance(self.dec, SpAdaINDecoder):
            return self.dec(content, style)
        # decode style codes to an image
        adain_params_w = torch.cat((self.mlp_w1(style), self.mlp_w2(style), self.mlp_w3(style), self.mlp_w4(style)), 1)
        adain_params_b = torch.cat((self.mlp_b1(style), self.mlp_b2(style), self.mlp_b3(style), self.mlp_b4(style)), 1)
        self.assign_AdaIN_params(adain_params_w, adain_params_b, self.dec)
        images = self.dec(content)
        return images

    def forward(self, content, style):
        return self.decode(content, style)

    def build_MLP_layers(self, style_dim, content_dim, activ='lrelu', mlp_dim=512, mlp_norm='none'):
        # MLP to generate AdaIN parameters
        # dim: style_dim -> mlp_dim -> content_dim
        # dim: 128 -> 512 -> 128x2
        self.mlp_w1 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w2 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w3 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_w4 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)

        self.mlp_b1 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b2 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b3 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)
        self.mlp_b4 = MLP(style_dim, 2 * content_dim, mlp_dim, 3, norm=mlp_norm, activ=activ)

    def assign_AdaIN_params(self, adain_params_w, adain_params_b, model):
        # assign the adain_params to the AdaIN layers in model
        dim = self.content_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:, :dim].contiguous()
                std = adain_params_w[:, :dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1) > dim:  # Pop the parameters
                    adain_params_b = adain_params_b[:, dim:]
                    adain_params_w = adain_params_w[:, dim:]

    def get_num_AdaIN_params(self):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += m.num_features
        return num_adain_params


class SpAdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, config, content_dim=128, style_dim=128, activ='lrelu', pad_type='reflect'):
        super(SpAdaINGen, self).__init__()
        # build network
        self.dec = SpAdaINDecoder(input_dim=content_dim, style_dim=style_dim, activ=activ, pad_type=pad_type)
        # init
        self.apply(weights_init('kaiming'))

    def decode(self, content, style):
        if len(style.shape) == 2:
            style = style.unsqueeze(-1).unsqueeze(-1).expand_as(
                content[0] if isinstance(content, tuple) else content)
        return self.dec(content, style)

    def forward(self, content, style):
        return self.decode(content, style)
