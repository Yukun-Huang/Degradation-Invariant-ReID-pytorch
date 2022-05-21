import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from core.network.block.basic import Conv2dBlock, UpSampleBlock
from core.network.block.sequential import ASPP, LiteResBlock
from core.network.block.init import weights_init
from core.network.reid.classifier import ClassBlock, ClassBlock_BNNeck


##################################################################################
# Content Encoder
##################################################################################
class ContentHead(nn.Module):
    def __init__(self, content_dim=128, identity_dim=2048, norm_type='bn'):
        super(ContentHead, self).__init__()
        self.aspp = ASPP(64, 128, norm='in', activation='lrelu', pad_type='reflect')
        input_dim = [128, 256, 512, 1024]
        # build invariance modules
        self.invariance = nn.ModuleList([LiteResBlock(dim, dim, norm_type) for dim in input_dim])
        # build content modules
        output_dim = [64, 64, 64, 64]
        scale_factors = [1, 1, 2, 4]
        self.content = nn.ModuleList([
            self.build_resize_block(input_dim[i], output_dim[i], scale_factors[i])
            for i in range(len(output_dim))
        ])
        self.content.append(Conv2dBlock(sum(output_dim), content_dim, 1, 1, 0, norm='in', activation='lrelu'))
        # build identity modules
        output_dim = [128, 256, 512, 1024]
        scale_factors = [-1, -1, -1, -1]
        self.identity = nn.ModuleList([
            self.build_resize_block(scale_factor=scale_factors[i],
                                    conv_layer=nn.Sequential(
                                    LiteResBlock(input_dim[i], input_dim[i], 'bn'),
                                    LiteResBlock(input_dim[i], output_dim[i], 'bn'),
                                    ))
            for i in range(len(output_dim))
        ])
        self.identity.append(Conv2dBlock(sum(output_dim), identity_dim, 1, 1, 0, norm='bn', activation='lrelu'))
        # init
        self.apply(weights_init('kaiming'))

    @staticmethod
    def build_resize_block(in_dim=None, out_dim=None, scale_factor=-1, conv_layer=None):
        if conv_layer is None:
            conv_layer = Conv2dBlock(in_dim, out_dim, 3, 1, 1, pad_type='zero', norm='bn', activation='lrelu')
        if scale_factor > 0:
            return UpSampleBlock(conv=conv_layer, scale_factor=scale_factor)
        else:
            return nn.Sequential(conv_layer, nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x_list):
        # invariance
        x_list[0] = F.max_pool2d(self.aspp(x_list[0]), kernel_size=2)
        # feature
        f, c = [], []
        for i, x in enumerate(x_list):
            x = self.invariance[i](x)
            f.append(self.identity[i](x))
            c.append(self.content[i](x))
        f = self.identity[-1](torch.cat(f, dim=1))
        c = self.content[-1](torch.cat(c, dim=1))
        if f.size(-1) == f.size(-2) == 1:
            f = f.view(f.size(0), -1)
        return f, c


class ContentEncoder(nn.Module):
    def __init__(self, config, pool_type='avg', last_stride=1, classifier_type='base'):
        super(ContentEncoder, self).__init__()
        class_num = config['num_class']
        identity_dim = config['feat_inv_dim']
        # Modules
        self.head = ContentHead(identity_dim=identity_dim)
        # Backbone: build self.model and self.classifier
        self._make_model(class_num, pool_type, last_stride, classifier_type, input_dim=identity_dim)

    def _make_model(self, class_num, pool_type, last_stride, classifier_type, input_dim):
        # Model
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential()
        # avg pooling to global pooling
        if pool_type == 'max':
            self.model.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool_type == 'avg':
            self.model.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # last stride
        if last_stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1, 1)
        self.model.layer4 = nn.Sequential()
        # Classifier
        if classifier_type == 'base':
            self.classifier = ClassBlock(input_dim, class_num)
        elif classifier_type == 'bnneck':
            self.classifier = ClassBlock_BNNeck(input_dim, class_num)

    def forward_before(self, x):
        x = self.model.conv1(x)  # 64,128x64
        x = self.model.bn1(x)  # 64,128x64
        x0 = self.model.relu(x)  # 64,128x64
        xp = self.model.maxpool(x0)  # 64,64x32
        x1 = self.model.layer1(xp)  # 256,64x32
        x2 = self.model.layer2(x1)  # 512,32x16
        x3 = self.model.layer3(x2)  # 1024,16x8
        return [x0, x1, x2, x3]

    def forward_before_with_no_grad(self, x):
        with torch.no_grad():
            return self.forward_before(x)

    def forward(self, x):
        x_list = self.forward_before_with_no_grad(x)
        f, c = self.head(x_list)
        return f, c

    def content(self, x):
        x_list = self.forward_before_with_no_grad(x)
        _, c = self.head(x_list)
        return c

    def identity(self, x):
        x_list = self.forward_before(x)
        f, _ = self.head(x_list)
        return f


class DegradationEncoder(nn.Module):
    def __init__(self, config, input_dim=3, dim=64, style_dim=128, n_down=4, norm='none', activ='relu', pad_type='reflect'):
        super(DegradationEncoder, self).__init__()
        self.conv1 = Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.conv2 = Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        # block
        self.block = []
        for i in range(2):
            self.block += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_down - 2):
            self.block += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.block = nn.Sequential(*self.block)
        # global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output layer
        self.conv3_real = nn.Conv2d(dim, style_dim, 1, 1, 0)
        self.conv3_sync = nn.Conv2d(dim, style_dim, 1, 1, 0)
        # init
        self.apply(weights_init('kaiming'))

    def forward(self, x, syn=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block(x)
        x = self.avgpool(x)
        if syn:
            x = self.conv3_sync(x)
        else:
            x = self.conv3_real(x)
        x = x.view(x.size(0), -1)
        return x


##################################################################################
# Test
##################################################################################
if __name__ == '__main__':
    net = ContentHead()
    print(net)
