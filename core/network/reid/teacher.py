import torch
import torch.nn as nn
from torchvision import models
from core.network.reid.classifier import ClassBlock, ClassBlock_BNNeck


class IdentityEncoder(nn.Module):
    def __init__(self, class_num, pool_type='avg', last_stride=1, classifier_type='base'):
        super(IdentityEncoder, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.fc = nn.Sequential()
        # avg pooling to global pooling
        if pool_type == 'max':
            model_ft.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool_type == 'avg':
            model_ft.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # last stride
        if last_stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        self.model = model_ft
        if classifier_type == 'base':
            self.classifier = ClassBlock(2048, class_num)
        elif classifier_type == 'bnneck':
            self.classifier = ClassBlock_BNNeck(2048, class_num)

    def identity(self, x):
        x = self.model.conv1(x)          # 64x128x64
        x = self.model.bn1(x)            # 64x128x64
        x0 = self.model.relu(x)          # 64x128x64
        x = self.model.maxpool(x0)       # 64x64x32
        x1 = self.model.layer1(x)        # 256x64x32
        x2 = self.model.layer2(x1)       # 512x32x16
        x3 = self.model.layer3(x2)       # 1024x16x8
        x4 = self.model.layer4(x3)       # 2048x8x4
        f = self.model.global_pool(x4)
        f = f.view(f.size(0), -1)
        return f

    def forward(self, x):
        return self.identity(x)
