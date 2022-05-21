import torch
import torch.nn as nn
import torch.nn.functional as F
from core.network.block.init import weights_init_kaiming, weights_init_classifier
from core.network.block.basic import NormLinear


######################################################################
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ClassBlock_BNNeck(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_BNNeck, self).__init__()

        self.add_block = nn.BatchNorm1d(input_dim)
        self.add_block.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(input_dim, class_num, bias=False)

        self.add_block.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ClassBlock_BNNeck_Attention(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock_BNNeck_Attention, self).__init__()

        self.add_block = nn.BatchNorm1d(input_dim)
        self.add_block.bias.requires_grad_(False)  # no shift
        self.add_block.apply(weights_init_kaiming)

        self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, w):
        # x: [N, C]
        # w: [N, C]
        x = self.add_block(x)

        w = torch.min(w, torch.ones_like(w) * 0.5)     # [N, C]
        # min operation can be replaced by NLSE (negative-log-sum-exp)

        w = F.normalize(w, p=1, dim=1) * w.size(1)

        x = self.classifier(w * x)
        return x


class ClassBlock_BNNeck_Norm(nn.Module):
    def __init__(self, input_dim, class_num, mode='before'):
        super(ClassBlock_BNNeck_Norm, self).__init__()
        self.add_block = nn.BatchNorm1d(input_dim)
        self.add_block.bias.requires_grad_(False)  # no shift
        # self.classifier = nn.Linear(input_dim, class_num, bias=False)
        self.classifier = NormLinear(input_dim, class_num, bias=False)

        self.add_block.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.mode = mode

    def forward(self, x):
        if self.mode == 'before_bn':
            x = self.add_block(F.normalize(x))
        elif self.mode == 'after_bn':
            x = F.normalize(self.add_block(x))
        elif self.mode == 'wo_bn':
            x = F.normalize(x)
        elif self.mode in ('wo_bn_wo_norm', 'wo_norm_wo_bn'):
            pass
        else:
            assert 0
        x = self.classifier(x)
        return x
