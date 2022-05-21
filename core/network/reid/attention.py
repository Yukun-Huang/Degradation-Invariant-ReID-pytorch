import torch
import torch.nn as nn
from core.network.block.basic import Conv2dBlock, LinearBlock
from core.network.block.sequential import MLP
from random import random


class ReIDAttention(nn.Module):
    def __init__(self, input_dim=128, output_dim=2048, middle_dim=256, n_blk=4, output_type='softmax'):
        super(ReIDAttention, self).__init__()
        self.output_dim = output_dim
        self.output_type = output_type
        if output_type == 'softmax':
            self.mlp = MLP(input_dim, output_dim*2, middle_dim, n_blk, norm='bn', activ='relu')
            self.activ = nn.Softmax(dim=1)
        elif output_type == 'sigmoid':
            self.mlp = MLP(input_dim, output_dim, middle_dim, n_blk, norm='bn', activ='relu')
            self.activ = nn.Sigmoid()
        else:
            assert 0

    def forward(self, d):
        w = self.mlp(d)
        if self.output_dim != w.size(1):
            w1 = w[:, :self.output_dim].unsqueeze(dim=1)
            w2 = w[:, self.output_dim:].unsqueeze(dim=1)
            w = self.activ(torch.cat((w1, w2), dim=1))[:, 0, :]
        else:
            w = self.activ(w)
        return w


class GuidedReIDAttention(nn.Module):
    def __init__(self, attention_type='128_bn_sigmoid'):
        super(GuidedReIDAttention, self).__init__()
        latent_dim, norm, output_type = attention_type.split('_')
        latent_dim = int(latent_dim)
        output_dim = 2048
        n_blk = 3
        self.identity2latent = MLP(output_dim, latent_dim, latent_dim*2, n_blk, norm=norm, activ='relu')
        self.deg2latent = LinearBlock(128, latent_dim, norm=norm, activation='relu')
        self.latent2weight = nn.Linear(latent_dim, output_dim, bias=False)

    def forward(self, f_sen, f_d=None):
        if f_d is not None:
            latent_deg = self.identity2latent(f_sen.detach())
            latent_id = self.deg2latent(f_d.detach())
            if random() < 0.5:
                w = torch.sigmoid(self.latent2weight(latent_id))
            else:
                w = torch.sigmoid(self.latent2weight(latent_deg))
        else:
            latent = self.identity2latent(f_sen.detach())
            w = torch.sigmoid(self.latent2weight(latent))
        return w
