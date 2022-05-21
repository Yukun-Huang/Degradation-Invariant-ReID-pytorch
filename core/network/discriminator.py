import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
from core.network.block.basic import Conv2dBlock
from core.network.block.sequential import ResBlock
from core.network.block.init import weights_init


##################################################################################
# Discriminator
##################################################################################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim=3, dim=32, num_scales=3, norm='none', activ='lrelu', pad_type='reflect', margin=1.0):
        super(MsImageDis, self).__init__()
        # Make networks
        self.nets = nn.ModuleList()
        for _ in range(num_scales):
            self.nets.append(self._make_net(input_dim, dim, norm, activ, pad_type))
        self.nets.apply(weights_init('gaussian'))
        self.down_sample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        # Others
        self.w_grad = 0.01
        self.num_scales = num_scales
        self.rank_criterion = nn.MarginRankingLoss(margin=margin)

    @staticmethod
    def _make_net(input_dim, dim, norm, activ, pad_type, n_layer=2, n_res=4):
        cnn_x = []
        cnn_x += [Conv2dBlock(input_dim, dim, 1, 1, 0, norm=norm, activation=activ, pad_type=pad_type, bias=True)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type, bias=False)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type, bias=False)]
        for i in range(n_layer - 1):
            dim2 = min(dim*2, 512)
            cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type, bias=False)]
            cnn_x += [Conv2dBlock(dim, dim2, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type, bias=False)]
            dim = dim2
        for i in range(n_res):
            cnn_x += [ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type, res_type='basic')]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    @staticmethod
    def compute_squared_grad(d_out, x_in):
        batch_size = x_in.size(0)
        grad_out = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_out = grad_out.pow(2)
        assert(grad_out.size() == x_in.size())
        return grad_out.view(batch_size, -1).sum(1)

    def forward(self, x):
        outputs = []
        for model in self.nets:
            outputs.append(model(x))
            x = self.down_sample(x)
        return outputs

    def predict(self, model, input0, input1):
        if hasattr(self, 'device'):
            input0 = input0.to(self.device)
            input1 = input1.to(self.device)
        outs0 = model.forward(input0)
        outs1 = model.forward(input1)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            out0 = adaptive_avg_pool2d(out0, output_size=(1, 1)).view(out0.size(0), -1)
            out1 = adaptive_avg_pool2d(out1, output_size=(1, 1)).view(out1.size(0), -1)
            loss += out0 - out1
        label = (torch.tensor(loss >= 0, dtype=torch.float) - 0.5) * 2
        if hasattr(self, 'main_device'):
            return label.to(self.main_device)
        else:
            return label

    def calc_dis_loss(self, model, input_fake, input_real):
        # calculate the loss to train D
        if hasattr(self, 'device'):
            input_fake = input_fake.to(self.device)
            input_real = input_real.to(self.device)
        input_real.requires_grad_()
        outs0 = model.forward(input_fake)
        outs1 = model.forward(input_real)
        loss, reg = 0, 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            reg += self.w_grad * self.compute_squared_grad(out1, input_real).mean()
        loss = (loss + reg) / self.num_scales
        if hasattr(self, 'main_device'):
            return loss.to(self.main_device)
        else:
            return loss

    def calc_gen_loss(self, model, input_fake, fake_or_real=1):
        # calculate the loss to train G
        if hasattr(self, 'device'):
            input_fake = input_fake.to(self.device)
        outs0 = model.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            loss += torch.mean((out0 - fake_or_real)**2) * 2  # LS-GAN
        loss = loss / self.num_scales
        if hasattr(self, 'main_device'):
            return loss.to(self.main_device)
        else:
            return loss

    def calc_rank_dis_loss(self, model, input_lower, input_higher, label=-1):
        # calculate the loss to train D
        if isinstance(label, int) or isinstance(label, float):
            label = label * torch.ones(input_lower.size(0), 1)
        if hasattr(self, 'device'):
            input_lower = input_lower.to(self.device)
            input_higher = input_higher.to(self.device)
        input_higher.requires_grad_()
        outs0 = model.forward(input_lower)
        outs1 = model.forward(input_higher)
        label = label.to(outs0[0].device)
        loss, reg = 0, 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            out0 = adaptive_avg_pool2d(out0, output_size=(1, 1)).view(out0.size(0), -1)
            out1 = adaptive_avg_pool2d(out1, output_size=(1, 1)).view(out1.size(0), -1)
            loss += self.rank_criterion(out0, out1, label)
            reg += self.w_grad * self.compute_squared_grad(out1, input_higher).mean()
        loss = (loss + reg) / self.num_scales
        if hasattr(self, 'main_device'):
            return loss.to(self.main_device)
        else:
            return loss

    def calc_rank_gen_loss(self, model, input_lower, input_higher, label=-1):
        # calculate the loss to train G
        if isinstance(label, int) or isinstance(label, float):
            label = label * torch.ones(input_lower.size(0), 1)
        if hasattr(self, 'device'):
            input_lower = input_lower.to(self.device)
            input_higher = input_higher.to(self.device)
        outs0 = model.forward(input_lower)
        outs1 = model.forward(input_higher)
        label = label.to(outs0[0].device)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += self.rank_criterion(
                input1=adaptive_avg_pool2d(out0, output_size=(1, 1)).view(out0.size(0), -1),
                input2=adaptive_avg_pool2d(out1, output_size=(1, 1)).view(out1.size(0), -1),
                target=label,
            )
        loss = loss / self.num_scales
        if hasattr(self, 'main_device'):
            return loss.to(self.main_device)
        else:
            return loss


class VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.
    It is used to train SRGAN and ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """
    def __init__(self, num_in_ch=3, num_feat=64):
        super(VGGStyleDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        # self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        # self.linear2 = nn.Linear(100, 1)
        self.linear1 = nn.Conv2d(num_feat * 8, 100, 1, 1, 0)
        self.linear2 = nn.Conv2d(100, 1, 1, 1, 0)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # assert x.size(2) == 128 and x.size(3) == 128, (
        #     f'Input spatial size must be 128x128, '
        #     f'but received {x.size()}.')
        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (4, 4)

        # feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out
