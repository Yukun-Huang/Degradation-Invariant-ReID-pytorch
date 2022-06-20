import torch
import torch.nn as nn
import random
from copy import deepcopy
from core.base import _Trainer
from core.network import *
from core.network.reid.memory import MemoryBank
from core.network.reid.attention import GuidedReIDAttention
from core.scheduler import get_scheduler


class DualReIDTrainer(_Trainer):
    def __init__(self, config):
        super(DualReIDTrainer, self).__init__()
        self.config = deepcopy(config)
        # Initiate the networks for DIL
        self.E_deg = build_degradation_encoder(config)
        self.A_deg = GuidedReIDAttention(config['attention_type'])
        # Build Content Encoder
        self.E_con = build_content_encoder(config)
        # Build Identity Encoder
        self.E_id = build_identity_encoder(config)
        # Build Classifier
        self.C = build_classifier(config, 2048 + config['feat_inv_dim'])
        # Initiate the optimizer
        self.id_opt, self.id_scheduler = self.make_optimizer(config)
        self.make_loss(config['num_class'])
        # Others
        self.sm = nn.Softmax(dim=1)
        self.log_sm = nn.LogSoftmax(dim=1)
        self.use_attention = config['use_attention']
        # Set train-eval mode
        self.train()

    def make_optimizer(self, config):
        # Setup the id optimizer
        if config['optim'] == 'adam':
            lr_id = config['lr_id']
            id_params = list(self.E_con.model.parameters()) + \
                list(self.E_con.classifier.parameters()) + \
                list(self.E_con.head.parameters()) + \
                list(self.E_id.model.parameters()) + \
                list(self.E_id.classifier.parameters()) + \
                list(self.E_deg.parameters()) + \
                list(self.A_deg.parameters()) + \
                list(self.C.parameters())
            id_opt = torch.optim.Adam([p for p in id_params if p.requires_grad], lr=lr_id, weight_decay=0.0005)
        elif config['optim'] == 'sgd':
            lr_id = config['lr_id']
            id_opt = torch.optim.SGD([
                {'params': self.E_con.model.parameters(),               'lr': lr_id * 0.1},
                {'params': self.E_con.classifier.parameters(),          'lr': lr_id},
                {'params': self.E_con.head.parameters(),               'lr': lr_id},
                {'params': self.E_id.model.parameters(),                'lr': lr_id * 0.1},
                {'params': self.E_id.classifier.parameters(),           'lr': lr_id},
                {'params': self.E_deg.parameters(),                     'lr': lr_id * 0.01},
                {'params': self.A_deg.parameters(),                     'lr': lr_id},
                {'params': self.C.parameters(),                         'lr': lr_id},
            ], momentum=0.9, weight_decay=1e-4, nesterov=True)
        else:
            assert 0, 'Invalid optim type: {}'.format(config['optim'])
        # Setup the id scheduler
        id_scheduler = get_scheduler(id_opt, config['scheduler'])
        return id_opt, id_scheduler

    def forward(self, x, f_d=None):
        f_sen = self.E_id.identity(x)
        if self.use_attention:
            assert f_d is not None
            f_sen = f_sen * self.A_deg(f_sen, f_d)
        p_sen = self.E_id.classifier(f_sen)
        f_inv = self.E_con.identity(x)
        p_inv = self.E_con.classifier(f_inv)
        f_fuse = torch.cat((f_sen, f_inv), dim=1)
        p_fuse = self.C(f_fuse)
        return p_fuse, p_inv, p_sen, f_fuse, f_inv, f_sen

    def id_criterion(self, p, f, target, loss_type):
        if loss_type == 'ce_tri':
            return self.ce_tri_criterion(p, f, target)
        elif loss_type == 'ce':
            return self.ce_criterion(p, target)
        elif loss_type == 'tri':
            return self.tri_criterion(p, f, target)
        elif loss_type == 'none':
            return 0.0

    def smooth_criterion(self, p_inv1, p_inv2, loss_type: str):
        if loss_type.endswith('detach'):
            p_inv1 = p_inv1.detach()
        if loss_type.startswith('kl'):
            return self.kl_criterion(self.log_sm(p_inv2), self.sm(p_inv1))
        elif loss_type.startswith('js'):
            return self.js_criterion(self.sm(p_inv2), self.sm(p_inv1))

    def calc_reid_loss(self, *feats, target, config):
        p_fuse1, p_inv1, p_sen1, f_fuse1, f_inv1, f_sen1 = feats[0]

        loss_id_fuse = self.id_criterion(p_fuse1, f_fuse1, target, config['id_fuse_type']) * config['w_id_fuse']
        loss_id_inv = self.id_criterion(p_inv1, f_inv1, target, config['id_inv_type']) * config['w_id_inv']
        loss_id_sen = self.id_criterion(p_sen1, f_sen1, target, config['id_sen_type']) * config['w_id_sen']

        loss_id_smooth = 0.0
        if config['w_id_smooth'] > 0.0:
            assert len(feats) > 1
            p_fuse2, p_inv2, p_sen2, f_fuse2, f_inv2, f_sen2 = feats[1]
            loss_id_smooth = self.smooth_criterion(p_inv1, p_inv2, config['id_smooth_type']) * config['w_id_smooth']
        # Total loss
        loss_dict = {
            'loss_reid_fuse': loss_id_fuse,
            'loss_reid_inv': loss_id_inv,
            'loss_reid_sen': loss_id_sen,
            'loss_reid_smooth': loss_id_smooth,
        }
        loss_reid_total = 0.0
        for v in loss_dict.values():
            loss_reid_total += v
        loss_dict['loss_reid_total'] = loss_reid_total
        return loss_dict

    def reid_update(self, loss, scaler=None):
        self.id_opt.zero_grad()
        if scaler is None:
            loss.backward()
            self.id_opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(self.id_opt)
            scaler.update()

    def save(self, snapshot_dir, iterations):
        self.save_one(snapshot_dir, iterations, name='E_con')
        self.save_one(snapshot_dir, iterations, name='E_id')
        self.save_one(snapshot_dir, iterations, name='E_deg')
        self.save_one(snapshot_dir, iterations, name='C')
        if self.use_attention:
            self.save_one(snapshot_dir, iterations, name='A_deg')

    def resume(self, checkpoint_dir, resume_DIL=False):
        self.resume_one(checkpoint_dir, name='E_con')
        if self.use_attention:
            self.resume_one(checkpoint_dir, name='E_deg')
        if not resume_DIL:
            self.resume_one(checkpoint_dir, name='E_id')
            self.resume_one(checkpoint_dir, name='C')
            if self.use_attention:
                self.resume_one(checkpoint_dir, name='A_deg')


class ReIDTrainer(DualReIDTrainer):
    def __init__(self, config):
        super(ReIDTrainer, self).__init__(config)
        self.E_id.model.conv1 = nn.Module()
        self.E_id.model.bn1 = nn.Module()
        self.E_id.model.layer1 = nn.Module()
        self.E_id.model.layer2 = nn.Module()
        self.E_id.model.layer3 = nn.Module()
        self.E_con.model.layer4 = nn.Module()

    def _extract_features(self, x):
        x_list = self.E_con.forward_before(x)
        f_inv, _ = self.E_con.head(x_list)
        f_sen = self.E_id.model.layer4(x_list[-1])
        f_sen = self.E_id.model.global_pool(f_sen)
        f_sen = f_sen.view(f_sen.size(0), -1)
        if self.use_attention:
            f_sen = f_sen * self.A_deg(f_sen)
        return f_inv, f_sen

    def forward(self, x, *args):
        f_inv, f_sen = self._extract_features(x)
        f_fuse = torch.cat((f_sen, f_inv), dim=1)
        p_fuse = self.C(f_fuse)
        p_inv = self.E_con.classifier(f_inv)
        p_sen = self.E_id.classifier(f_sen)
        return p_fuse, p_inv, p_sen, f_fuse, f_inv, f_sen

    def inference(self, x, feat_type='fuse'):
        f_inv, f_sen = self._extract_features(x)
        fs_inv = self.E_con.classifier.add_block(f_inv)
        fs_sen = self.E_id.classifier.add_block(f_sen)
        if feat_type == 'inv':
            return fs_inv
        elif feat_type == 'sen':
            return fs_sen
        f_fuse = torch.cat((fs_sen, fs_inv), dim=1)
        return f_fuse


class Augmenter(_Trainer):
    def __init__(self, config):
        super(Augmenter, self).__init__()
        self.config = deepcopy(config)
        # build the networks
        self.E_con = build_content_encoder(config).eval()
        self.E_deg = build_degradation_encoder(config).eval()
        self.G = build_generator(config).eval()
        self.bank = None

    def initialize(self, data_source):
        if isinstance(data_source, str):
            self.bank = MemoryBank(torch.load(data_source, map_location='cuda'))
        else:
            self.bank = MemoryBank(data_source, self.E_deg)

    @torch.no_grad()
    def augment(self, x, aug_mode, step):
        c = self.E_con.content(x)
        n, nch, h, w = c.size()
        if aug_mode == 'bank':
            c = c.unsqueeze(1).expand(n, step, nch, h, w).reshape((n*step, nch, h, w))
            d = self.bank.get_a_batch(n*step).reshape((n*step, -1))
        elif aug_mode == 'shuffle':
            d = self.E_deg(x)
            d = d[torch.randperm(x.size(0))]
        else:
            raise NotImplementedError
        return self.G(c, d)

    @torch.no_grad()
    def forward(self, x, label, config, step=1):
        x_aug = self.augment(x, config['aug_mode'], step)
        l_aug = label.unsqueeze(1).expand(x.size(0), step).flatten()
        return x_aug, l_aug

    @staticmethod
    def mix(x, x_aug, prob=0.5):
        for i in range(x.size(0)):
            if random.random() > prob:
                x[i], x_aug[i] = x_aug[i], x[i]
        return x, x_aug

    def resume(self, checkpoint_dir):
        self.resume_one(checkpoint_dir, name='E_con')
        self.resume_one(checkpoint_dir, name='E_deg')
        self.resume_one(checkpoint_dir, name='G')

    def save_memories(self, save_path):
        deg_feats = self.bank.bank.tensors[0]
        torch.save(deg_feats.cpu(), save_path)
        print('Save memories to {}!'.format(save_path))
