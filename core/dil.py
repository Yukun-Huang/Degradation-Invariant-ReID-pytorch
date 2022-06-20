import random
import torch
from core.network import *
from core.scheduler import get_scheduler
from core.base import _Trainer


class DILTrainer(_Trainer):
    def __init__(self, config):
        super(DILTrainer, self).__init__()
        # Initiate the networks
        self.E_con = build_content_encoder(config)
        self.E_deg = build_degradation_encoder(config)
        self.G = build_generator(config)
        self.D_deg = MsImageDis(num_scales=1) if config['degrade_type'] == 'res' else MsImageDis()
        self.D_real = MsImageDis()
        # Initiate the optimizer
        self.dis_opt, self.gen_opt, self.dis_scheduler, self.gen_scheduler = self.make_joint_optimizer(config)
        self.make_loss(config['num_class'])
        # Others
        self.use_rank_discriminator = (config['folder_type'] == 'un_triplet')
        from core.loss import l1_criterion, l2_criterion, smooth_l1_criterion
        self.pixel_criterion = eval('{}_criterion'.format(config['pixel_loss_type']))

    def make_joint_optimizer(self, config):
        # Setup the optimizers of dis and gen
        lr_g = 0.0001
        lr_d = 0.0001
        beta1 = 0
        beta2 = 0.999
        weight_decay = 0.0005
        dis_params = list(self.D_deg.parameters()) + list(self.D_real.parameters())
        dis_params = [p for p in dis_params if p.requires_grad]
        gen_params = list(self.G.parameters()) + list(self.E_deg.parameters()) + list(self.E_con.parameters())
        gen_params = [p for p in gen_params if p.requires_grad]
        dis_opt = torch.optim.Adam(dis_params, lr=lr_d, betas=(beta1, beta2), weight_decay=weight_decay)
        gen_opt = torch.optim.Adam(gen_params, lr=lr_g, betas=(beta1, beta2), weight_decay=weight_decay)
        dis_scheduler = get_scheduler(dis_opt, config['scheduler'])
        gen_scheduler = get_scheduler(gen_opt, config['scheduler'])
        return dis_opt, gen_opt, dis_scheduler, gen_scheduler

    def forward(self, x_a, x_b, syn):
        f_inv_a, c_a = self.E_con(x_a)
        f_inv_b, c_b = self.E_con(x_b)
        d_a = self.E_deg(x_a)
        d_b = self.E_deg(x_b, syn)
        x_aa = self.G(c_a, d_a)
        x_bb = self.G(c_b, d_b)
        x_ab = self.G(c_a, d_b)
        x_ba = self.G(c_b, d_a)
        return x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, c_a, c_b

    def calc_dis_self_loss(self, x_a, x_b, x_ab, x_ba, config):
        # Reality GAN loss
        if random.random() < 0.5:
            loss_dis_real = self.D_real.calc_dis_loss(self.D_real, x_ab.detach(), x_a)
        else:
            loss_dis_real = self.D_real.calc_dis_loss(self.D_real, x_ba.detach(), x_a)
        loss_dis_real *= config['w_dil_real']
        # Degradation GAN loss
        loss_dis_deg = self.D_deg.calc_rank_dis_loss(self.D_deg, x_b, x_a) * config['w_dil_deg']
        # Total loss
        loss_dis_total = loss_dis_deg + loss_dis_real
        loss_dict = {
            'loss_dis_total': loss_dis_total,
            'loss_dis_deg': loss_dis_deg,
            'loss_dis_real': loss_dis_real,
        }
        return loss_dict

    def calc_dis_inter_loss(self, x_a, x_b, x_ab, x_ba, config):
        # Reality GAN loss
        loss_dis_real = (self.D_real.calc_dis_loss(self.D_real, x_ab.detach(), x_a) +
                         self.D_real.calc_dis_loss(self.D_real, x_ba.detach(), x_b)) * config['w_dil_real']
        # Degradation GAN loss
        if self.use_rank_discriminator:
            with torch.no_grad():
                label = self.D_deg.predict(self.D_deg, x_b, x_a)
            loss_dis_deg = self.D_deg.calc_rank_dis_loss(self.D_deg, x_b, x_a, label)
        else:
            loss_dis_deg = self.D_deg.calc_dis_loss(self.D_deg, x_b, x_a) * 2
        loss_dis_deg *= config['w_dil_deg']
        # Total loss
        loss_dis_total = loss_dis_deg + loss_dis_real
        loss_dict = {
            'loss_dis_total': loss_dis_total,
            'loss_dis_deg': loss_dis_deg,
            'loss_dis_real': loss_dis_real,
        }
        return loss_dict

    def calc_gen_self_loss(self, x_a, x_b, x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, c_a, c_b, config):
        # Reality GAN loss
        loss_gen_real = (self.D_real.calc_gen_loss(self.D_real, x_ab, 1) +
                         self.D_real.calc_gen_loss(self.D_real, x_ba, 1)) * config['w_dil_real']
        # Degradation GAN loss
        loss_gen_deg = self.D_deg.calc_rank_gen_loss(self.D_deg, x_ab, x_ba) * config['w_dil_deg']
        # Image Self-Reconstruction loss
        loss_gen_self_rec = (self.pixel_criterion(x_aa, x_a) + self.pixel_criterion(x_bb, x_b)) * config['w_dil_rec']
        # Image Reconstruction loss
        loss_gen_rec = (self.pixel_criterion(x_ab, x_b) + self.pixel_criterion(x_ba, x_a)) * config['w_dil_rec']
        # Content Invariant loss
        loss_gen_inv = self.pixel_criterion(c_a, c_b) * config['w_dil_inv']
        # Identity loss
        loss_gen_id = 0.0
        if config['enable_kl'] and config['w_dil_id'] > 0.0 and f_inv_a is not None:
            loss_gen_id = self.kl_criterion(self.log_sm(self.E_con.classifier(f_inv_b)),
                                            self.sm(self.E_con.classifier(f_inv_a))) * config['w_dil_id']
        # Ablation Analysis
        if config['wo_L_p_inv']: loss_gen_inv = 0.0
        if config['wo_L_p_rec']: loss_gen_rec = 0.0
        if config['wo_L_p_self_rec']: loss_gen_self_rec = 0.0
        if config['wo_L_p_deg']: loss_gen_deg = 0.0
        if config['wo_L_p_real']: loss_gen_real = 0.0
        if config['wo_L_p_id']: loss_gen_id = 0.0
        # Total loss
        loss_dict = {
            'loss_gen_inv': loss_gen_inv,
            'loss_gen_rec': loss_gen_rec,
            'loss_gen_self_rec': loss_gen_self_rec,
            'loss_gen_deg': loss_gen_deg,
            'loss_gen_real': loss_gen_real,
            'loss_gen_id': loss_gen_id,
        }
        loss_gen_total = 0.0
        for v in loss_dict.values():
            loss_gen_total += v
        loss_dict['loss_gen_total'] = loss_gen_total
        return loss_dict

    def calc_gen_inter_loss(self, x_a, x_b, x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, l_a, l_b, config):
        # Reality GAN loss
        loss_gen_real = (self.D_real.calc_gen_loss(self.D_real, x_ab, 1) +
                         self.D_real.calc_gen_loss(self.D_real, x_ba, 1)) * config['w_dil_real']
        # Degradation GAN loss
        if self.use_rank_discriminator:
            with torch.no_grad():
                label = self.D_deg.predict(self.D_deg, x_b, x_a)
            loss_gen_deg = self.D_deg.calc_rank_gen_loss(self.D_deg, x_ab, x_ba, label)
        else:
            loss_gen_deg = self.D_deg.calc_gen_loss(self.D_deg, x_ab, 0) + \
                           self.D_deg.calc_gen_loss(self.D_deg, x_ba, 1)
        loss_gen_deg *= config['w_dil_deg']
        # Image Self-Reconstruction loss
        loss_gen_self_rec = (self.pixel_criterion(x_aa, x_a) + self.pixel_criterion(x_bb, x_b)) * config['w_dil_rec']
        # Image Reconstruction loss
        loss_gen_rec = (self.pixel_criterion(self.G(self.E_con.content(x_ab), self.E_deg(x_ba)), x_a) +
                        self.pixel_criterion(self.G(self.E_con.content(x_ba), self.E_deg(x_ab)), x_b))
        loss_gen_rec *= config['w_dil_rec']
        # Identity loss
        loss_gen_id = 0.0
        if config['w_dil_id'] > 0.0 and f_inv_a is not None:
            if config['degrade_type'] == 'hazy':
                loss_gen_id = self.ce_criterion(self.E_con.classifier(f_inv_a), l_a) * 2 * config['w_dil_id']
            else:
                loss_gen_id = (self.ce_criterion(self.E_con.classifier(f_inv_a), l_a) +
                               self.ce_criterion(self.E_con.classifier(f_inv_b), l_b)) * config['w_dil_id']
        # Ablation Analysis
        if config['wo_L_d_rec']: loss_gen_rec = 0.0
        if config['wo_L_d_self_rec']: loss_gen_self_rec = 0.0
        if config['wo_L_d_deg']: loss_gen_deg = 0.0
        if config['wo_L_d_real']: loss_gen_real = 0.0
        if config['wo_L_d_id']: loss_gen_id = 0.0
        # Total loss
        loss_dict = {
            'loss_gen_rec': loss_gen_rec,
            'loss_gen_self_rec': loss_gen_self_rec,
            'loss_gen_deg': loss_gen_deg,
            'loss_gen_real': loss_gen_real,
            'loss_gen_id': loss_gen_id,
        }
        loss_gen_total = 0.0
        for v in loss_dict.values():
            loss_gen_total += v
        loss_dict['loss_gen_total'] = loss_gen_total
        return loss_dict

    def save(self, snapshot_dir, iterations):
        self.save_one(snapshot_dir, iterations, name='E_con')
        self.save_one(snapshot_dir, iterations, name='E_deg')
        self.save_one(snapshot_dir, iterations, name='G')
        self.save_one(snapshot_dir, iterations, name='D_deg')
        self.save_one(snapshot_dir, iterations, name='D_real')

    def resume(self, checkpoint_dir, n_iteration=None):
        self.resume_one(checkpoint_dir, name='E_con', n_iteration=n_iteration)
        self.resume_one(checkpoint_dir, name='E_deg', n_iteration=n_iteration)
        self.resume_one(checkpoint_dir, name='G', n_iteration=n_iteration)

    def sample(self, x_a, x_b, x_c):
        with torch.no_grad():
            c_a = self.E_con.content(x_a)
            c_b = self.E_con.content(x_b)
            c_c = self.E_con.content(x_c)
            d_a = self.E_deg(x_a, syn=False)
            d_b = self.E_deg(x_b, syn=True)
            d_c = self.E_deg(x_c, syn=False)
            x_aa, x_bb, x_cc = self.G(c_a, d_a), self.G(c_b, d_b), self.G(c_c, d_c)
            x_ab, x_ba = self.G(c_a, d_b), self.G(c_b, d_a)
            x_ac, x_ca = self.G(c_a, d_c), self.G(c_c, d_a)
        return (x_a, x_aa, x_b, x_bb, x_c, x_cc), (x_a, x_ab, x_b, x_ba), (x_a, x_ac, x_c, x_ca)

    def swap(self, x_a, x_b, syn=False):
        with torch.no_grad():
            c_a = self.E_con.content(x_a)
            c_b = self.E_con.content(x_b)
            d_a = self.E_deg(x_a, syn=syn)
            d_b = self.E_deg(x_b, syn=syn)
            x_ab, x_ba = self.G(c_a, d_b), self.G(c_b, d_a)
            x_aa, x_bb = self.G(c_a, d_a), self.G(c_b, d_b)
        return x_aa, x_ab, x_bb, x_ba

    def inference(self, x):
        with torch.no_grad():
            f_inv = self.E_con.identity(x)
        return f_inv
