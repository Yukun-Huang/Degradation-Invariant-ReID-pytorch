import os
import os.path as osp
import ast
import yaml
import argparse
from .utils import print_options, sync_options


SINGLE_SHOT_DATASETS = ['caviar', 'mlr_cuhk03', 'mlr_viper']
UNPAIRED_DATASETS = ['msmt17']
LOW_LEVEL_DATASETS = ['enlighten']

dataset_info = {
    'market':       ('Market-1501', 751),
    'msmt17':       ('MSMT17_V1', 1041),
    'mlr_cuhk03':   ('MLR_CUHK03', 1367),
    'mlr_viper':    ('MLR_VIPER', 316),
    'caviar':       ('CAVIAR', 25),
    'enlighten':    ('EnlightenGAN', 2),
}


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--name', type=str, default=None, help="model name")
        parser.add_argument('--config_name', type=str, default=None, help='name of the config file.')

        parser.add_argument('--batch_size', default=None, type=int, help='batch size')
        parser.add_argument('--val_batch_size', default=None, type=int, help='val batch size')
        parser.add_argument('--test_batch_size', default=None, type=int, help='test batch size')
        parser.add_argument('--num_instance', default=None, type=int, help='num_instance per identity')
        parser.add_argument('--num_workers', default=4, type=int, help='num of workers')
        parser.add_argument('--display_size', default=8, type=int, help='display_size')

        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--fast", action="store_true")
        parser.add_argument("--data_parallel", action="store_true")

        parser.add_argument('--data_root', default='./data/resolution-reid/', type=str, help='dataset')
        parser.add_argument('--output_root', type=str, default='./outputs/', help="outputs path")
        parser.add_argument("--teacher_root", type=str, default=None, help="teacher path")

        parser.add_argument('--output_path', type=str, default=None, help="output path")
        parser.add_argument("--resume_path", type=str, default=None, help="resume path")

        self.parser = parser

    def general_init(self):
        self.parser.add_argument('--dataset', default='mlr_cuhk03', type=str, help='dataset')
        self.parser.add_argument('--degrade_type', default='res', type=str, help='degrade type')
        # For ReID
        self.parser.add_argument('--eval_mode', type=str, default='default', help='multi_shot, single_shot')
        self.parser.add_argument('--align_norm', default='bn', type=str, help='bn, in')
        self.parser.add_argument('--feat_inv_dim', default=2048, type=int, help='feat_inv_dim')
        self.parser.add_argument('--hazyset', type=str, default='real', help='voc, real')
        # For Enhance
        self.parser.add_argument('--input_size', default=256, type=int, help='input_size: 128')
        self.parser.add_argument('--input_size_test', default=256, type=int, help='input_size')
        self.parser.add_argument('--n_cluster', default=128, type=int)

    def dil_init(self):
        self.parser.add_argument('--folder_type', type=str, default='auto', help='auto')
        self.parser.add_argument('--deg_sample_mode', type=str, default='none', help='none, rand, seq')
        self.parser.add_argument('--sync_prob', type=float, default=-1., help='')
        self.parser.add_argument('--sync_ds_ratio', type=str, default='(3.0,5.0)', help='')
        # loss weights
        self.parser.add_argument('--w_dil_real', type=float, default=1.0, help='')
        self.parser.add_argument('--w_dil_deg', type=float, default=1.0, help='')
        self.parser.add_argument('--w_dil_rec', type=float, default=10.0, help='')
        self.parser.add_argument('--w_dil_inv', type=float, default=5.0, help='')
        self.parser.add_argument('--w_dil_id', type=float, default=0.5, help='')
        # loss weights
        self.parser.add_argument('--pixel_loss_type', type=str, default='l1', help='l1, smooth_l1, l2')
        self.parser.add_argument('--enable_kl', action="store_true")
        # loss ablation --- pixel-aligned
        self.parser.add_argument('--wo_L_p_inv', action="store_true")
        self.parser.add_argument('--wo_L_p_self_rec', action="store_true")
        self.parser.add_argument('--wo_L_p_rec', action="store_true")
        self.parser.add_argument('--wo_L_p_id', action="store_true")
        self.parser.add_argument('--wo_L_p_real', action="store_true")
        self.parser.add_argument('--wo_L_p_deg', action="store_true")
        # loss ablation --- pixel-unaligned
        self.parser.add_argument('--wo_L_d_self_rec', action="store_true")
        self.parser.add_argument('--wo_L_d_rec', action="store_true")
        self.parser.add_argument('--wo_L_d_id', action="store_true")
        self.parser.add_argument('--wo_L_d_real', action="store_true")
        self.parser.add_argument('--wo_L_d_deg', action="store_true")

    def reid_init(self):
        # basic
        self.parser.add_argument('--folder_type', type=str, default='reid', help='folder type')
        self.parser.add_argument('--wo_DIL', action="store_true")
        # attention
        self.parser.add_argument('--use_attention', type=ast.literal_eval, default=False)
        self.parser.add_argument('--attention_type', type=str, default='128_bn_sigmoid')
        # loss weights
        self.parser.add_argument('--w_id_inv', type=float, default=0.5, help='')
        self.parser.add_argument('--w_id_sen', type=float, default=0.5, help='')
        self.parser.add_argument('--w_id_fuse', type=float, default=1.0, help='')
        self.parser.add_argument('--w_id_smooth', type=float, default=5.0, help='')
        self.parser.add_argument('--aug_mode', type=str, default='bank', help='none, shuffle, shuffle_hard, bank')
        # identity loss type
        self.parser.add_argument('--id_fuse_type', type=str, default='ce_tri', help='')
        self.parser.add_argument('--id_inv_type', type=str, default='ce_tri', help='')
        self.parser.add_argument('--id_sen_type', type=str, default='ce_tri', help='')
        self.parser.add_argument('--id_smooth_type', type=str, default='kl_detach', help='kl_detach, kl, js, js_detach')
        # test
        self.parser.add_argument('--testset', type=str, default='default', help='')

    @staticmethod
    def special_options(opt):
        opt.num_class = dataset_info[opt.dataset][1]
        opt.name = opt.dataset if opt.name is None else '{}_{}'.format(opt.dataset, opt.name)
        opt.data_root = os.path.join(opt.data_root, dataset_info[opt.dataset][0])
        opt.output_path = osp.join(opt.output_root, opt.name) if not opt.output_path else opt.output_path
        opt.resume_path = osp.join(opt.output_root, opt.name) if not opt.resume_path else opt.resume_path
        if '_dil' in opt.config_name:
            if opt.folder_type == 'auto':
                opt.folder_type = 'un_triplet' if opt.dataset in UNPAIRED_DATASETS else 'semi_triplet'
        opt.degrade_type = opt.degrade_type.replace(',', '+')
        return opt

    def parse(self, config_name):
        # add options
        self.general_init()
        if '_dil' in config_name:
            self.dil_init()
        if '_reid' in config_name:
            self.reid_init()
        # gather options
        opt = self.parser.parse_args()
        if not opt.config_name:
            opt.config_name = config_name
        opt = self.special_options(opt)
        # load config
        with open(osp.join(osp.split(__file__)[0], opt.config_name), 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        # sync
        opt, config = sync_options(opt, config)
        # print and return
        print_options(opt, self.parser)
        return opt, config
