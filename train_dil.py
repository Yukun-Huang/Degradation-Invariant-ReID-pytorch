import os
from time import strftime, localtime
import torch
import torch.backends.cudnn as cudnn
from torch.cuda import amp
import tensorboardX
from random import choice, random

from config import Config
from core.reid_eval import Tester
from core.data import ReidDataset, get_data_iterator
from utils import prepare_sub_folder, set_random_seed, Timer
from utils.visualize import get_loss_info, write_images, get_display_images, write_loss_by_dict
from core.dil import DILTrainer


def checkpoints(save_dir, n_iter, final_test=False):
    trainer.save(save_dir, n_iter)
    if final_test and tester:
        trainer.eval()
        tester.reid_test(trainer.inference)
        trainer.train()


def current_degrade_type(_degrade_list, sample_mode='seq'):
    if len(_degrade_list) == 1 or sample_mode == 'none':
        return _degrade_list[0]
    if sample_mode == 'rand':
        return choice(_degrade_list)
    elif sample_mode.startswith('seq'):
        if not hasattr(current_degrade_type, 'count'):
            current_degrade_type.count = 0
            if sample_mode.startswith('seq_'):
                current_degrade_type.seq_stride = int(sample_mode.lstrip('seq_'))
            else:
                current_degrade_type.seq_stride = 1
        else:
            current_degrade_type.count += 1
        return _degrade_list[current_degrade_type.count // current_degrade_type.seq_stride % len(degrade_list)]


def train(n_iter=1):
    start_iter = n_iter + 1
    cur_deg = None
    scaler = amp.GradScaler()
    while n_iter <= max_iter:
        # Multi-Degradation
        if n_iter % 2:
            cur_deg = current_degrade_type(degrade_list, opts.deg_sample_mode)
            if hasattr(trainer, 'set_current_degradation'):
                trainer.set_current_degradation(cur_deg)
        # Use synthetic data?
        if cur_deg == 'hazy':
            use_sync = False
        elif 0.0 <= config['sync_prob'] <= 1.0:
            use_sync = random() < config['sync_prob']
        else:
            use_sync = n_iter % 2 == 1
        # Main Process
        if use_sync:
            x_a, x_b, _, _, _ = train_data_iter[cur_deg].next()
            x_a, x_b = x_a.half(), x_b.half()
            with amp.autocast(enabled=True):
                x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, c_a, c_b = trainer_forward(x_a, x_b, syn=use_sync)
                dis_info = trainer.calc_dis_self_loss(x_a, x_b, x_ab, x_ba, config)
            trainer.update(dis_info['loss_dis_total'], trainer.dis_opt, scaler)
            with amp.autocast(enabled=True):
                gen_info = trainer.calc_gen_self_loss(x_a, x_b, x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, c_a, c_b, config)
            trainer.update(gen_info['loss_gen_total'], trainer.gen_opt, scaler)
        else:
            x_a, _, x_b, l_a, l_b = train_data_iter[cur_deg].next()
            x_a, x_b = x_a.half(), x_b.half()
            with amp.autocast(enabled=True):
                x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, c_a, c_b = trainer_forward(x_a, x_b, syn=use_sync)
                dis_info = trainer.calc_dis_inter_loss(x_a, x_b, x_ab, x_ba, config)
            trainer.update(dis_info['loss_dis_total'], trainer.dis_opt, scaler)
            with amp.autocast(enabled=True):
                gen_info = trainer.calc_gen_inter_loss(x_a, x_b, x_aa, x_bb, x_ab, x_ba, f_inv_a, f_inv_b, l_a, l_b, config)
            trainer.update(gen_info['loss_gen_total'], trainer.gen_opt, scaler)
        # Scheduler
        trainer.update_learning_rate()
        # Write Logs
        if n_iter % config['log_iter'] in (0, 1) or n_iter == start_iter:
            stage = '-Syn' if use_sync else ''
            loss_info = get_loss_info(dis_info, 'DIS'+stage) + '\n' + get_loss_info(gen_info, 'GEN'+stage) + '\n'
            print("Iter: {:06d}/{:06d},  {}".format(n_iter, max_iter, strftime("%Y-%m-%d %H:%M:%S", localtime())))
            print(loss_info)
            if '=nan' in loss_info or '=inf' in loss_info:
                print('[ERROR] loss value out of range!')
                exit(1)
            write_loss_by_dict(n_iter, dis_info, train_writer)
            write_loss_by_dict(n_iter, gen_info, train_writer)
        # Write Images
        if n_iter % config['image_save_iter'] == 0 or n_iter == start_iter:
            with torch.no_grad():
                x_a, x_b, x_c, _, _ = train_data_iter[cur_deg].next()
                outputs_rec, outputs_self, outputs_inter = trainer.sample(x_a, x_b, x_c)
            write_images(outputs_rec, image_directory, 'DIL_rec_%06d.jpg' % n_iter, config['display_size'])
            write_images(outputs_self, image_directory, 'DIL_self_%06d.jpg' % n_iter)
            write_images(outputs_inter, image_directory, 'DIL_inter_%06d.jpg' % n_iter)
            del outputs_rec, outputs_self, outputs_inter
        # Save Weights
        if n_iter % config['snapshot_save_iter'] == 0 or n_iter == start_iter:
            checkpoints(checkpoint_directory, max_iter - 1)
        # Update Iter
        n_iter += 1
    # Final Save
    checkpoints(checkpoint_directory, max_iter - 1, final_test=True)


if __name__ == '__main__':
    # Init
    set_random_seed()
    num_gpu = torch.cuda.device_count()
    print('Now you are using %d GPUs.' % num_gpu)

    # Setup Config
    opts, config = Config().parse(config_name='config_dil.yml')  # config_dil_half.yml for re-id
    init_iter, max_iter = 1, config['max_iter'],
    degrade_list = opts.degrade_type.split(',') if ',' in opts.degrade_type else [opts.degrade_type]
    config['degrade_list'] = degrade_list

    # Setup logger and output folders
    checkpoint_directory, image_directory, log_directory = \
        prepare_sub_folder(opts.output_path, ('checkpoints', 'images', 'logs'))
    train_writer = tensorboardX.SummaryWriter(log_directory, comment=opts.name)

    # Setup Dataset
    dataset = ReidDataset(config, opts.folder_type)
    tester = Tester(config, dataset)

    # Setup DataLoader
    train_data_iter = {}
    val_data_iter = {}
    train_img_num, val_img_num = None, None
    for degrade_type in degrade_list:
        dataset.config['degrade_type'] = degrade_type
        dataset.config['num_workers'] = round(config['num_workers'] / len(degrade_list) + 0.5)
        # train data
        if degrade_type == 'hazy':
            train_loader, val_loader = dataset.non_sync_train_loader()
        else:
            train_loader, val_loader = dataset.train_loader()
        train_data_iter[degrade_type] = get_data_iterator(train_loader, cuda_prefetch=True)
        # val data
        val_data_iter[degrade_type] = get_display_images(val_loader, config['val_batch_size'])
        # statistics
        if not train_img_num or not val_img_num:
            train_img_num = train_loader.dataset.img_num
            val_img_num = val_loader.dataset.img_num
    n_iter_per_epoch = train_img_num // config['batch_size']
    print('Every epoch need %d iterations' % n_iter_per_epoch)

    # Build Model
    trainer = DILTrainer(config).cuda()

    # Resume Model
    if opts.resume:
        trainer.resume(os.path.join(opts.resume_path, 'checkpoints'))

    # Setup Model
    trainer_forward = trainer.forward
    if num_gpu > 1:
        if opts.data_parallel:
            trainer_forward = torch.nn.DataParallel(trainer)
        else:
            assert num_gpu == 2, 'ModelParallel with num_gpu > 2 is not implemented.'
            main_device = 'cuda:0'
            aux_device = 'cuda:1'
            trainer.D_real.to(aux_device)
            trainer.D_deg.to(aux_device)
            trainer.D_real.device = aux_device
            trainer.D_deg.device = aux_device
            trainer.D_real.main_device = main_device
            trainer.D_deg.main_device = main_device
            print('using ModelParallel')

    del dataset

    # Start Training
    cudnn.benchmark = True
    with Timer('Training complete in {}.'):
        train(init_iter)
