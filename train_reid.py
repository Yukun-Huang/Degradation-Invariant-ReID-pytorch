import os
import torch
import torch.backends.cudnn as cudnn
import tensorboardX
from torch.cuda import amp
from functools import partial

from config import Config
from core.reid import ReIDTrainer, Augmenter
from core.reid_eval import Tester
from core.data import ReidDataset, get_data_iterator
from utils import prepare_sub_folder, set_random_seed, Timer
from utils.visualize import write_images, get_loss_info, write_loss_by_dict


def checkpoints(save_dir, n_iter):
    trainer.save(save_dir, n_iter)


def test():
    trainer.eval()
    results = [
        tester.reid_test(partial(trainer.inference, feat_type='fuse')),
        tester.reid_test(partial(trainer.inference_bn, feat_type='fuse')),
        tester.reid_test(partial(trainer.inference, feat_type='inv')),
        tester.reid_test(partial(trainer.inference_bn, feat_type='inv')),
        tester.reid_test(partial(trainer.inference, feat_type='sen')),
        tester.reid_test(partial(trainer.inference_bn, feat_type='sen')),
    ]
    trainer.train()
    names = ['r1', 'r5', 'r10', 'r20', 'map']
    result_dict = {}
    for r, r_bn, r_inv, r_inv_bn, r_sen, r_sen_bn, name in zip(*results, names):
        if name == 'r20':
            continue
        result_dict[name] = r * 100
        result_dict[name + '_bn'] = r_bn * 100
        result_dict[name + '_inv'] = r_inv * 100
        result_dict[name + '_inv_bn'] = r_inv_bn * 100
        result_dict[name + '_sen'] = r_sen * 100
        result_dict[name + '_sen_bn'] = r_sen_bn * 100
    return result_dict


def train(init_iter):
    n_iter = init_iter
    scaler = amp.GradScaler()
    for n_epoch in range(1, max_epoch + 1):
        # Main training code
        for x, target, indices in train_loader:
            # Forward
            x, target = x.cuda().half(), target.cuda()
            with amp.autocast(enabled=True):
                if not augmenter:
                    feats = trainer_forward(x)
                    reid_info = trainer.calc_reid_loss(feats, target=target, config=config)
                else:
                    with torch.no_grad():
                        x_aug, l_aug = augmenter(x, target, config)
                        if opts.degrade_type == 'hazy':
                            x, x_aug = augmenter.mix(x, x_aug, prob=0.5)
                    if trainer_parallel:
                        p_fuse, p_inv, p_sen, f_fuse, f_inv, f_sen = trainer_forward(torch.cat((x, x_aug), dim=0))
                        n = x.size(0)
                        feats1 = (p_fuse[:n], p_inv[:n], p_sen[:n], f_fuse[:n], f_inv[:n], f_sen[:n])
                        feats2 = (p_fuse[n:], p_inv[n:], p_sen[n:], f_fuse[n:], f_inv[n:], f_sen[n:])
                    else:
                        feats1 = trainer_forward(x)
                        feats2 = trainer_forward(x_aug)
                    reid_info = trainer.calc_reid_loss(feats1, feats2, target=target, config=config)
            # Update
            trainer.reid_update(reid_info['loss_reid_total'], scaler)
            # Logging
            if n_iter % config['log_iter'] == 0 or n_iter == init_iter:
                loss_info = get_loss_info(reid_info, 'ReID') + '\n'
                print("Epoch: %03d/%03d Iteration: %08d/%08d" % (n_epoch, max_epoch, n_iter, max_iter))
                print(loss_info)
                if '=nan' in loss_info or '=inf' in loss_info:
                    print('[ERROR] loss value out of range!')
                    exit(1)
                write_loss_by_dict(n_iter, reid_info, train_writer)
            # Save images
            if n_iter % config['image_save_iter'] == 0 or n_iter == init_iter:
                write_images((x,) if augmenter is None else (x, x_aug), image_directory, 'reid_%08d.jpg' % n_iter,
                             max_display_size=config['display_size'])
            # Checkpoints
            if n_iter == init_iter:
                trainer.save(checkpoint_directory, max_iter - 1)
                # write_loss_by_dict(n_iter, test(), train_writer)
            # Update each iteration
            n_iter += 1
        # Test
        # write_loss_by_dict(n_iter, test(), train_writer)
        # Update each epoch
        trainer.update_learning_rate()
        if n_epoch % config['snapshot_save_epoch'] == 0:
            trainer.save(checkpoint_directory, max_iter - 1)

    # Training finished
    trainer.save(checkpoint_directory, max_iter - 1)
    write_loss_by_dict(n_iter, test(), train_writer)


if __name__ == '__main__':
    # Init
    set_random_seed()

    # Setup Config
    opts, config = Config().parse(config_name='config_reid.yml')
    num_gpu = torch.cuda.device_count()
    print('Now you are using %d GPUs.' % num_gpu)

    # Setup Directories
    checkpoint_directory, image_directory, log_directory = \
        prepare_sub_folder(opts.output_path, ('checkpoints', 'images', 'logs'))
    train_writer = tensorboardX.SummaryWriter(log_directory, comment=opts.name)
    resume_directory = os.path.join(opts.resume_path, 'checkpoints')

    # Setup Dataset
    reid_dataset = ReidDataset(config, config['folder_type'])
    train_loader, val_loader = reid_dataset.train_loader()

    # Setup Tester
    if opts.testset in ['hazy', 'haze'] or opts.degrade_type in ['hazy', 'haze']:
        tester = Tester(config, reid_dataset, query_folder='query-hazy')
    else:
        tester = Tester(config, reid_dataset)

    # Setup Augmenter
    augmenter, aug_mode = None, config['aug_mode']
    if aug_mode not in (None, 'none'):
        # Initialize Augmenter
        augmenter = Augmenter(config).cuda()
        augmenter.resume(resume_directory)
        # Initialize Memory Bank
        if aug_mode.startswith('bank'):
            if opts.degrade_type == 'hazy':
                augmenter.initialize(reid_dataset.non_sync_train_loader()[0])
            else:
                augmenter.initialize(reid_dataset.reid_loader(folder_name='train'))

    # Setup Trainer
    trainer = ReIDTrainer(config).cuda()
    if not opts.wo_DIL:
        trainer.resume(resume_directory, DIL_only=True)

    # Data Parallel
    if num_gpu > 1:
        trainer_parallel = torch.nn.DataParallel(trainer)
        trainer_forward = trainer_parallel.forward
        print('using DataParallel')
    else:
        trainer_parallel = None
        trainer_forward = trainer.forward

    # Release Dataset
    del reid_dataset

    # Start training
    max_epoch = config['max_epoch']
    max_iter = max_epoch * len(train_loader) + trainer.resume_iteration
    assert trainer.resume_iteration not in (1, 2)

    cudnn.benchmark = True
    with Timer('Training complete in {}.'):
        train(1 + trainer.resume_iteration)
