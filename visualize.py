import torch
import os
import os.path as osp
from tqdm import tqdm
from config import Config
from functools import partial
from core import build_dil_trainer, Augmenter
from utils.image import load_image_as_tensor
from utils.visualize import write_images

load_func = partial(load_image_as_tensor, size=(256, 128), normalize=True)


@torch.no_grad()
def visualization_via_swapping(input_hr_path, input_lr_path, output_folder):
    x_a = load_func(input_hr_path).unsqueeze(0).cuda()
    x_b = load_func(input_lr_path).unsqueeze(0).cuda()
    x_aa, x_ab, x_bb, x_ba = trainer.swap(x_a, x_b, syn=False)
    os.makedirs(output_folder, exist_ok=True)
    write_images((x_aa,), output_folder, 'img_hr_rec.jpg')
    write_images((x_bb,), output_folder, 'img_lr_rec.jpg')
    write_images((x_ab,), output_folder, 'img_hr2lr.jpg')
    write_images((x_ba,), output_folder, 'img_lr2hr.jpg')


@torch.no_grad()
def visualization_via_memory_replay(input_path, output_path, step=8):
    x = load_func(input_path).unsqueeze(0).cuda()
    x_aug = augmenter.augment(x, aug_mode='bank', step=step)
    write_images((x_aug,), output_path)


if __name__ == '__main__':
    # Setup Config
    opts, config = Config().parse(config_name='config_dil.yml')

    # Visualization - Degradation Swapping
    trainer = build_dil_trainer(config).eval().cuda()
    trainer.resume(os.path.join(opts.output_root, 'checkpoints', 'DIL_trained'))
    visualization_via_swapping(
        './demo/img_hr.jpg',
        './demo/img_lr.jpg',
        './demo/',
    )

    # Visualization - Degradation Memory Replay
    augmenter = Augmenter(config).eval().cuda()
    augmenter.resume(os.path.join(opts.output_root, 'checkpoints', 'DIL_trained'))
    augmenter.initialize(os.path.join(opts.output_root, 'checkpoints', 'DIL_trained', 'deg_memories.pt'))
    visualization_via_memory_replay(
        './demo/img_hr.jpg',
        './demo/img_hr_aug.jpg',
    )
    visualization_via_memory_replay(
        './demo/img_lr.jpg',
        './demo/img_lr_aug.jpg',
    )
