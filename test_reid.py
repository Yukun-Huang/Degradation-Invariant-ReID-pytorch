import os
import torch
import torch.backends.cudnn as cudnn
from core import build_reid_trainer
from core.data import ReidDataset
from core.reid_eval import Tester
from config import Config
from utils import set_random_seed
from functools import partial


if __name__ == '__main__':
    set_random_seed()
    cudnn.benchmark = True
    opts, config = Config().parse(config_name='config_reid.yml')
    num_gpu = torch.cuda.device_count()
    print('Now you are using %d GPUs.' % num_gpu)

    trainer = build_reid_trainer(config).eval().cuda()
    trainer.resume(os.path.join(opts.output_root, 'checkpoints'))

    reid_dataset = ReidDataset(config)
    if opts.testset in ['hazy', 'haze'] or opts.degrade_type in ['hazy', 'haze']:
        tester = Tester(config, reid_dataset, query_folder='query-hazy')
    else:
        tester = Tester(config, reid_dataset)

    print('Evaluation with f_fuse:')
    tester.reid_test(partial(trainer.inference, feat_type='fuse'), verbose=False)
    print('Evaluation with f_inv:')
    tester.reid_test(partial(trainer.inference, feat_type='inv'), verbose=False)
    print('Evaluation with f_sen:')
    tester.reid_test(partial(trainer.inference, feat_type='sen'), verbose=False)
