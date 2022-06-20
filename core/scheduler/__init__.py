# encoding: utf-8
from torch.optim import lr_scheduler
from .warmup import WarmupMultiStepLR


def get_scheduler(optimizer, config, gamma=0.1, iterations=-1):
    if 'lr_policy' not in config or config['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif config['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=gamma, last_epoch=iterations)
    elif config['lr_policy'] == 'multistep':
        step = config['step_size']
        milestones = [step, step+step//2, step+step//2+step//4]  # 50000 -- 75000 --
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=iterations)
    elif config['lr_policy'] == 'warmup':
        scheduler = WarmupMultiStepLR(optimizer, milestones=config['milestones'], gamma=gamma, last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['lr_policy'])
    return scheduler
