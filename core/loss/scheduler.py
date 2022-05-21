# encoding: utf-8
from bisect import bisect_right
import torch
from torch.optim import lr_scheduler


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones=(40, 70),
        gamma=0.1,
        warmup_factor=0.01,
        warmup_iters=10,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def get_scheduler(optimizer, config, gamma=0.1, iterations=-1):
    if 'lr_policy' not in config or config['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif config['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=gamma, last_epoch=iterations)
    elif config['lr_policy'] == 'multistep':
        step = config['step_size']
        milestones = [step, step+step//2, step+step//2+step//4] # #50000 -- 75000 --
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=iterations)
    elif config['lr_policy'] == 'warmup':
        scheduler = WarmupMultiStepLR(optimizer, milestones=config['milestones'], gamma=gamma, last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['lr_policy'])
    return scheduler

