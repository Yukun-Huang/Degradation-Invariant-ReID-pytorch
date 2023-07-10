import os.path as osp
from copy import deepcopy
from torch.utils.data import DataLoader
from .folder import PairedFolder, UnPairedFolder, TestFolder


class LowLightDataset:
    def __init__(self, config, degrade_type=None):
        self.config = deepcopy(config)
        if degrade_type is not None:
            self.config['degrade_type'] = degrade_type

    def train_loader(self):
        return self._train_loader(), self._val_loader()

    def _train_loader(self, root='train/trainA', gt_root='train/trainB', train_or_test='train', distributed=False):
        if train_or_test == 'test':
            shuffle, drop_last = False, False
        else:
            shuffle, drop_last = True, True
        dataset = UnPairedFolder(
                osp.join(self.config['data_root'], root),
                osp.join(self.config['data_root'], gt_root),
                train_or_test=train_or_test,
                config=self.config,
            )
        # Setup the distributed sampler to split the dataset to each GPU.
        sampler = None
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['num_workers'],
            drop_last=drop_last,
            sampler=sampler,
        )

    def _val_loader(self, root='train/valA', gt_root='train/valB'):
        return DataLoader(
            dataset=PairedFolder(
                osp.join(self.config['data_root'], root),
                osp.join(self.config['data_root'], gt_root),
                train_or_test='test',
                config=self.config,
            ),
            batch_size=self.config['val_batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            drop_last=False,
        )

    def test_loader(self, root='test/testA'):
        return DataLoader(
            dataset=TestFolder(
                osp.join(self.config['data_root'], root),
                train_or_test='test',
                config=self.config,
            ),
            batch_size=self.config['test_batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            drop_last=False,
        )
