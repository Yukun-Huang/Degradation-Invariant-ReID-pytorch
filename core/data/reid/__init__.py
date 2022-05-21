import os
import os.path as osp
from copy import deepcopy
from torch.utils.data import DataLoader
from .samplers import RandomIdentitySampler, train_collate_fn
from .folder import get_data_folder
from .cam_parser import get_cam_and_label


# ---------------------------
# DATASET
# ---------------------------
class ReidDataset:
    def __init__(self, config, folder_type=None):
        self.config = deepcopy(config)
        # data folder
        if folder_type is None:
            folder_type = config['folder_type']
        self.train_folder = get_data_folder(folder_type)
        self.test_folder = get_data_folder('reid')
        # triplet or not
        self.num_instance = config['num_instance']
        if self.num_instance > 0:
            if config['dataset'] == 'mlr_viper':
                self.num_instance = min(self.num_instance, 2)
        # default config
        self.batch_dict = {'train': self.config['batch_size'], 'val': self.config['val_batch_size']}
        self.shuffle_dict = {'train': True, 'val': False}

    def train_loader(self, train_folder_name='train', val_folder_name='val'):
        self.batch_dict = {'train': self.config['batch_size'], 'val': self.config['val_batch_size']}
        self.shuffle_dict = {'train': True, 'val': False}
        dataset_dict = {
            'train': self.train_folder(osp.join(self.config['data_root'], train_folder_name), 'train', self.config),
            'val':   self.train_folder(osp.join(self.config['data_root'], val_folder_name), 'train', self.config)
        }
        if self.num_instance > 0:
            loader_dict = {x: DataLoader(dataset_dict[x],
                                         batch_size=self.batch_dict[x],
                                         sampler=RandomIdentitySampler(dataset_dict[x].imgs,
                                                                       self.batch_dict[x],
                                                                       self.num_instance),
                                         num_workers=self.config['num_workers'],
                                         collate_fn=train_collate_fn,
                                         drop_last=True)
                           for x in ['train', 'val']}
        else:
            loader_dict = {x: DataLoader(dataset_dict[x],
                                         batch_size=self.batch_dict[x],
                                         shuffle=self.shuffle_dict[x],
                                         num_workers=self.config['num_workers'],
                                         drop_last=True)
                           for x in ['train', 'val']}
        return loader_dict['train'], loader_dict['val']

    def non_sync_train_loader(self, train_folder_name='train', val_folder_name='val'):
        self.batch_dict = {'train': self.config['batch_size'], 'val': self.config['val_batch_size']}
        self.shuffle_dict = {'train': True, 'val': False}
        if self.config['hazyset'] == 'voc':
            hazy_path = osp.join(self.config['root'], 'low-quality', 'VOC2012-hazy')
        elif self.config['hazyset'] == 'real':
            hazy_path = osp.join(self.config['root'], 'low-quality', 'RESIDE-beta', 'Unannotated_Real-world_Hazy_Images')
        else:
            assert 0, 'Invalid hazyset: {}'.format(self.config['hazyset'])
        train_folder = get_data_folder('paired')
        dataset_dict = {
            'train': train_folder(osp.join(self.config['data_root'], train_folder_name), hazy_path, 'train', self.config),
            'val':   train_folder(osp.join(self.config['data_root'], val_folder_name), hazy_path, 'val', self.config)
        }
        loader_dict = {x: DataLoader(dataset_dict[x],
                                     batch_size=self.batch_dict[x],
                                     shuffle=self.shuffle_dict[x],
                                     num_workers=self.config['num_workers'],
                                     drop_last=True)
                       for x in ['train', 'val']}
        return loader_dict['train'], loader_dict['val']

    def test_loader(self, gallery_folder='gallery', query_folder='query'):
        dataset_dict = {
            'gallery': self.test_folder(osp.join(self.config['data_root'], gallery_folder), 'test', self.config),
            'query':   self.test_folder(osp.join(self.config['data_root'], query_folder), 'test', self.config)
        }
        loader_dict = {x: DataLoader(dataset_dict[x],
                                     batch_size=self.config['test_batch_size'],
                                     num_workers=self.config['num_workers'],
                                     shuffle=False,
                                     )
                       for x in ['gallery', 'query']}
        gallery_cam, gallery_label = get_cam_and_label(self.config['dataset'], dataset_dict['gallery'].imgs)
        query_cam, query_label = get_cam_and_label(self.config['dataset'], dataset_dict['query'].imgs)
        return loader_dict, gallery_cam, gallery_label, query_cam, query_label

    def reid_loader(self, folder_name, folder_type='reid', train_or_test='test', batch_size=None, shuffle=False):
        dataset = get_data_folder(folder_type)(
            root=osp.join(self.config['data_root'], folder_name),
            train_or_test=train_or_test,
            config=self.config,
        )
        return DataLoader(dataset=dataset,
                          batch_size=batch_size if batch_size else self.config['test_batch_size'],
                          num_workers=self.config['num_workers'],
                          shuffle=shuffle)
