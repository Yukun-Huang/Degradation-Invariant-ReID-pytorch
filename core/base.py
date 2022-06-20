import re
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from core.loss import make_reid_loss, kl_criterion, js_criterion, l1_criterion
from distutils.version import LooseVersion
if LooseVersion(torch.__version__) >= LooseVersion('1.6'):
    from functools import partial
    torch_save = partial(torch.save, _use_new_zipfile_serialization=False)
else:
    torch_save = torch.save


def get_model_list(dirname, key, n_split=None):
    gen_models = []
    for filename in os.listdir(dirname):
        pattern = key + r'_((\d+)|(best)).pt.part' if n_split else key + r'_((\d+)|(best)).pt'
        # pattern = key + r'_(\d+).pt.part' if n_split else key + r'_(\d+).pt'
        matches = re.match(pattern, filename)
        if matches is not None and matches.string == filename:
            gen_models.append(osp.join(dirname, filename))
    if len(gen_models):
        gen_models.sort()
        if n_split:
            final_model_list = gen_models[-n_split:]
            name1 = final_model_list[0].split('.pt.part')[0]
            name2 = final_model_list[1].split('.pt.part')[0]
            assert name1 == name2
        else:
            final_model_list = [gen_models[-1], ]
        return final_model_list


def load_teacher_weights(weight_path, file_name, n_split=None, remove_classifier=False):
    # load state_dict
    if osp.isfile(osp.join(weight_path, file_name)):
        state_dict = torch.load(osp.join(weight_path, file_name))
    else:
        last_model_name_list = get_model_list(weight_path, file_name, n_split=n_split)
        state_dict = OrderedDict()
        for last_model_name in last_model_name_list:
            state_dict.update(torch.load(last_model_name))
    # remove some weights
    remove_key = []
    for key in state_dict.keys():
        if key.startswith('model.fc'):
            remove_key.append(key)
        if key.startswith('classifier') and remove_classifier:
            remove_key.append(key)
    for key in remove_key:
        state_dict.pop(key)
    # return state_dict
    return state_dict


class _Trainer(nn.Module):
    def __init__(self):
        super(_Trainer, self).__init__()
        self.resume_iteration = 0

    @staticmethod
    def update(loss, optimizer, scaler=None):
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    def update_learning_rate(self):
        if hasattr(self, 'dis_scheduler'):
            self.dis_scheduler.step()
        if hasattr(self, 'gen_scheduler'):
            self.gen_scheduler.step()
        if hasattr(self, 'id_scheduler'):
            self.id_scheduler.step()

    def make_loss(self, num_classes=None):
        self.sm = nn.Softmax(dim=1)
        self.log_sm = nn.LogSoftmax(dim=1)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = l1_criterion
        self.kl_criterion = kl_criterion
        if num_classes is not None:
            self.tri_criterion = make_reid_loss('triplet', num_classes)
            self.ce_tri_criterion = make_reid_loss('softmax_triplet', num_classes)
            self.js_criterion = js_criterion

    def save_one(self, snapshot_dir, iterations, name, file_name=None, n_split=None):
        file_name = name if file_name is None else file_name
        if hasattr(self, name):
            # state_dict
            state_dict = getattr(self, name).state_dict()
            if n_split:
                state_dict_list = [OrderedDict() for _ in range(n_split)]
                for i, (n, v) in enumerate(state_dict.items()):
                    state_dict_list[np.random.randint(0, n_split)][n] = v
            else:
                state_dict_list = [state_dict]
            # save
            if isinstance(iterations, int):
                save_path = osp.join(snapshot_dir, '%s_%08d.pt' % (file_name, iterations + 1))
            else:
                save_path = osp.join(snapshot_dir, '%s_%s.pt' % (file_name, iterations))
            for i, state_dict in enumerate(state_dict_list):
                torch_save(state_dict, '{}.part{}'.format(save_path, i) if n_split else save_path)
            print('Save {} to {}'.format(name, save_path))

    def resume_one(self, checkpoint_dir, name, n_iteration=None, file_name=None, strict=True, asserting=True, n_split=None):
        file_name = name if file_name is None else file_name
        if hasattr(self, name):
            try:
                if n_iteration is None:
                    last_model_name_list = get_model_list(checkpoint_dir, file_name, n_split=n_split)
                else:
                    if n_split is not None and n_split > 1:
                        last_model_name_list = [
                            osp.join(checkpoint_dir, '{}_{:08d}.pt.part{}'.format(name, n_iteration, i))
                            for i in range(n_split)
                        ]
                    else:
                        last_model_name_list = [
                            osp.join(checkpoint_dir, '{}_{:08d}.pt'.format(name, n_iteration))
                        ]
                state_dict = OrderedDict()
                for last_model_name in last_model_name_list:
                    state_dict.update(torch.load(last_model_name))
                getattr(self, name).load_state_dict(state_dict, strict=strict)
                print('Resume weights from {} to {}'.format(last_model_name_list[0], name))
                # Save resume_iteration
                resume_iteration = osp.split(last_model_name_list[0])[-1].split('.pt')[0].split('_')[-1]
                resume_iteration = int(resume_iteration) if resume_iteration != 'best' else -1
                if hasattr(self, 'resume_iteration'):
                    self.resume_iteration = max(resume_iteration, self.resume_iteration)
                else:
                    self.resume_iteration = resume_iteration
            except Exception as e:
                if asserting:
                    print(e)
                    assert 0, 'Resume {} failed from {}'.format(name, checkpoint_dir)
        else:
            if asserting:
                assert 0, 'Module {} not found'.format(name)

    def resume_teacher(self, weight_path, name, file_name=None, n_split=None, strict=True, remove_classifier=False):
        state_dict = load_teacher_weights(weight_path, file_name, n_split, remove_classifier)
        getattr(self, name).load_state_dict(state_dict, strict)
        print('Resume weights from {} to self.{}, remove_classifier={}'.format(
            weight_path, name, remove_classifier
        ))
        if not strict:
            missing_keys = []
            for k in getattr(self, name).state_dict().keys():
                if k not in state_dict.keys():
                    missing_keys.append(k)
            if len(missing_keys):
                print('missing_keys: ', missing_keys)

    def forward(self, **kwargs):
        raise NotImplementedError
