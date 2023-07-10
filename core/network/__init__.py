import os
from .generator import AdaINGen, SpAdaINGen
from .discriminator import MsImageDis
from .reid.encoder import ContentEncoder, DegradationEncoder
from .reid.teacher import IdentityEncoder
from .reid.classifier import ClassBlock, ClassBlock_BNNeck
from ..base import load_teacher_weights

__all__ = ['build_generator', 'build_degradation_encoder', 'build_content_encoder', 'build_identity_encoder',
           'build_classifier', 'MsImageDis']

BACKBONE_PARAMS = {
    'base': {'last_stride': 2, 'pool_type': 'max', 'classifier_type': 'base'},
    'bot': {'last_stride': 1, 'pool_type': 'avg', 'classifier_type': 'bnneck'},
}


def build_generator(config):
    return AdaINGen(config)


def build_degradation_encoder(config):
    return DegradationEncoder(config)


def build_content_encoder(config, remove_classifier=True):
    # Build Content Encoder
    net = ContentEncoder(config, **BACKBONE_PARAMS['base'])
    # Resume weights
    if config['teacher_root'] is not None:
        teacher_directory = os.path.join(config['teacher_root'], config['dataset'])
        state_dict = load_teacher_weights(teacher_directory, file_name='net_last.pth', remove_classifier=remove_classifier)
        net.load_state_dict(state_dict, strict=False)
        print(f'Resume weights from {teacher_directory} to E_con, remove_classifier={remove_classifier}')
    return net


def build_identity_encoder(config, remove_classifier=False):
    # Build Identity Encoder
    net = IdentityEncoder(config['num_class'], **BACKBONE_PARAMS['base'])
    # Resume weights
    if config['teacher_root'] is not None:
        teacher_directory = os.path.join(config['teacher_root'], config['dataset'])
        state_dict = load_teacher_weights(teacher_directory, file_name='net_last.pth', n_split=None, remove_classifier=remove_classifier)
        net.load_state_dict(state_dict, strict=False)
        print('Resume weights from {} to E_id, remove_classifier={}'.format(teacher_directory, remove_classifier))
    return net


def build_classifier(config, input_dim):
    return ClassBlock(input_dim, config['num_class'])
