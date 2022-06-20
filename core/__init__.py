from .dil import DILTrainer
from .reid import ReIDTrainer, Augmenter


def build_dil_trainer(config):
    return DILTrainer(config)


def build_reid_trainer(config):
    return ReIDTrainer(config)
