import torch
import torch.nn as nn
from .estimator import EstimatorCV


class ISDALoss(nn.Module):

    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, features, labels):
        self.estimator.update_CV(features.detach(), labels)

    def augment(self, features, y, fc, labels, ratio):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV_temp = self.estimator.CoVariance.detach()[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)
        aug_result = y + 0.5 * sigma2
        return aug_result

    def forward(self, features, y, labels, fc_model, ratio):
        # self.update(features, labels)
        isda_aug_y = self.augment(features, y, fc_model, labels, ratio)
        loss = self.cross_entropy(isda_aug_y, labels)
        return loss

# DEMO
# self.isda_criterion.update(features=f, target_x=l)
# self.loss_id_cls = self.isda_criterion(y, f, self.C.classifier[0], l, isda_ratio)
