import os
import scipy.io
import torch
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from utils.distance import normalize
from config import SINGLE_SHOT_DATASETS


######################################################################
# extract features
# --------------------


def flip(img):
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(inference_api, loader, use_flip=True, use_norm=True, verbose=False):
    n_samples = len(loader.dataset)
    features = None
    count = 0
    for one_batch in loader:
        img = one_batch[0]
        n, _, _, _ = img.size()
        img = img.cuda()
        ff = inference_api(img)
        if use_flip:
            ff += inference_api(flip(img))
            ff /= 2.0
        if use_norm:
            ff = normalize(ff)
        if features is None:
            n_channels = ff.size(1)
            features = torch.zeros(n_samples, n_channels).cuda()
        features[count:(count+n)] = ff
        count += n
        if verbose:
            print(count, '/', n_samples)
    return features


######################################################################
# compute_CMC_and_mAP
# ------------------------
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.zeros(len(index), dtype=torch.int)
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def compute_CMC_and_mAP_multi_shot(index, ql, qc, gl, gc):
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())
    # compute cmc and map
    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc


def compute_CMC_and_mAP_single_shot(order, ql, qc, gl, gc, N=100):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed N times (default: N=100).
    """
    # good matches, remove gallery samples that have the same pid and camid with query
    matches = np.asarray(gl[order] == ql, dtype=np.int32)
    keep = np.invert((gl[order] == ql) & (gc[order] == qc))
    matches = matches[keep]  # binary vector, positions with value 1 are correct matches
    if not np.any(matches):
        assert 0, 'This condition is true when query identity does not appear in gallery.'

    # gl dict
    kept_gl = gl[order][keep]
    gl_dict = defaultdict(list)
    for idx, pid in enumerate(kept_gl):
        gl_dict[pid].append(idx)

    # compute cmc curve
    cmc, ap = 0., 0.
    for repeat_idx in range(N):
        mask = np.zeros(len(matches), dtype=np.bool)
        for _, idxs in gl_dict.items():
            # randomly sample one image for each gallery person
            rnd_idx = np.random.choice(idxs)
            mask[rnd_idx] = True
        masked_orig_cmc = matches[mask]
        _cmc = masked_orig_cmc.cumsum()
        _cmc[_cmc > 1] = 1
        cmc += _cmc.astype(np.float32)
        # compute AP
        num_rel = masked_orig_cmc.sum()
        tmp_cmc = masked_orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * masked_orig_cmc
        ap += tmp_cmc.sum() / num_rel
    cmc /= N
    ap /= N
    cmc = torch.tensor(cmc)

    return ap, cmc


######################################################################
# Tester
# -----------
class Tester(object):

    re_ranking_method = ('k_reciprocal', 'ecn')

    def __init__(self, config, dataset, gallery_folder='gallery', query_folder='query'):

        test_loaders, gallery_cam, gallery_label, query_cam, query_label = \
            dataset.test_loader(gallery_folder, query_folder)
        self.gallery_loader, self.query_loader = test_loaders['gallery'], test_loaders['query']
        self.gallery_cam, self.query_cam = np.array(gallery_cam), np.array(query_cam)
        self.gallery_label, self.query_label = np.array(gallery_label), np.array(query_label)
        self.gallery_num, self.query_num = len(gallery_label), len(query_label)

        self.dataset = config['dataset']
        if self.dataset in SINGLE_SHOT_DATASETS or config['eval_mode'] == 'single_shot':
            self.compute_cmc_map = partial(compute_CMC_and_mAP_single_shot, N=100)
            self.cmc_length = len(set(gallery_label))
            self.eval_mode = 'single_shot'
        else:
            self.compute_cmc_map = compute_CMC_and_mAP_multi_shot
            self.cmc_length = self.gallery_num
            self.eval_mode = 'multi_shot'
        print('eval mode: ', self.eval_mode)

    def _extract(self, model, verbose=False):
        with torch.no_grad():
            gallery_feature = extract_features(model, self.gallery_loader, verbose=verbose)
            query_feature = extract_features(model, self.query_loader, verbose=verbose)
            return gallery_feature, query_feature

    def _save(self, gallery_feature, query_feature, save_path):
        result = {
            'gallery_f':   gallery_feature.cpu().numpy(),
            'query_f':   query_feature.cpu().numpy(),
            'gallery_label':   self.gallery_label,
            'gallery_cam':   self.gallery_cam,
            'query_label':   self.query_label,
            'query_cam':   self.query_cam,
        }
        if save_path.endswith('.mat'):
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'results.mat')
        scipy.io.savemat(save_path, result)

    def _eval(self, gallery_feature, query_feature, fast, rerank, verbose):
        cmc, ap = torch.zeros(self.cmc_length), 0.0
        use_rerank = rerank in self.re_ranking_method
        with torch.no_grad():
            if use_rerank:
                with torch.no_grad():
                    q_g_dist = torch.matmul(query_feature, gallery_feature.t())
                    q_q_dist = torch.matmul(query_feature, query_feature.t())
                    g_g_dist = torch.matmul(gallery_feature, gallery_feature.t())
                q_g_dist = q_g_dist.detach().cpu().numpy()
                q_q_dist = q_q_dist.detach().cpu().numpy()
                g_g_dist = g_g_dist.detach().cpu().numpy()
                # re ranking
                from .ecn import re_ranking_ecn
                from .k_reciprocal import re_ranking_k_reciprocal
                if rerank == 'ecn':
                    print('using re_ranking: ecn...')
                    q_g_dist = re_ranking_ecn(q_g_dist, q_q_dist, g_g_dist)
                elif rerank == 'k_reciprocal':
                    print('using re_ranking: k_reciprocal...')
                    q_g_dist = re_ranking_k_reciprocal(q_g_dist, q_q_dist, g_g_dist)
                else:
                    assert 0, 'invalid re_ranking method: {}'.format(rerank)
            elif fast:
                q_g_dist = 2. - 2 * torch.mm(query_feature, gallery_feature.t()).cpu().numpy()
            for i in range(self.query_num):
                if verbose: print(i, '/', self.query_num)
                if fast or use_rerank:
                    distances = q_g_dist[i]
                else:
                    distances = torch.mm(gallery_feature, query_feature[i].view(-1, 1)).squeeze(1).cpu().numpy()
                    distances = 2. - 2 * distances
                order = np.argsort(distances)  # from small to large
                ap_tmp, cmc_tmp = self.compute_cmc_map(order, self.query_label[i], self.query_cam[i],
                                                       self.gallery_label,  self.gallery_cam)
                if cmc_tmp[0] == -1:
                    continue
                cmc += cmc_tmp.float()
                ap += ap_tmp
        cmc, ap = cmc/self.query_num, ap/self.query_num
        rank1, rank5, rank10, rank20 = cmc[0].item(), cmc[4].item(), cmc[9].item(), cmc[19].item()
        print('Rank@1=%.1f  Rank@5=%.1f  Rank@10=%.1f  Rank@20=%.1f  mAP=%.1f' % (
            rank1*100, rank5*100, rank10*100, rank20*100, ap*100))
        return rank1, rank5, rank10, rank20, ap

    def _eval_distmat(self, q_g_dist, reverse, verbose):
        cmc, ap = torch.zeros(self.cmc_length), 0.0
        with torch.no_grad():
            for i in tqdm(range(self.query_num)):
                if verbose: print(i, '/', self.query_num)
                score = q_g_dist[i]
                order = np.argsort(score)
                if reverse:
                    order = order[::-1]  # from small to large
                ap_tmp, cmc_tmp = self.compute_cmc_map(order, self.query_label[i], self.query_cam[i],
                                                       self.gallery_label,  self.gallery_cam)
                if cmc_tmp[0] == -1:
                    continue
                cmc += cmc_tmp.float()
                ap += ap_tmp
        cmc, ap = cmc/self.query_num, ap/self.query_num
        rank1, rank5, rank10, rank20 = cmc[0].item(), cmc[4].item(), cmc[9].item(), cmc[19].item()
        print('Rank@1=%.1f  Rank@5=%.1f  Rank@10=%.1f  Rank@20=%.1f  mAP=%.1f' % (
            rank1*100, rank5*100, rank10*100, rank20*100, ap*100))
        return rank1, rank5, rank10, rank20, ap

    def reid_test(self, model, save_path=None, fast=False, rerank='none', verbose=False):
        gallery_feature, query_feature = self._extract(model, verbose)
        if save_path is not None:
            self._save(gallery_feature, query_feature, save_path)
        return self._eval(gallery_feature, query_feature, fast, rerank, verbose)

    def reid_test_from_feat(self, resume_path, fast=False, rerank='none', verbose=False):
        if not resume_path.endswith('.mat'):
            resume_path = os.path.join(resume_path, 'results.mat')
        result = scipy.io.loadmat(resume_path)
        if verbose:
            print('Resume features from {}'.format(resume_path))
        query_feature = torch.tensor(result['query_f'], dtype=torch.float).cuda()
        gallery_feature = torch.tensor(result['gallery_f'], dtype=torch.float).cuda()
        self.query_cam = result['query_cam'][0]
        self.gallery_cam = result['gallery_cam'][0]
        self.query_label = result['query_label'][0]
        self.gallery_label = result['gallery_label'][0]
        return self._eval(gallery_feature, query_feature, fast, rerank, verbose)

    def reid_test_from_distmat(self, resume_path, distmat_name='distmat', reverse=False, verbose=False):
        if not resume_path.endswith('.mat'):
            resume_path = os.path.join(resume_path, 'results.mat')
        result = scipy.io.loadmat(resume_path)
        if verbose:
            print('Resume features from {}'.format(resume_path))
        distmat = result[distmat_name]
        self.query_cam = result['query_cam'][0]
        self.gallery_cam = result['gallery_cam'][0]
        self.query_label = result['query_label'][0]
        self.gallery_label = result['gallery_label'][0]
        return self._eval_distmat(distmat, reverse, verbose)
