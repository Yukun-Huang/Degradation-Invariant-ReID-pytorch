import torch
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm


class MemoryBank(object):
    @staticmethod
    def flip(img):
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
        inv_idx = inv_idx.to(img.device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def __init__(self, *args):
        if len(args) == 2:
            loader, encoder = args
            deg_features = self.extract_features(loader, encoder)
        else:
            assert len(args) == 1
            deg_features = args[0]
        self.bank = TensorDataset(deg_features)

    def extract_features(self, loader, encoder):
        features, cnt = None, 0
        with torch.no_grad():
            print('Building degradation memory bank...')
            for one_batch in tqdm(loader):
                if len(one_batch) == 2 or len(one_batch) == 3:
                    img = one_batch[0]
                elif len(one_batch) == 5:
                    # img1, img2, _, _, _ = one_batch
                    _, img, _, _, _ = one_batch
                else:
                    assert 0
                n, _, _, _ = img.size()
                img = img.cuda()
                ff = (encoder(img) + encoder(self.flip(img))) / 2
                if features is None:
                    features = torch.zeros(len(loader.dataset), ff.size(1)).cuda()
                features[cnt:(cnt+n)] = ff
                cnt += n
        return features

    def get_a_batch(self, batch_size):
        idx = torch.randint(high=len(self.bank), size=(batch_size,), dtype=torch.int64).tolist()
        return self.bank[idx][0]

    def __len__(self):
        return len(self.bank)
