import numpy as np
from torch.utils.data import DataLoader
from data.S3DIS import S3DISDataset
from torchvision.transforms import Compose as VisionCompose
from scipy.linalg import expm, norm
import torch


class Compose(VisionCompose):

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class RandomRotation:

    def __init__(self, axis=None, max_theta=180):
        self.axis = axis
        self.max_theta = max_theta

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, feats):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(
            axis,
            (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        return coords @ R, feats


class RandomScale:

    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords, feats):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s, feats


class RandomShear:

    def __call__(self, coords, feats):
        T = np.eye(3) + 0.1 * np.random.randn(3, 3)
        return coords @ T, feats


class RandomTranslation:

    def __call__(self, coords, feats):
        trans = 0.05 * np.random.randn(1, 3)
        return coords + trans, feats


def collate_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, labels = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    lens = [len(c) for c in coords]
    curr_ptr = 0
    num_tot_pts = sum(lens)
    coords_batch = torch.zeros(num_tot_pts, 4)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()
    label_batch = torch.from_numpy(np.vstack(labels)).squeeze().long()

    for batch_id in range(eff_num_batch):
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], :3] = torch.from_numpy(coords[batch_id])
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], 3] = batch_id
        curr_ptr += len(coords[batch_id])

    # Concatenate all lists
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': label_batch,
    }


def dataloader(batch_size, path, fold=0, data_type='train', voxel_size=0.05, transform=False, shuffle=False, drop_last=False):
    transformations = []
    if transform:
        transformations.append(RandomRotation())
        transformations.append(RandomTranslation())
        transformations.append(RandomScale(0.8, 1.2))
        transformations.append(RandomShear())
    data = S3DISDataset(path, fold, data_type, voxel_size, Compose(transformations))
    # data = S3DISDataset(path, fold, data_type, voxel_size)
    dataloader = DataLoader(data, batch_size, shuffle, drop_last=drop_last, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    data = dataloader(12, "/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version/split", transform=True)
    for i in data:
        print('s')