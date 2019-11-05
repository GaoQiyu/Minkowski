import numpy as np
import torch
from torch.utils.data import DataLoader
from data.S3DIS import S3DISDataset
from torchvision.transforms import Compose as VisionCompose
from scipy.linalg import expm, norm


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
    feats_batch = np.vstack(feats)
    label_batch = np.vstack(labels)

    for batch_id in range(eff_num_batch):
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], :3] = torch.from_numpy(coords[batch_id])
        coords_batch[curr_ptr:curr_ptr + lens[batch_id], 3] = batch_id
        curr_ptr += len(coords[batch_id])
    # Concatenate all lists

    feats_batch = torch.from_numpy(np.vstack(feats)).float()
    label_batch = torch.from_numpy(np.vstack(labels)).squeeze().long()
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': label_batch,
    }


def dataloader(batch_size, path, fold=0, data_type='train', voxel_size=0.05, shuffle=False, drop_last=False, transformations=False):
    data = S3DISDataset(path, fold, data_type, voxel_size, transformations=transformations)
    dataloader = DataLoader(data, batch_size, shuffle, drop_last=drop_last, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    data = dataloader(1, "/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version/split")
    for i in data:
        print('s')