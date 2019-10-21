import os
import numpy as np
import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME


class S3DISDataset(Dataset):
    category = ['ceiling', 'floor', 'wall', 'beam', 'column', 'door', 'window', 'table', 'chair', 'sofa', 'bookcase',
                'board', 'clutter', 'stairs']

    def __init__(self, path, fold=0, data_type='train', voxel_size=0.05):
        super(Dataset, self).__init__()
        self.voxel_size = voxel_size
        self.path = np.array(open(os.path.join(path, 'fold_{}'.format(fold), '{}.txt').format(data_type), 'r').readlines())
        self.length = len(self.path)
        self.path = self.path[np.random.choice(np.arange(self.length), self.length)]

    def __getitem__(self, index):
        folder_path = os.path.join(self.path[index][:-1], 'Annotations')
        # folder_path = '/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations'
        files = os.listdir(folder_path)
        point = np.empty((1, 7))
        for file in files:
            if file != '.DS_Store' and file != 'Icon':
                # file = 'ceiling_1.txt'
                # print(os.path.join(folder_path, file))
                cat = self.category.index(file.split('_')[0])
                tmp = open(os.path.join(folder_path, file), 'r').readlines()
                # for tt, i in enumerate(tmp):
                #     print(tt)
                #     y = i[:-1].split(' ')
                tmp = np.array([i.split(' ') for i in tmp]).astype(np.float)
                tmp = np.append(tmp, np.array(cat).repeat(len(tmp)).reshape(-1, 1), axis=1)
                point = np.append(point, tmp, axis=0)
        point = np.delete(point, 0, axis=0)
        coords, feats, label = np.floor(point[:, :3]/self.voxel_size), point[:, 3:6]/256, point[:, -1]

        inds = ME.utils.sparse_quantize(coords)
        coordinates_, featrues_ = tuple([coords[inds]]), tuple([feats[inds]])
        coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

        # discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
        #     coords=coords,
        #     feats=feats,
        #     labels=label,
        #     hash_type='ravel',
        #     set_ignore_label_when_collision=False,
        #     quantization_size=self.voxel_size)
        # shuffle_index = np.random.choice(discrete_coords.shape[0], self.points_num)

        return [ME.SparseTensor(features, coords=coordinates), torch.from_numpy(label[inds])]
        # return [discrete_coords[shuffle_index], unique_feats[shuffle_index], unique_labels[shuffle_index]]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    data = S3DISDataset("/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version/split")
    data.__getitem__(1)