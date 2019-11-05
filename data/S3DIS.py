import os
import numpy as np
from torch.utils.data import Dataset
from data import transforms as t


class S3DISDataset(Dataset):
    category = ['wall', 'floor', 'beam', 'chair', 'sofa', 'table',
                'door', 'window', 'bookcase', 'column', 'clutter', 'ceiling', 'board']
    CLIP_SIZE = None
    CLIP_BOUND = None
    LOCFEAT_IDX = 2
    ROTATION_AXIS = 'z'
    NUM_LABELS = 13
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 2  # 2cm
    IS_TEMPORAL = False
    data_aug_color_trans_ratio = 0.1
    data_aug_color_jitter_std = 0.05
    data_aug_hue_max = 0.5
    data_aug_saturation_max = 0.2
    transform = True

    def __init__(self, path, fold=0, data_type='train', voxel_size=0.05, transformations=False):
        super(Dataset, self).__init__()
        self.transformations = transformations
        self.voxel_size = voxel_size
        self.path = np.array(open(os.path.join(path, 'fold_{}'.format(fold), '{}.txt').format(data_type), 'r').readlines())
        self.length = len(self.path)
        if self.transformations:
            self.compose = t.Compose([
                t.RandomHorizontalFlip(S3DISDataset.ROTATION_AXIS, S3DISDataset.IS_TEMPORAL),
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(S3DISDataset.data_aug_color_trans_ratio),
                t.ChromaticJitter(S3DISDataset.data_aug_color_jitter_std),
                t.HueSaturationTranslation(S3DISDataset.data_aug_hue_max, S3DISDataset.data_aug_saturation_max),
            ])

    def __getitem__(self, index):
        folder_path = os.path.join(self.path[index][:-1], 'Annotations')
        files = os.listdir(folder_path)
        point = np.empty((1, 7))
        for file in files:
            if file != '.DS_Store' and file != 'Icon' and file != '':
                name = file.split('_')[0]
                if name == 'stairs':  # remove stairs, following SegCloud
                    continue
                cat = self.category.index(name)
                tmp = open(os.path.join(folder_path, file), 'r').readlines()
                tmp = np.array([i.split(' ') for i in tmp]).astype(np.float)
                tmp = np.append(tmp, np.array(cat).repeat(len(tmp)).reshape(-1, 1), axis=1)
                point = np.append(point, tmp, axis=0)
        point = np.delete(point, 0, axis=0)
        coords, feats, label = point[:, :3], point[:, 3:6]/256 - 0.5, point[:, -1]
        if self.transformations:
            coords, feats, label = self.compose(coords, feats, label)
        return (coords, feats, label.reshape(-1, 1))

    def __len__(self):
        return self.length


if __name__ == '__main__':
    data = S3DISDataset("/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version/split")
    data.__getitem__(1)
