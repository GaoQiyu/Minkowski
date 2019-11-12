import numpy as np
import os
from data.utils import save_point_cloud
from data.utils import read_plyfile
CATEGORY = ['wall', 'floor', 'beam', 'chair', 'sofa', 'table', 'door', 'window', 'bookcase', 'column', 'ceiling', 'board']


def S3DIS(data_path):
    fold = 2
    data_type = ['train', 'val']
    for type_ in data_type:
        path = np.array(open(os.path.join(data_path, "split", 'fold_{}'.format(fold), '{}.txt').format(type_), 'r').readlines())
        for i in range(len(path)):
            scene = path[i][:-1].split('/')[-2:]
            folder_path = os.path.join(path[i][:-1], 'Annotations')
            files = os.listdir(folder_path)
            point = np.empty((1, 7))
            for file in files:
                if file != '.DS_Store' and file != 'Icon' and file != '':
                    name = file.split('_')[0]
                    if name == 'stairs' or name == 'clutter':
                        # remove stairs, following SegCloud
                        # remove clutter, following paper
                        continue
                    cat = CATEGORY.index(name)
                    tmp = open(os.path.join(folder_path, file), 'r').readlines()
                    tmp = np.array([i.split(' ') for i in tmp]).astype(np.float)
                    tmp = np.append(tmp, np.array(cat).repeat(len(tmp)).reshape(-1, 1), axis=1)
                    point = np.append(point, tmp, axis=0)
            point = np.delete(point, 0, axis=0)
            save_path = os.path.join(data_path, 'ply_{}'.format(fold), type_, scene[0] + '_' + scene[1]+'.ply')
            save_point_cloud(point, save_path, with_label=True)


if __name__ == '__main__':
    S3DIS("/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version")
