import numpy as np
import os
from data.utils import save_point_cloud
from data.utils import read_plyfile
CATEGORY = ['wall', 'floor', 'beam', 'chair', 'sofa', 'table', 'door', 'window', 'bookcase', 'column', 'clutter', 'ceiling', 'board']


def S3DIS(data_path):
    fold = 0
    data_type = ['train', 'val']
    for type_ in data_type:
        path = np.array(open(os.path.join(data_path, "split", 'fold_{}'.format(fold), '{}.txt').format(type_), 'r').readlines())
        txt = open(os.path.join(data_path, 'ply', '{}.txt'.format(type_)), 'w')
        for i in range(len(path)):
            scene = path[i][:-1].split('/')[-2:]
            folder_path = os.path.join(path[i][:-1], 'Annotations')
            files = os.listdir(folder_path)
            point = np.empty((1, 7))
            for file in files:
                if file != '.DS_Store' and file != 'Icon' and file != '':
                    name = file.split('_')[0]
                    if name == 'stairs':  # remove stairs, following SegCloud
                        continue
                    cat = CATEGORY.index(name)
                    tmp = open(os.path.join(folder_path, file), 'r').readlines()
                    tmp = np.array([i.split(' ') for i in tmp]).astype(np.float)
                    tmp = np.append(tmp, np.array(cat).repeat(len(tmp)).reshape(-1, 1), axis=1)
                    point = np.append(point, tmp, axis=0)
            point = np.delete(point, 0, axis=0)
            save_path = os.path.join(data_path, 'ply', type_, scene[0] + '_' + scene[1]+'.ply')
            txt.write(save_path + '\n')
            save_point_cloud(point, save_path, with_label=True)
        txt.close()


def shapenet(path):
    data_type = ['train', 'val']
    for type_ in data_type:
        data_path = os.path.join(path, "{}_data_ply".format(type_))
        label_path = os.path.join(path, "{}_label".format(type_))
        data_category = os.listdir(data_path)
        label_category = os.listdir(label_path)
        assert len(data_category) == len(label_category)

        ply_path = os.path.join(path, "{}_ply".format(type_))
        if not os.path.exists(ply_path):
            os.mkdir(ply_path)

        for cat in data_category:
            data_files = os.listdir(os.path.join(data_path, cat))
            label_files = os.listdir(os.path.join(label_path, cat))
            data_files.sort()
            label_files.sort()
            assert len(data_files) == len(label_files)

            for ith in range(len(data_files)):
                data = read_plyfile(os.path.join(data_path, cat, data_files[ith]))

                label_file = open(os.path.join(label_path, cat, label_files[ith]), 'r')
                label = [int(i.split('\n')[0]) for i in label_file.readlines()]
                label_file.close()

                assert len(data) == len(label)
                if len(data) != len(label):
                    print(os.path.join(data_path, cat, data_files[ith]))
                point = np.concatenate([np.array(data).astype(np.float), np.array(label).astype(np.int).reshape(-1, 1)], axis=1)
                save_path = os.path.join(ply_path, cat + '_' + data_files[ith].split('.')[0]) + ".ply"
                save_point_cloud(point, save_path, with_label=True)


if __name__ == '__main__':
    shapenet("/media/gaoqiyu/File/data/shapenet_official_labeled")
