import os


split_fold = [
    ([1, 2, 3, 4, 6], [5]),
    ([1, 3, 5, 6], [2, 4]),
    ([2, 4, 5], [1, 3, 6])
]

data_path = "/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version"

for i, split in enumerate(split_fold):
    train, val = split
    train_txt = open(os.path.join(data_path, "split/fold_{}/train.txt".format(i)), 'w')
    for train_index in train:
        train_folder = os.path.join(data_path, "Area_{}".format(train_index))
        train_file = os.listdir(train_folder)
        for train_file_name in train_file:
            if train_file_name != '.DS_Store':
                train_path = os.path.join(data_path, "Area_{}".format(train_index), train_file_name)
                train_txt.write(train_path + '\n')
    train_txt.close()

    val_txt = open(os.path.join(data_path, "split/fold_{}/val.txt".format(i)), 'w')
    for val_index in val:
        val_folder = os.path.join(data_path, "Area_{}".format(val_index))
        val_file = os.listdir(val_folder)
        for val_file_name in val_file:
            if val_file_name != '.DS_Store':
                val_path = os.path.join(data_path, "Area_{}".format(val_index), val_file_name)
                val_txt.write(val_path + '\n')
    val_txt.close()

