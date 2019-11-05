import os
import numpy as np
import open3d as o3d
import torch
import json
import MinkowskiEngine as ME
from data.S3DIS import S3DISDataset
import model.resunet as ResUNet

# CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
#                 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
#                 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
#                 'bathtub', 'otherfurniture')
# VALID_CLASS_IDS = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
# ]
# SCANNET_COLOR_MAP = {
#     0: (0., 0., 0.),
#     1: (174., 199., 232.),
#     2: (152., 223., 138.),
#     3: (31., 119., 180.),
#     4: (255., 187., 120.),
#     5: (188., 189., 34.),
#     6: (140., 86., 75.),
#     7: (255., 152., 150.),
#     8: (214., 39., 40.),
#     9: (197., 176., 213.),
#     10: (148., 103., 189.),
#     11: (196., 156., 148.),
#     12: (23., 190., 207.),
#     14: (247., 182., 210.),
#     15: (66., 188., 102.),
#     16: (219., 219., 141.),
#     17: (140., 57., 197.),
#     18: (202., 185., 52.),
#     19: (51., 176., 203.),
#     20: (200., 54., 131.),
#     21: (92., 193., 61.),
#     22: (78., 71., 183.),
#     23: (172., 114., 82.),
#     24: (255., 127., 14.),
#     25: (91., 163., 138.),
#     26: (153., 98., 156.),
#     27: (140., 153., 101.),
#     28: (158., 218., 229.),
#     29: (100., 125., 154.),
#     30: (178., 127., 135.),
#     32: (146., 111., 194.),
#     33: (44., 160., 44.),
#     34: (112., 128., 144.),
#     35: (96., 207., 209.),
#     36: (227., 119., 194.),
#     37: (213., 92., 176.),
#     38: (94., 106., 211.),
#     39: (82., 84., 163.),
#     40: (100., 85., 144.),
# }
CLASS_LABELS = ('wall', 'floor', 'stairs', 'beam', 'chair', 'sofa', 'table',
                'door', 'window', 'bookcase', 'column', 'clutter', 'ceiling', 'board')
VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (227., 209., 212.),
    2: (143., 100., 21.),
    3: (242., 171., 39.),
    4: (100., 143., 156.),
    5: (3., 3., 3.),
    6: (255., 255., 0.),
    7: (255., 0., 17.),
    8: (45., 204., 193.),
    9: (204., 45., 191.),
    10: (85., 45., 204.),
    11: (45., 204., 85.),
    12: (81., 94., 51.),
    13: (0., 9., 255.),
    14: (122., 126., 240.)
}


def load_file(file_name, voxel_size):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    feats = np.array(pcd.colors)

    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords)

    return quantized_coords[inds], feats[inds], pcd


def generate_input_sparse_tensor(file_name, voxel_size=0.05):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size)]
    coordinates_, featrues_, pcds = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    # Normalize features and create a sparse tensor
    return ME.SparseTensor(features - 0.5, coords=coordinates).to(device)


def data_preprocess(data_dict, voxel_size):
    coords = data_dict[0]
    coords = np.concatenate([coords, np.zeros(shape=(coords.shape[0], 1))], axis=1)
    feats = data_dict[1]
    labels = data_dict[2]

    coords[:, :3] = np.floor(coords[:, :3] / voxel_size)
    inds = ME.utils.sparse_quantize(coords[:, :3], return_index=True)
    coordinates, features, labels = coords[inds], torch.from_numpy(feats[inds]).float(), labels[inds]
    points = ME.SparseTensor(features, torch.from_numpy(coordinates).int())

    points, labels = points.to(torch.device(0)), torch.from_numpy(labels).cuda()
    return points, labels


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath("./"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    model = ResUNet.ResUNetIN14(3, 14).to(device)
    model_dict = torch.load(os.path.join(config['resume_path'], 'parameters.pth'))["model"]
    # model_dict = torch.load(os.path.join(config['resume_path'], 'weights.pth'))
    model.load_state_dict(model_dict)
    model.eval()

    val_data = S3DISDataset(config["data_path"], 0, 'val', config["voxel_size"])
    inds = np.random.choice(len(val_data), 1).item()
    print(val_data.path[inds])
    sinput, _ = data_preprocess(val_data[inds], voxel_size=0.02)

    soutput = model(sinput.to(device))

    # Feed-forward pass and get the prediction
    _, pred = soutput.F.max(1)
    pred = pred.cpu().numpy()

    # Map color
    colors = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])

    # Create a point cloud file
    pred_pcd = o3d.geometry.PointCloud()
    coordinates = soutput.C.numpy()[:, :3]  # last column is the batch index
    pred_pcd.points = o3d.utility.Vector3dVector(coordinates * 0.02)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)

    # # Move the original point cloud
    # pcd = o3d.io.read_point_cloud('/media/gaoqiyu/File/Pycharm/PointCloudSeg_Minkowski/resume/1.ply')
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) + np.array([0, 5, 0]))

    # Visualize the input point cloud and the prediction
    # o3d.visualization.draw_geometries([pcd, pred_pcd])

    o3d.visualization.draw_geometries([pred_pcd])
