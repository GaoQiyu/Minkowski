import os
import numpy as np
import open3d as o3d
import torch
import json
import MinkowskiEngine as ME
from data.S3DIS import S3DISDataset
from data.dataset import initialize_data_loader
import model.res16unet as ResUNet
CLASS_LABELS = ('wall', 'floor', 'beam', 'chair', 'sofa', 'table',
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


def data_preprocess(data_dict):
    coords = data_dict[0]
    feats = data_dict[1] / 256 - 0.5
    labels = data_dict[2]
    length = coords.shape[0]

    if length > config["point_num"]:
        inds = np.random.choice(np.arange(length), config["point_num"], replace=False)
        coords, feats, labels = coords[inds], feats[inds], labels[inds]

    points = ME.SparseTensor(feats, coords.int())

    if config["use_cuda"]:
        points, labels = points.to(device), labels.to(device)
    return points, coords, feats, labels.long()


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath("./"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()
    # voxel_size = config["val_voxel_size"]
    voxel_size = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    model = ResUNet.Res16UNet34C(3, 13).to(device)
    model_dict = torch.load(os.path.join(config['resume_path'], 'parameters.pth'))["model"]

    model.load_state_dict(model_dict)
    model.eval()

    val_data = initialize_data_loader(
        S3DISDataset,
        config,
        threads=1,
        phase='VAL',
        augment_data=False,
        shuffle=True,
        repeat=False,
        batch_size=1,
        limit_numpoints=False)
    data_iter = val_data.__iter__()
    data = data_iter.__next__()

    sinput, coords, feats, label = data_preprocess(data)

    soutput = model(sinput.to(device))

    # Feed-forward pass and get the prediction
    _, pred = soutput.F.max(1)
    pred = pred.cpu().numpy()

    # Map color
    colors_pred = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])
    # Create a point cloud file
    pred_pcd = o3d.geometry.PointCloud()
    coordinates = soutput.C.numpy()[:, :3]  # last column is the batch index
    pred_pcd.points = o3d.utility.Vector3dVector(coordinates * voxel_size)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors_pred / 255)

    colors_ground = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in label])
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(coordinates * voxel_size + np.array([0, 2, 0]))
    ground_pcd.colors = o3d.utility.Vector3dVector(colors_ground / 255)

    color_pcd = o3d.geometry.PointCloud()
    color_pcd.points = o3d.utility.Vector3dVector(data[0].numpy()[:, :3] * voxel_size + np.array([0, 2, 2]))
    color_pcd.colors = o3d.utility.Vector3dVector(data[1].numpy() / 255)

    o3d.visualization.draw_geometries([pred_pcd, ground_pcd, color_pcd])
