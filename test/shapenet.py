import os
import numpy as np
import open3d as o3d
import torch
import json
import MinkowskiEngine as ME
import model.res16unet as ResUNet
from model.evaluator import Evaluator
from data.utils import read_plyfile
from data.dataset import Voxelizer
CLASS_LABELS = ('1', '2', '3', '4')
VALID_CLASS_IDS = [1, 2, 3, 4]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (227., 209., 212.),
    2: (143., 100., 21.),
    3: (242., 171., 39.),
    4: (100., 143., 156.)
}


def data_preprocess(data_dict, voxel_size):
    coords = data_dict[:, :3]
    feats = data_dict[:, 3:6]/256-0.5

    voxelizer = Voxelizer(
        voxel_size=voxel_size,
        clip_bound=None,
        use_augmentation=False,
        scale_augmentation_bound=(0.9, 1.1),
        rotation_augmentation_bound=((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)),
        translation_augmentation_ratio_bound=((-0, 0), (-0, 0), (-0.0, 0.0)),
        ignore_label=255)
    coords, feats, _, _ = voxelizer.voxelize(coords, feats, np.zeros(shape=(coords.shape[0])))
    coords = np.concatenate([coords, np.zeros(shape=(coords.shape[0], 1))], axis=1)
    points = ME.SparseTensor(torch.from_numpy(feats), torch.from_numpy(coords).int())

    if config["use_cuda"]:
        points = points.to(device)
    return points, coords, feats


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath("../"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()
    evaluator = Evaluator(4)
    voxel_size = config["val_voxel_size"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    model = ResUNet.Res16UNet34C(3, 4).to(device)
    model_dict = torch.load(os.path.join(config['resume_path'], 'parameters.pth'))["model"]
    model.load_state_dict(model_dict)
    model.eval()

    test_data = read_plyfile("/home/gaoqiyu/文档/shapenet/val/02691156_013517.ply")
    sinput, coords, feats = data_preprocess(test_data, voxel_size)

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

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(test_data[:, :3] + np.array([0, 0.5, 0]))
    ground_pcd.colors = o3d.utility.Vector3dVector(test_data[:, 3:6] / 255)

    o3d.visualization.draw_geometries([pred_pcd, ground_pcd])
