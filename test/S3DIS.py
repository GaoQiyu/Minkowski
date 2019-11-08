import os
import numpy as np
import open3d as o3d
import torch
import json
import MinkowskiEngine as ME
from data.dataset import Voxelizer
from data.utils import read_plyfile
import model.res16unet as ResUNet
from model.evaluator import Evaluator
CLASS_LABELS = ('wall', 'floor', 'beam', 'chair', 'sofa', 'table',
                'door', 'window', 'bookcase', 'column', 'clutter', 'ceiling', 'board')
VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

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
    13: (0., 9., 255.)
}


def data_preprocess(data_dict, voxel_size):
    coords = data_dict[:, :3]
    feats = data_dict[:, 3:6]/256-0.5
    label = data_dict[:, 6]

    voxelizer = Voxelizer(
        voxel_size=voxel_size,
        clip_bound=None,
        use_augmentation=False,
        scale_augmentation_bound=(1, 1),
        rotation_augmentation_bound=(0, 0, 0),
        translation_augmentation_ratio_bound=((-0, 0), (-0, 0), (-0.0, 0.0)),
        ignore_label=255)
    coords, feats, labels, _ = voxelizer.voxelize(coords, feats, label)
    coords = np.concatenate([coords, np.zeros(shape=(coords.shape[0], 1))], axis=1)
    points = ME.SparseTensor(torch.from_numpy(feats), torch.from_numpy(coords).int())

    if config["use_cuda"]:
        points = points.to(device)
    return points, coords, feats, labels.astype(np.long)


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath("../"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()
    voxel_size = config["val_voxel_size"]
    num_class = config["class"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    model = ResUNet.Res16UNet34C(3, num_class).to(device)
    evaluator = Evaluator(num_class)
    model_dict = torch.load(os.path.join(config['resume_path'], 'parameters.pth'))["model"]
    model.load_state_dict(model_dict)
    model.eval()

    test_data = read_plyfile("/home/gaoqiyu/文档/Stanford3dDataset_v1.2_Aligned_Version/ply/val/Area_5_office_23.ply")
    sinput, coords, feats, labels = data_preprocess(test_data, voxel_size)

    soutput = model(sinput.to(device))

    # Feed-forward pass and get the prediction
    _, pred = soutput.F.max(1)
    pred = pred.cpu().numpy()

    evaluator.generate(pred, labels)
    print("mIOU:  ", evaluator.mIOU())
    # Map color
    colors_pred = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])
    # Create a point cloud file
    pred_pcd = o3d.geometry.PointCloud()
    coordinates = soutput.C.numpy()[:, :3]  # last column is the batch index
    pred_pcd.points = o3d.utility.Vector3dVector(coordinates * voxel_size)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors_pred / 255)

    colors_ground = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in labels])
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(coordinates * voxel_size + np.array([0, 10, 0]))
    ground_pcd.colors = o3d.utility.Vector3dVector(colors_ground / 255)

    o3d.visualization.draw_geometries([pred_pcd, ground_pcd])
