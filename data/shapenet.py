import os
import numpy as np
from data.dataset import VoxelizationDataset, DatasetPhase


class ShapeNet(VoxelizationDataset):
    category = ['wall', 'floor', 'beam', 'chair', 'sofa', 'table',
                'door', 'window', 'bookcase', 'column', 'clutter', 'ceiling', 'board']
    CLIP_SIZE = None
    CLIP_BOUND = None
    LOCFEAT_IDX = 2
    ROTATION_AXIS = 'z'
    NUM_LABELS = 13
    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    CLIP_BOUND = None
    IGNORE_LABELS = []  # remove stairs, following SegCloud

    # Augmentation arguments
    ELASTIC_DISTORT_PARAMS = ((20, 100), (80, 320))
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0, 0), (-0, 0), (-0.0, 0.0))

    def __init__(self, config, prevoxel_transform=None, input_transform=None, target_transform=None, cache=False, augment_data=True, elastic_distortion=False, phase=DatasetPhase.Train):
        if phase == DatasetPhase.Train:
            data_root = os.path.join(config["data_path"], 'train')
        elif phase == DatasetPhase.Val:
            data_root = os.path.join(config["data_path"], 'val')
        VoxelizationDataset.__init__(
            self,
            os.listdir(data_root),
            data_root=data_root,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config["ignore_label"],
            return_transformation=config["return_transformation"],
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            config=config)



