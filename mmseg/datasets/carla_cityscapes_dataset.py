import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from PIL import Image

from .custom import CustomDataset

# convert dataset annotation to semantic segmentation map
data_root = "/home/aad13694zb/carla-semantic-segmentation/working"
img_dir = "img"
ann_dir = "labels"

classes = (
    "Unlabeled",
    "Building",
    "Fence",
    "Other",
    "Pedestrian",
    "Pole",
    "Road line",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Vehicle",
    "Wall",
    "Traffic sign",
    "Sky",
    "Traffic Light",
)

palette = [
    [0, 0, 0],
    [70, 70, 70],
    [100, 40, 40],
    [55, 90, 80],
    [220, 20, 60],
    [153, 153, 153],
    [157, 234, 50],
    [128, 64, 128],
    [244, 35, 232],
    [107, 142, 35],
    [0, 0, 142],
    [102, 102, 156],
    [220, 220, 0],
    [70, 130, 180],
    [250, 170, 30]
]

# split_dir = 'splits'
# mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
# filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
#     osp.join(data_root, ann_dir), suffix='.png')]
# with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
#   # select first 4/5 as train set
#   train_length = int(len(filename_list)*4/5)
#   f.writelines(line + '\n' for line in filename_list[:train_length])
# with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
#   # select last 1/5 as train set
#   f.writelines(line + '\n' for line in filename_list[train_length:])

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CarlaCityScapesDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, **kwargs):
        super(CarlaCityScapesDataset, self).__init__(
          img_suffix='.png', 
          seg_map_suffix='.png', 
          **kwargs)
        # assert osp.exists(self.img_dir)


