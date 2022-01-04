# General Data Science
# Miscellaneous
import os
import os.path as osp
from pathlib import Path
from shutil import copyfile, rmtree

import mmcv
import numpy as np
import pandas as pd

data_root = "/home/digital/sgk/deep/sem-seg/carla-semantic-segmentation/working/mine"
img_path = Path("/home/digital/sgk/deep/sem-seg/carla-semantic-segmentation/working/mine/imgs")
labels_path = Path("/home/digital/sgk/deep/sem-seg/carla-semantic-segmentation/working/mine/labels")
img_dir = "img"
ann_dir = "labels"
split_dir = 'splits'

# os.mkdir(img_path)
# os.mkdir(labels_path)

for dirname, _, filenames in os.walk('/home/digital/sgk/data/carla/carla-semantic-segmnetation'):
    if dirname.endswith("imgs") and any(dataset in dirname for dataset in ["2021-12-01-17-20-21", "2021-12-01-17-28-50", "2021-12-28-09-48-12"]):
        for filename in filenames:
            copyfile(os.path.join(dirname, filename), img_path/filename)
    elif dirname.endswith("labels") and any(dataset in dirname for dataset in ["2021-12-01-17-20-21", "2021-12-01-17-28-50", "2021-12-28-09-48-12"]):
        for filename in filenames:
            copyfile(os.path.join(dirname, filename), labels_path/filename)


split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.jpg')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line + '\n' for line in filename_list[train_length:])
