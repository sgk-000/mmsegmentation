import os.path as osp

from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

# convert dataset annotation to semantic segmentation map
data_root = "/home/aad13694zb/tsukuba/working"
img_dir = "imgs"
ann_dir = "labels"

classes = (
    "Unlabeled",
    "Road",
    "Sidewalk",
    "Building",
    "Pole",
    "Vegetation",
    "Terrain",
    "Sky",
    "Person",
    "Car",
    "Bycycle",
    "Motorcycle",
    "Traffic Light",
    "Traffic sign",
)

palette = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [153, 153, 153],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [0, 0, 142],
    [119, 11, 32],
    [0, 0, 230],
    [250, 170, 30]
    [220, 220, 0],
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



@DATASETS.register_module()
class TsukubaCityScapesDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, **kwargs):
        super(TsukubaCityScapesDataset, self).__init__(
          img_suffix='.png', 
          seg_map_suffix='.png', 
          **kwargs)
        # assert osp.exists(self.img_dir)


