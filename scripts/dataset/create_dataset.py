# General Data Science
# Miscellaneous
import os
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np
import pandas as pd

img_path = Path("/home/digital/sgk/deep/sem-seg/carla-semantic-segmentation/working/img")
labels_path = Path("/home/digital/sgk/deep/sem-seg/carla-semantic-segmentation/working/labels")
# os.mkdir(img_path)
# os.mkdir(labels_path)

for dirname, _, filenames in os.walk('/home/digital/sgk/data/carla/carla-semantic-segmnetation'):
    if dirname.endswith("CameraRGB") and any(dataset in dirname for dataset in ["dataa", "datab", "datac", "datad", "datae"]):
        for filename in filenames:
            copyfile(os.path.join(dirname, filename), img_path/filename)
    elif dirname.endswith("CameraSeg") and any(dataset in dirname for dataset in ["dataa", "datab", "datac", "datad", "datae"]):
        for filename in filenames:
            copyfile(os.path.join(dirname, filename), labels_path/filename)

