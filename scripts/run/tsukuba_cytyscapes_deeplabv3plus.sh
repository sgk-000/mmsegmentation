#!/bin/bash

/home/digital/mmsegmentation/tools/dist_train.sh ~/mmsegmentation/configs/deeplabv3plus/tsukuba/deeplabv3plus_r50-d8_769x769_40k_cityscapes_config.py 1 --load-from ~/mmsegmentation/checkpoints/deeplabv3plus/deeplabv3plus_r50-d8_769x769_40k_cityscapes_20200606_114143-1dcb0e3c.pth
