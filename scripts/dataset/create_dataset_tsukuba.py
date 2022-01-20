# General Data Science
# Miscellaneous
import glob
import mmcv
import numpy as np
import os
import os.path as osp
import pandas as pd
from pathlib import Path
from PIL import Image
from shutil import copyfile, rmtree

img_path = Path("/home/digital/sgk/data/tsukuba/working/imgs")
labels_path = Path("/home/digital/sgk/data/tsukuba/working/labels")
# os.mkdir(img_path)
# os.mkdir(labels_path)

data_root = '/home/digital/sgk/data/tsukuba/imgs'
working_root = '/home/digital/sgk/data/tsukuba/working'
img_dir = 'imgs'
ann_dir = 'labels'
rgb_filename = 'image[0-9][0-9][0-9][0-9][0-9][0-9].jpg'
label_filename = 'image[0-9][0-9][0-9][0-9][0-9][0-9].png'

rgb_l = glob.glob(data_root + '/*/*/' + rgb_filename)
label_l = glob.glob(data_root + '/*/*/label/' + label_filename)


# print(rgb_l)
# print(label_l)

all_count = 0
label_count = 0
for p in rgb_l: 
    if os.path.isfile(p):
        all_count += 1
        dir = "/".join(p.split('/')[0:-1])
        filename = p.split('/')[-1].split('.')[0]
        if p.split('/')[-1].split('.')[1] != 'png':
            img = Image.open(p)
            img.save(dir + '/' + filename + '.png', "PNG", quality=100)
        # print("dir", dir)
        # print("filename", filename)
        for pp in label_l:
            filename2 = pp.split('/')[-1].split('.')[0]
            dir2 = "/".join(pp.split('/')[0:-2])
            # print("dir2", dir2)
            # print("filename2", filename2)
            if(dir == dir2 and filename == filename2):
                label_count += 1
                new_filename = 'image{:0=4}.png'.format(label_count)
                # print("label: ", labels_path/filename2)
                copyfile(dir +  '/' + filename + '.png', img_path/new_filename)
                copyfile(pp, labels_path/new_filename)
                copyfile("/".join(pp.split('/')[0:-1]) + '/' + filename + '_pseudo.png', labels_path/'pseudo'/new_filename)
                break
print(all_count)
print(label_count)

