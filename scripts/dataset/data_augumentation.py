import glob
import matplotlib.pyplot as plt
import mmcv
import os.path as osp
import torch
from PIL import Image
from shutil import copyfile, rmtree
from torchvision import datasets, transforms

working_root="/home/digital/sgk/data/tsukuba/working/" 
dataset_root="/home/digital/sgk/data/tsukuba/working/data/"
img_dir='imgs/'
label_dir='labels/'

image_filename = 'image[0-9][0-9][0-9][0-9].png'
label_filename = 'image[0-9][0-9][0-9][0-9]_label.png'

img_list = glob.glob(dataset_root + image_filename)
label_list = glob.glob(dataset_root + label_filename)

data_transform = transforms.RandomOrder([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomResizedCrop(769, scale=(0.5, 1.0), ratio=(4 / 5, 5 / 4)),
    # transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2),
])

count = 0
start = 135
count = start
for img_str in img_list:
    # dir = "/".join(p.split('/')[0:-2])
    filename = img_str.split('/')[-1]
    ann_filename = filename.split('.')[0] + '_label.png'
    # img = Image.open(img_str)
    # img_augumented = data_transform(img)
    img_augumented.save( working_root + img_dir + 'image{:0=4}.png'.format(count), quality=95)
    copyfile(dataset_root + ann_filename, working_root + label_dir + 'image{:0=4}.png'.format(count))
    copyfile(working_root + label_dir + 'pseudo/' + filename, working_root + label_dir + 'pseudo/' + 'image{:0=4}.png'.format(count))
    count += 1

split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(working_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(working_root, label_dir), suffix='.png')]
with open(osp.join(working_root, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(working_root, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line + '\n' for line in filename_list[train_length:])
