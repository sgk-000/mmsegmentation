import numpy as np
import random
import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
from torchvision import transforms


class TsukubaDataset(data.Dataset):
    def __init__(self, dir_path, input_size, phase):
        super().__init__()
        
        self.dir_path = dir_path
        self.input_size = input_size
        
        self.image_paths = [str(p) for p in Path(self.dir_path).glob("imgs/*.png")]
        self.len = len(self.image_paths)

        if phase == "train":
            transform_ops.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(769, scale=(0.5, 1.0), ratio=(4 / 5, 5 / 4))
                # transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2),
            ])
        elif phase == "val":
            transform_ops.append(
                transforms.Resize(input_size)
            )
            
        transform_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transformer = transforms.Compose(transform_ops)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        p = self.image_paths[index]
        
        # 入力
        image = Image.open(p)
        image = image.resize(self.input_size)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

        img_filename = p.split('.')[0]
        label_str = self.dir_path + '/labels/' + img_filename
        label = Image.open(p)
        label = label.resize(self.input_size)
        label = np.array(label)
        label = np.transpose(label, (2, 0, 1))
        label = torch.from_numpy(label)
      
        return image, label

train_dataset = TsukubaDataset("/home/digital/sgk/data/tsukuba/working/", (769, 769))

image, label = train_dataset[0]
print(image.size(), label)  # torch.Size([3, 224, 224]) 1
