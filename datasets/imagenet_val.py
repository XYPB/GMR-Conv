import os
import torch
from torchvision.datasets.folder import default_loader
from glob import glob
import pandas as pd

class ImageNetVal(torch.utils.data.Dataset):
    
    def __init__(
        self,
        root,
        transform=None,
        class_to_idx=None,
        loader=default_loader,
        class_map_path="./data/imagenet-object-localization-challenge/LOC_val_solution.csv"
    ):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.class_to_idx = class_to_idx
        self.classes = sorted(list(self.class_to_idx.keys())) # sorted alphabetically
        
        class_map = pd.read_csv(class_map_path)
        name2label = {row['ImageId']: row['PredictionString'].split(' ')[0] for _, row in class_map.iterrows()}
        img_paths = sorted(glob(os.path.join(self.root, "*.JPEG")))
        self.imgs = []
        for img_path in img_paths:
            image_id = os.path.basename(img_path).split('.')[0]
            label = self.class_to_idx[name2label[image_id]]
            self.imgs.append((img_path, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label