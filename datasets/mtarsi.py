import os
import json
import torch
import numpy as np
from PIL import Image

class MTARSI(torch.utils.data.Dataset):

    def __init__(self, train=True, split_file=None, transform=None):
        self.train = train
        self.transform = transform
        
        self.split_file = split_file
        if split_file == None or not os.path.exists(split_file):
            raise Exception("Split file not found")
        with open(self.split_file, 'r') as fp:
            split = json.load(fp)

        if train:
            self.data = split['train']
        else:
            self.data = split['test']

        text_label = sorted(list(set([p.split('/')[-2] for p in self.data])))
        assert len(text_label) == 20

        self.label2idx = {label:idx for idx, label in enumerate(text_label)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        assert os.path.exists(img_path)
        text_label = img_path.split('/')[-2]
        label = self.label2idx[text_label]

        img = Image.open(img_path)
        if np.array(img).shape[-1] == 4:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

