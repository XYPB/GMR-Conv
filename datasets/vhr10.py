import os
import json
import torch
import numpy as np
from glob import glob
from PIL import Image
import random

class VHR10(torch.utils.data.Dataset):

    def __init__(self, root, train=True, split_file=None, transform=None):
        assert os.path.exists(root)
        img_path = sorted(glob(os.path.join(root, '*.png')))
        self.path_label = [(p, int(p.split('/')[-1].split('_')[-1].replace('.png', ''))-1) for p in img_path]
        self.train = train
        self.transform = transform
        
        self.split_file = split_file
        if split_file == None or not os.path.exists(split_file):
            self.__make_split__()
        with open(self.split_file, 'r') as fp:
            split = json.load(fp)

        if train:
            self.data = split['train']
        else:
            self.data = split['test']


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        assert os.path.exists(img_path)

        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __make_split__(self, train_cnt_per_class=100, seed=0):
        split_dest = './data/NWPU_VHR-10_dataset/VHR_split.json'
        random.seed(seed)
        
        data = [[] for _ in range(10)]
        for p, label in self.path_label:
            data[label-1].append((p, label))

        train_set = []
        test_set = []
        for i in range(10):
            random.shuffle(data[i])
            train_set += data[i][:train_cnt_per_class]
            test_set += data[i][train_cnt_per_class:]

        with open(split_dest, 'w') as fp:
            json.dump({'train': train_set, 'test': test_set}, fp)
        self.split_file = split_dest
