import os
import torch
from PIL import Image
from glob import glob
from collections import Counter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class NCT_CRC(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        assert os.path.exists(root)

        label2idx = {'ADI': 0, 'BACK': 1,  'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6,  'STR': 7, 'TUM': 8}
        img_path = sorted(glob(os.path.join(root, '*/*.tif')))
        self.path_label = [(p, label2idx[p.split('/')[-2]]) for p in img_path]
        self.transform = transform

        print(f'Loaded {len(self.path_label)} images')
        print(Counter([label for _, label in self.path_label]))

    def __len__(self):
        return len(self.path_label)
    
    def __getitem__(self, idx):
        img_path, label = self.path_label[idx]
        assert os.path.exists(img_path)

        # uint8 images
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label