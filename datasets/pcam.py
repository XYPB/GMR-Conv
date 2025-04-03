import os
import torch
import h5py
from PIL import Image

class PatchCamelyon(torch.utils.data.Dataset):
    def __init__(self, train_h5, test_h5, transform=None):
        assert os.path.exists(train_h5) and os.path.exists(test_h5)
        self.train_h5 = train_h5
        self.test_h5 = test_h5
        self.transform = transform

        self.images = h5py.File(train_h5, 'r')['x']
        self.labels = h5py.File(test_h5, 'r')['y']

        print(f'Loaded {len(self.images)} images')
        print(f'Loaded {len(self.labels)} labels')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label.squeeze()