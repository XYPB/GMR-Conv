import os
from collections import Counter
import math
import torch
import torch.utils
import torch.utils.data
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from tqdm import tqdm


def point_cloud_to_voxels(data, resolution=32):
    """
    Convert data.pos (N x 3) into a 3D occupancy grid (resolution^3).
    Assumes positions are in [-1,1]^3.
    """
    pos = data.pos  # shape [N, 3]
    
    # Initialize empty occupancy grid
    # We'll store it as (D, H, W) in a PyTorch tensor
    voxels = torch.zeros((resolution, resolution, resolution), dtype=torch.float)
    
    for x, y, z in pos:
        # Map from [-1,1] to [0, resolution-1]
        i = math.floor(((x.item() + 1) / 2) * resolution)
        j = math.floor(((y.item() + 1) / 2) * resolution)
        k = math.floor(((z.item() + 1) / 2) * resolution)
        
        # Clamp in case any rounding goes out of bounds
        i = max(0, min(i, resolution - 1))
        j = max(0, min(j, resolution - 1))
        k = max(0, min(k, resolution - 1))
        
        voxels[i, j, k] = 1.0

    # Attach the voxel grid to the data object
    data.voxels = voxels.unsqueeze(0)  # [1, D, H, W], if you prefer a "channel" dimension
    return data


class MyModelNet(torch.utils.data.Dataset):
    
    def __init__(
        self,
        name="10",
        root="./data",
        train=True,
        transform=None,
    ):
        pre_process = T.Compose([
            T.NormalizeScale(),
            T.SamplePoints(2048)
        ])
        root = os.path.join(root, "ModelNet" + name)
        if not os.path.exists(root):
            os.makedirs(root)
        self.data = ModelNet(root, name=name, train=train, transform=pre_process)
        self.transform = transform
        
        self.voxels = []
        self.labels = []
        print("Converting point clouds to voxels...")
        for i in tqdm(range(len(self.data))):
            pc = self.data.__getitem__(i)
            self.voxels.append(point_cloud_to_voxels(pc).voxels)
            self.labels.append(pc.y)
        print(Counter([l.item() for l in self.labels]))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        voxel = self.voxels[idx]
        label = self.labels[idx]
        if self.transform:
            voxel = self.transform(voxel)

        return voxel, label.squeeze()