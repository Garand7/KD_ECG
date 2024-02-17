import os
import numpy as np
import torch

from torch.utils.data import Dataset

class SegData(Dataset):
    def __init__(self, root):
        self.root=root

        root_data=os.path.join(root, 'sigs')
        root_lab=os.path.join(root, 'labels')

        self.datas=[os.path.join(root_data, x) for x in os.listdir(root_data) if x.endswith('.npy')]
        self.labels=[os.path.join(root_lab, x) for x in os.listdir(root_lab) if x.endswith('.npy')]

    def __getitem__(self, idx):
        data = np.load(self.datas[idx])
        label = np.load(self.labels[idx])

        label = (label*2).astype(int)

        return torch.from_numpy(data).float().unsqueeze(0), torch.from_numpy(label).long()

    def __len__(self):
        return len(self.datas)