import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.data, self.labels = [], []
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'rb') as fo:
                entry = pickle.load(fo, encoding='bytes')
                self.data.append(entry[b'data'])
                self.labels += entry[b'labels']
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = torch.tensor(self.data, dtype=torch.float) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
