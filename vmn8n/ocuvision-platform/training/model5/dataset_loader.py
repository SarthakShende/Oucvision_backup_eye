from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p,label = self.items[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label,dtype=torch.long)
