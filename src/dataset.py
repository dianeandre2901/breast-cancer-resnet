"""
dataset.py
-----------
Defines the CustomImageDataset class for loading breast cancer histopathology images from preprocessed arrays.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Array of image data (e.g., shape: N x H x W x C)
            labels (np.ndarray): Array of labels (e.g., 0 for benign, 1 for malignant)
            transform (callable, optional): Optional torchvision transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
