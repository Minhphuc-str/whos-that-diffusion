import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class PokemonDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Paths to A (Input) and B (Target)
        self.dir_A = os.path.join(root_dir, mode, 'A')
        self.dir_B = os.path.join(root_dir, mode, 'B')
        
        # List of image filenames (ensure they match in both folders)
        self.image_names = sorted(os.listdir(self.dir_A))
        
        # Filter out non-image files (like .DS_Store on Mac)
        self.image_names = [x for x in self.image_names if x.endswith(('.png', '.jpg'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name)

        # Load images (BGR)
        img_A = cv2.imread(path_A) # Input: Silhouette
        img_B = cv2.imread(path_B) # Target: Color

        # Safety check if image failed to load
        if img_A is None or img_B is None:
            return self.__getitem__((index + 1) % len(self))

        # Convert BGR to RGB
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1] range (Standard for GANs)
        # (pixel - 127.5) / 127.5
        img_A = (img_A.astype(np.float32) - 127.5) / 127.5
        img_B = (img_B.astype(np.float32) - 127.5) / 127.5

        # --- Runtime Augmentation ---
        # For small datasets, we flip horizontally to effectively double data size
        if self.mode == 'train' and random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

        # Convert to PyTorch Tensor format: (H, W, C) -> (C, H, W)
        img_A = torch.from_numpy(img_A.transpose(2, 0, 1).copy())
        img_B = torch.from_numpy(img_B.transpose(2, 0, 1).copy())

        return {'A': img_A, 'B': img_B, 'name': img_name}