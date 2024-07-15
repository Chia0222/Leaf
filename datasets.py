import json
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms

class LeafDataset(data.Dataset):
    def __init__(self, opt):
        super(LeafDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.data_dir = osp.join(opt.dataset_dir, 'segmented')  # Adjusted path to segmented folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load data list
        img_pairs = []
        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img1_name, img2_name = line.strip().split(' ', 1)  # Split only once from the left
                img_pairs.append((img1_name, img2_name))

        self.img_pairs = img_pairs

        print(f"Loaded {len(self.img_pairs)} image pairs.")

    def __getitem__(self, index):
        img1_name, img2_name = self.img_pairs[index]

        # Load leaf images from segmented folders
        img1_path = self._find_image_path(img1_name)
        img2_path = self._find_image_path(img2_name)

        img1 = Image.open(img1_path).convert('RGB')
        img1 = transforms.Resize((self.load_width, self.load_height), interpolation=2)(img1)
        img1 = self.transform(img1)  # [-1,1]

        img2 = Image.open(img2_path).convert('RGB')
        img2 = transforms.Resize((self.load_width, self.load_height), interpolation=2)(img2)
        img2 = self.transform(img2)  # [-1,1]

        result = {
            'img1_name': img1_name,
            'img1': img1,
            'img2_name': img2_name,
            'img2': img2
        }
        return result

    def _find_image_path(self, img_name):
        for root, dirs, files in os.walk(self.data_dir):
            if img_name in files:
                return osp.join(root, img_name)

        raise FileNotFoundError(f"Image '{img_name}' not found in '{self.data_dir}'.")

    def __len__(self):
        return len(self.img_pairs)


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


# Example of an Options class
class Options:
    def __init__(self):
        self.load_height = 256
        self.load_width = 256
        self.dataset_dir = './datasets'
        self.dataset_list = 'test_pairs.txt'
        self.batch_size = 32
        self.workers = 4
        self.shuffle = True

# Create options instance
opt = Options()

# Initialize the dataset
dataset = LeafDataset(opt)

# Verify dataset length
print(f"Dataset length: {len(dataset)}")

# Initialize the data loader
data_loader = VITONDataLoader(opt, dataset)

# Check for batches
try:
    batch = data_loader.next_batch()
    print("Batch loaded successfully.")
except StopIteration:
    print("No more batches available.")

# Optionally, loop through the data loader to ensure it works correctly
for i, batch in enumerate(data_loader.data_loader):
    print(f"Batch {i} loaded successfully.")
    if i >= 2:  # Limit the number of batches printed
        break
