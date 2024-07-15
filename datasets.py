import json
from os import path as osp

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
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        labels = []
        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                parts = line.strip().rsplit(' ', 1)  # Split from the right, only once
                if len(parts) != 2:
                    print(f"Error: Line '{line.strip()}' does not have exactly 2 parts")
                else:
                    img_name, label = parts
                    img_names.append(img_name)
                    labels.append(label)

        self.img_names = img_names
        self.labels = labels

    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = self.labels[index]

        # load leaf image
        img = Image.open(osp.join(self.data_path, 'images', img_name)).convert('RGB')
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        img = self.transform(img)  # [-1,1]

        result = {
            'img_name': img_name,
            'img': img,
            'label': label
        }
        return result

    def __len__(self):
        return len(self.img_names)


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
        self.dataset_dir = 'datasets'
        self.dataset_list = 'test_pairs.txt'
        self.dataset_mode = 'test'
        self.batch_size = 32
        self.workers = 4
        self.shuffle = True

# Create options instance
opt = Options()

# Initialize the dataset
dataset = LeafDataset(opt)

# Initialize the data loader
data_loader = VITONDataLoader(opt, dataset)

# Get the next batch
batch = data_loader.next_batch()
