import json
from os import path as osp
import numpy as np
from PIL import Image
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
            transforms.Resize((self.load_height, self.load_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load data pairs list
        self.pairs = []
        with open(osp.join(self.data_path, 'test_pairs.txt'), 'r') as f:
            for line in f.readlines():
                # Split line into two parts based on the first space encountered
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    healthy_img_name, diseased_img_name = parts
                    self.pairs.append((healthy_img_name, diseased_img_name))
                else:
                    print(f"Ignoring invalid line in test_pairs.txt: {line}")

    def __getitem__(self, index):
        healthy_img_name, diseased_img_name = self.pairs[index]

        # Load healthy image
        healthy_img = Image.open(osp.join(self.data_path, healthy_img_name)).convert('RGB')
        healthy_img = self.transform(healthy_img)

        # Load diseased image
        diseased_img = Image.open(osp.join(self.data_path, diseased_img_name)).convert('RGB')
        diseased_img = self.transform(diseased_img)

        result = {
            'healthy_img_name': healthy_img_name,
            'diseased_img_name': diseased_img_name,
            'healthy_img': healthy_img,
            'diseased_img': diseased_img,
        }
        return result

    def __len__(self):
        return len(self.pairs)


class LeafDataLoader:
    def __init__(self, opt, dataset):
        super(LeafDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = iter(self.data_loader)

    def next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)

        return batch


class Options:
    def __init__(self):
        self.load_height = 256
        self.load_width = 192
        self.dataset_dir = 'datasets'
        self.dataset_mode = 'segmented'
        self.batch_size = 4
        self.workers = 4
        self.shuffle = True


opt = Options()

# Initialize the dataset
dataset = LeafDataset(opt)

# Initialize the data loader
data_loader = LeafDataLoader(opt, dataset)

# Fetch a batch
batch = data_loader.next_batch()

# Print batch contents
print(batch['healthy_img_name'])
print(batch['diseased_img_name'])
print(batch['healthy_img'].shape)
print(batch['diseased_img'].shape)
