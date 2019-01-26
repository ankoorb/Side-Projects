import json
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    COCO Dataset to be used in DataLoader for creating batches 
    during training.
    """

    def __init__(self, config, split='TRAIN', transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        # Open files where images are stored in HDF5 data fromat, captions & their lengths
        if self.split == 'TRAIN':
            self.hdf5 = h5py.File(name=self.config.train_hdf5, mode='r')
            self.captions = self.read_json(self.config.train_captions)
        else:
            self.hdf5 = h5py.File(name=self.config.val_hdf5, mode='r')
            self.captions = self.read_json(self.config.val_captions)

        # Get image data
        self.images = self.hdf5['images']

    def read_json(self, json_path):
        with open(json_path, 'r') as j:
            json_data = json.load(j)
        return json_data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        # There are 5 captions so randomly sample 1 caption
        cap_idx = np.random.randint(0, high=5)
        caption = torch.LongTensor(self.captions[idx][0][cap_idx])
        length = torch.LongTensor([self.captions[idx][1][cap_idx]])

        if self.split == 'TRAIN':
            return img, caption, length
        else:
            captions = torch.LongTensor(self.captions[idx][0])
            return img, caption, length, captions

