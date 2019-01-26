import os
import json
import numpy as np
import torch.utils.data as data

from PIL import Image


class Cityscapes(data.Dataset):
    """
    Modified from: https://pytorch.org/docs/master/_modules/torchvision/datasets/cityscapes.html#Cityscapes

    `Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Examples:

        Get semantic segmentation target

        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='gtFine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='gtFine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "gtCoarse" set

        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='val', mode='gtCoarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    def __init__(self, root, split='train', mode='gtFine', target_type='instance', transform=None):
        self.root = os.path.expanduser(root)
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, mode, split)
        self.transform = transform
        self.target_type = target_type
        self.split = split
        self.mode = mode
        self.images = []
        self.targets = []

        # Modifications to ignore trainId = [255, -1] as per Cityscapes label file and for training with correct index
        self.ignore_index = 255
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.n_classes = len(self.valid_classes)
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        if mode not in ['gtFine', 'gtCoarse']:
            raise ValueError('Invalid mode! Please use mode="gtFine" or mode="gtCoarse"')

        if mode == 'gtFine' and split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode "gtFine"! Please use split="train", split="test"'
                             ' or split="val"')
        elif mode == 'gtCoarse' and split not in ['train', 'train_extra', 'val']:
            raise ValueError('Invalid split for mode "gtCoarse"! Please use split="train", split="train_extra"'
                             ' or split="val"')

        if not isinstance(target_type, list):
            self.target_type = [target_type]

        if not all(t in ['instance', 'semantic', 'polygon', 'color'] for t in self.target_type):
            raise ValueError('Invalid value for "target_type"! Valid values are: "instance", "semantic", "polygon"'
                             ' or "color"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])

            # Modifications added to take care of ignore ids and updating ids
            elif t == 'semantic':
                temp = np.array(Image.open(self.targets[index][i])).astype(np.int32)
                temp = self._encode_target(temp)
                target = Image.fromarray(temp)
            else:
                target = np.array(Image.open(self.targets[index][i])).astype(np.int32)

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        sample = {'image': image, 'label': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Mode: {}\n'.format(self.mode)
        fmt_str += '    Type: {}\n'.format(self.target_type)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    def _encode_target(self, mask):
        # Fill void class with value 255
        for void_class in self.void_classes:
            mask[mask == void_class] = self.ignore_index

        # Fill valid class with updated index
        for valid_class in self.valid_classes:
            mask[mask == valid_class] = self.class_map[valid_class]

        return mask

