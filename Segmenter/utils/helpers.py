import torch
import random
import numpy as np

from PIL import Image, ImageFilter, ImageOps


def _make_divisible(v, divisor, min_value=None):
    """
    This function makes sure that number of channels number is divisible by 8.
    Source: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class AverageMeter(object):
    """
    Computes and stores the average and current value of some metric.

    Reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Normalize(object):
    """
    Normalize an image (tensor) with mean and standard deviation. This
    should be just before ToTensor.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Convert PIL data to NumPy array
        img = np.array(img).astype(np.float32)
        lab = np.array(lab).astype(np.float32)

        # Normalize img
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img, 'label': lab}


class ToTensor(object):
    """
    Convert NumPy arrays to PyTorch tensors. This should be 
    the last transformation.
    """

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Convert PIL data to NumPy array
        img = np.array(img).astype(np.float32)
        lab = np.array(lab).astype(np.float32)

        # H x W x C -> C x H x W
        img = img.transpose((2, 0, 1))

        # Convert NumPy array to PyTorch tensor
        img = torch.from_numpy(img).float()
        lab = torch.from_numpy(lab).float()

        return {'image': img, 'label': lab}


class RandomHorizontalFlip(object):
    """
    Randomly horizontal flip image and label.

    NOTE: Returns data in PIL format
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Horizontally flip
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img, 'label': lab}


class RandomGaussianBlur(object):
    """
    Randomly apply Gaussian blur to image only.

    NOTE: Returns data in PIL format
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Apply Gaussian blur to image
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {'image': img, 'label': lab}


class FixedResize(object):
    """
    Resizes image and label to a fixed size.

    NOTE: Returns data in PIL format
    """

    def __init__(self, size=256):
        self.size = (size, size)

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Resize image and label
        img = img.resize(size=self.size, resample=Image.BILINEAR)
        lab = lab.resize(size=self.size, resample=Image.NEAREST)

        return {'image': img, 'label': lab}


class FixedScaleCrop(object):
    """
    Resizes image and label to a fixed size and then returns
    center cropped image and label

    NOTE: Returns data in PIL format
    """

    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Compute resize width and height
        width, height = img.size
        if width > height:
            resize_h = self.crop_size
            resize_w = int(resize_h * float(width) / height)
        else:
            resize_w = self.crop_size
            resize_h = int(resize_w * float(height) / width)

        # Resize image and label
        img = img.resize(size=(resize_w, resize_h), resample=Image.BILINEAR)
        lab = lab.resize(size=(resize_w, resize_h), resample=Image.NEAREST)

        # Center crop the resized image
        x1 = int(round(resize_w - self.crop_size) / 2.0)
        y1 = int(round(resize_h - self.crop_size) / 2.0)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size

        img = img.crop(box=(x1, y1, x2, y2))
        lab = lab.crop(box=(x1, y1, x2, y2))

        return {'image': img, 'label': lab}


class RandomScaleCrop(object):
    """
    Resize image and label by a random scale, and then randomly 
    crop the resized image and label.

    base_size must be > crop_size and multiple of 8

    fill: int, for ignoring purpose, as labelId 255 is to be ignored

    NOTE: Returns data in PIL format
    """

    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        # Extract PIL image and PIL label from dict
        img = sample['image']
        lab = sample['label']

        # Randomly scale short edge
        short_size = random.randint(int(self.base_size * 0.75),
                                    int(self.base_size * 1.75))

        # Compute resize width and height
        width, height = img.size
        if width > height:
            resize_h = short_size
            resize_w = int(resize_h * float(width) / height)
        else:
            resize_w = short_size
            resize_h = int(resize_w * float(height) / width)

        # Resize image and label
        img = img.resize(size=(resize_w, resize_h), resample=Image.BILINEAR)
        lab = lab.resize(size=(resize_w, resize_h), resample=Image.NEAREST)

        # Pad image and label
        if short_size < self.crop_size:
            pad_h = self.crop_size - resize_h if resize_h < self.crop_size else 0
            pad_w = self.crop_size - resize_w if resize_w < self.crop_size else 0

            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            lab = ImageOps.expand(lab, border=(0, 0, pad_w, pad_h), fill=self.fill)

        # Randomly crop the resized image and label
        max_x = 1 if resize_w - self.crop_size < 0 else resize_w - self.crop_size
        max_y = 1 if resize_h - self.crop_size < 0 else resize_h - self.crop_size
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size

        img = img.crop(box=(x1, y1, x2, y2))
        lab = lab.crop(box=(x1, y1, x2, y2))

        return {'image': img, 'label': lab}

