# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import io

from torchvision.datasets import ImageFolder
from torchvision import transforms


import os
import scipy.io as sio

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class Rescale(object):
    def __init__(self):
        self.xmax = None
        self.xmin = None

    def __call__(self, pic):
        if self.xmax is None:
            self.xmax = pic.max()
            self.xmin = pic.min()
            pic = 255 * (pic - self.xmin) / (self.xmax - self.xmin)
            return pic.astype('uint8')
        return self.xmin + pic * (self.xmax - self.xmin)

def create_dataset(root, train=True, dataugmentation=False):
    # load dataset
    if not '.mat' in root:
        mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]
        tr = [transforms.ToTensor(), transforms.Normalize(mean=mean_pix, std=std_pix)]
        if dataugmentation:
            dt = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            tr = dt + tr
        dataset = torchvision.datasets.CIFAR10(
            root,
            train=train,
            transform=transforms.Compose(tr),
            download=True,
        )
        return dataset
    else:
        tr = [transforms.ToTensor()]
        if dataugmentation:
            dt = [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            tr = dt + tr
        dataset = CIFARmatlab(
            root,
            train=train,
            transform=transforms.Compose(tr),
            augment=dataugmentation
        )
        return dataset 


class CIFARmatlab(data.Dataset):
    def __init__(self, root, train=True, transform=None, augment=False, dtype='float32'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train # training set or test set
        if self.train:
            split = 'tr'
        else:
            split = 'te'
        matdata = sio.loadmat(root)
        R = matdata['X' + split][:, :32, :].transpose(2, 1, 0)
        G = matdata['X' + split][:, 32: 64, :].transpose(2, 1, 0)
        B = matdata['X' + split][:, 64:, :].transpose(2, 1, 0)
        data = np.stack([R, G, B], axis=3)
        labels = [e[0] for e in matdata['Y' + split]]
        data = data.astype(dtype)
        labels = labels
        if self.train:
            self.train_data = data
            self.train_labels = labels
        else:
            self.test_data = data
            self.test_labels = labels
        self.augment = augment
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            if self.augment:
                rs = Rescale()
                img = rs(img)
            img = self.transform(img)
            if self.augment:
                img = rs(img)
                del rs
        target = torch.tensor(target, dtype=torch.long)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)












def create_tiny_imagenet_dataset(root, train=True, data_augmentation=False):
    """
    Create a Tiny-ImageNet dataset loader using ImageFolder.
    
    Args:
        root (str): Root directory of the Tiny-ImageNet dataset (i.e. the directory containing train/ and val/ folders).
        train (bool): If True, load the training split; otherwise load the validation split.
        data_augmentation (bool): If True, apply data augmentation on training images.
    
    Returns:
        dataset: A torchvision ImageFolder dataset.
    """
    # Choose the proper subfolder.
    if train:
        data_path = os.path.join(root, 'train')
    else:
        data_path = os.path.join(root, 'val')
    
    # Define the transforms. Tiny-ImageNet images are 64x64.
    transform_list = [transforms.Resize((64, 64))]
    if train and data_augmentation:
        transform_list.extend([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_list.append(transforms.ToTensor())
    # Optionally, add normalization if you have mean and std values.
    transform = transforms.Compose(transform_list)
    
    dataset = ImageFolder(data_path, transform=transform)
    return dataset


class TinyImageNetDeeplake(Dataset):
    def __init__(self, deeplake_path, train=True, transform=None):
        """
        Args:
            deeplake_path (str): path to the Deeplake Tiny-ImageNet dataset.
            train (bool): whether to load the training split.
            transform: torchvision transforms to apply.
        """
        # Load the dataset from Deeplake.
        self.ds = deeplake.load(deeplake_path, read_only=True)
        # Assume the Deeplake dataset contains separate collections for train and test,
        # or a field that indicates which split the sample belongs to.
        # For example, if your dataset has a field "split" with values "train" and "test":
        self.split = 'train' if train else 'test'
        self.transform = transform

        # Filter the dataset based on the split.
        self.samples = [sample for sample in self.ds if sample['split'] == self.split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # Assume the sample contains an image in raw bytes (or as a file path) and a label.
        # Here we assume sample['image'] is raw image bytes.
        image_bytes = sample['image']
        # Open the image using PIL.
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = int(sample['label'])
        return image, label

def create_deeplake_dataset(path, train=True, data_augmentation=False):
    # Define transformations appropriate for Tiny-ImageNet.
    # Typically, you might want to resize images to 64x64 if they are not already that size.
    base_transforms = [transforms.Resize((64, 64)), transforms.ToTensor()]
    if data_augmentation and train:
        aug_transforms = [transforms.RandomCrop(64, padding=4),
                          transforms.RandomHorizontalFlip()]
        transform = transforms.Compose(aug_transforms + base_transforms)
    else:
        transform = transforms.Compose(base_transforms)
    return TinyImageNetDeeplake(deeplake_path=path, train=train, transform=transform)


# In data.py, add:

from torchvision.datasets import SVHN
from torchvision import transforms

def create_svhn_dataset(root, split='train', data_augmentation=False):
    """
    Create an SVHN dataset loader.
    
    Args:
        root (str): Directory where SVHN will be stored.
        split (str): Either 'train' or 'test'.
        data_augmentation (bool): If True, apply data augmentation.
    
    Returns:
        dataset: A torchvision SVHN dataset.
    """
    transform_list = []
    if data_augmentation and split == 'train':
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_list.append(transforms.ToTensor())
    # Optionally, add normalization (using SVHN's mean and std if available)
    transform = transforms.Compose(transform_list)
    
    dataset = SVHN(root=root, split=split, transform=transform, download=True)
    return dataset


from torchvision.datasets import STL10
import torchvision.transforms as transforms

def create_stl10_dataset(root, split='train', data_augmentation=False):
    """
    Creează un dataset STL-10. STL-10 are imagini de 96×96 px, 10 clase.
    split poate fi: 'train', 'test', 'unlabeled' etc.

    Args:
        root (str): Calea unde se va stoca/căuta STL-10.
        split (str): 'train', 'test', 'unlabeled' etc.
        data_augmentation (bool): Dacă True, aplică transformări la datele de train.

    Returns:
        Un obiect dataset de tip torchvision.datasets.STL10
    """
    transform_list = []
    if data_augmentation and split == 'train':
        transform_list.extend([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    dataset = STL10(root=root, split=split, transform=transform, download=True)
    return dataset


