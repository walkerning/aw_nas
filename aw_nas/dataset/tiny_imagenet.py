# -*- coding: utf-8 -*-
import os

from torchvision import (datasets, transforms)

from aw_nas.utils.torch_utils import Cutout
from aw_nas.dataset.base import BaseDataset

class TinyImagenet(BaseDataset):
    NAME = "tiny-imagenet"

    def __init__(self, color_jitter=False, train_crop_size=64, test_crop_size=64, cutout=None):
        super(TinyImagenet, self).__init__()

        self.cutout = cutout
        self.train_data_dir = os.path.join(self.data_dir, "train")
        # the test split has no label information
        self.test_data_dir = os.path.join(self.data_dir, "val")
        self.datasets = {}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if not color_jitter:
            train_transform = transforms.Compose([
                # transforms.RandomCrop(64, padding=4),
                transforms.RandomResizedCrop(train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                # transforms.RandomCrop(64, padding=4),
                # random crop, resize->tarin_crop_size
                transforms.RandomResizedCrop(train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout))

        test_transform = transforms.Compose([
            transforms.Resize(test_crop_size + 8),
            transforms.CenterCrop(test_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.datasets = {}
        self.datasets["train"] = datasets.ImageFolder(root=self.train_data_dir,
                                                      transform=train_transform)

        self.datasets["test"] = datasets.ImageFolder(root=self.test_data_dir,
                                                     transform=test_transform)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"
