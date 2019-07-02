# -*- coding: utf-8 -*-
import os

from torchvision import (datasets, transforms)

from aw_nas.dataset.base import BaseDataset

class TinyImagenet(BaseDataset):
    NAME = "tiny-imagenet"

    def __init__(self):
        super(TinyImagenet, self).__init__()

        self.train_data_dir = os.path.join(self.data_dir, "train")
        # the test split has no label information
        self.test_data_dir = os.path.join(self.data_dir, "val")
        self.datasets = {}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
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
