# -*- coding: utf-8 -*-
from torchvision import (datasets, transforms)

from aw_nas.utils.torch_utils import Cutout
from aw_nas.dataset.base import BaseDataset

class Cifar10(BaseDataset):
    NAME = "cifar10"

    def __init__(self, cutout=None):
        super(Cifar10, self).__init__()
        self.cutout = cutout

        cifar_mean = [0.49139968, 0.48215827, 0.44653124]
        cifar_std = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])

        self.datasets = {}
        self.datasets["train"] = datasets.CIFAR10(root=self.data_dir, train=True,
                                                  download=True, transform=train_transform)
        self.datasets["train_testTransform"] = datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=test_transform)
        # temp for debug...
        # self.datasets["train"].data = self.datasets["train"].data[:1024]
        self.datasets["test"] = datasets.CIFAR10(root=self.data_dir, train=False,
                                                 download=True, transform=test_transform)

    def same_data_split_mapping(self):
        return {"train_testTransform": "train"}

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"

    def __reduce__(self):
        """
        Python 3
        reduce for pickling (mainly for use with async search see trainer/async_trainer.py)
        """
        return Cifar10, (self.cutout,)

    def __getinitargs__(self):
        """
        Python 2
        getinitargs for pickling (mainly for use with async search see trainer/async_trainer.py)
        """
        return (self.cutout,)
