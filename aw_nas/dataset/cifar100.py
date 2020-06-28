# -*- coding: utf-8 -*-
from torchvision import (datasets, transforms)

from aw_nas.utils.torch_utils import Cutout
from aw_nas.dataset.base import BaseDataset


class Cifar100(BaseDataset):
    NAME = "cifar100"

    def __init__(self, cutout=None):
        super(Cifar100, self).__init__()
        self.cutout = cutout

        cifar_mean = [0.5070751592371322, 0.4865488733149497, 0.44091784336703466]
        cifar_std = [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]

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
        self.datasets["train"] = datasets.CIFAR100(root=self.data_dir, train=True,
                                                   download=True, transform=train_transform)
        # temp for debug...
        # self.datasets["train"].data = self.datasets["train"].data[:1024]
        self.datasets["test"] = datasets.CIFAR100(root=self.data_dir, train=False,
                                                  download=True, transform=test_transform)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"
