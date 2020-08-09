# -*- coding: utf-8 -*-
from torchvision import (datasets, transforms)

from aw_nas.utils.torch_utils import Cutout
from aw_nas.dataset.base import BaseDataset


class SVHN(BaseDataset):
    NAME = "SVHN"

    def __init__(self, cutout=None):
        super(SVHN, self).__init__()
        self.cutout = cutout

        svhn_mean = [0.4377, 0.4438, 0.4728]
        svhn_std = [0.1980, 0.2010, 0.1970]

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])

        extra_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])

        self.datasets = {}
        self.datasets["train"] = datasets.SVHN(root=self.data_dir, split='train',
                                                   download=True, transform=train_transform)
        self.datasets["test"] = datasets.SVHN(root=self.data_dir, split='test',
                                                  download=True, transform=test_transform)
        self.datasets["extra"] = datasets.SVHN(root=self.data_dir, split='extra',
                                              download=True, transform=extra_transform)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"