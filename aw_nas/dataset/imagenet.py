# -*- coding: utf-8 -*-
import os
from datetime import datetime
import six

import numpy as np
from torchvision import (datasets, transforms)

from aw_nas.utils.torch_utils import Cutout
from aw_nas.dataset.base import BaseDataset

class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(ImageNetDataset, self).__init__(*args, **kwargs)

    def filter(self, num_classes=100, random_choose=False, random_seed=None, class_names=None):
        _total_classes = len(self.classes)
        if num_classes is None or num_classes == _total_classes:
            return self.classes

        if class_names is not None:
            _samples = []
            _cls_map = {self.class_to_idx[name]:idx for idx, name in enumerate(class_names)}
            for item in self.samples:
                if item[1] in _cls_map:
                    _samples.append((item[0], _cls_map[item[1]]))
            self.samples = _samples
            self.classes = class_names
            self.class_to_idx = {n: i for i, n in enumerate(class_names)}
            return self.classes

        if random_choose:
            if not random_seed is None:
                assert isinstance(random_seed, int)
                np.random.seed(random_seed)
            choosen_cls_idx = list(np.random.choice(list(range(_total_classes)),
                                                    num_classes, replace=False))
        else:
            choosen_cls_idx = list(range(num_classes))

        _cls_map = {idx: i for i, idx in enumerate(choosen_cls_idx)}
        _samples = []
        for item in self.samples:
            if item[1] in _cls_map:
                _samples.append((item[0], _cls_map[item[1]]))
        self.samples = _samples
        idx_to_class = {idx: name for name, idx in six.iteritems(self.class_to_idx)}
        self.classes = [idx_to_class[idx] for idx in choosen_cls_idx]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        return self.classes


class ImageNet(BaseDataset):
    NAME = "imagenet"

    def __init__(self, load_train_only=False, class_name_file=None, num_sample_classes=None,
                 random_choose=False, random_seed=123, color_jitter=False, cutout=None,
                 train_crop_size=224, test_crop_size=224):
        super(ImageNet, self).__init__()

        self.cutout = cutout

        self.load_train_only = load_train_only
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.class_name_file = class_name_file
        if class_name_file is not None:
            class_names = self._read_names_from_file(class_name_file)
        else:
            class_names = None

        self.datasets = {}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if not color_jitter:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
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

        self.datasets["train"] = ImageNetDataset(root=self.train_data_dir,
                                                 transform=train_transform)
        self.choosen_classes = self.datasets["train"].filter(num_classes=num_sample_classes,
                                                             random_choose=random_choose,
                                                             random_seed=random_seed,
                                                             class_names=class_names)
        self.logger.info("Number of choosed classes: %d", len(self.choosen_classes))
        if class_name_file is None:
            self.class_name_file = os.path.join(self.data_dir,
                                                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") \
                                                + ".txt")
            self._write_names_to_file(self.choosen_classes, self.class_name_file)
            self.logger.info("Write class names to %s", self.class_name_file)

        if not self.load_train_only:
            self.test_data_dir = os.path.join(self.data_dir, "test")
            test_transform = transforms.Compose([
                transforms.Resize(test_crop_size + 32),
                transforms.CenterCrop(test_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            self.datasets["test"] = ImageNetDataset(root=self.test_data_dir,
                                                    transform=test_transform)
            self.datasets["test"].filter(num_classes=num_sample_classes,
                                         class_names=self.choosen_classes)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"

    @staticmethod
    def _read_names_from_file(path):
        with open(path, "r") as f:
            return f.read().strip().split("\n")

    @staticmethod
    def _write_names_to_file(names, path):
        with open(path, "w") as f:
            f.write("\n".join(names))
