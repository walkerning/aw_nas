# -*- coding: utf-8 -*-
import os
import pickle

from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import vision

from aw_nas.utils.torch_utils import Cutout
from aw_nas.dataset.base import BaseDataset

class ImageNetDownsampleDataset(vision.VisionDataset):
    train_list = [
        "train_data_batch_1",
        "train_data_batch_2",
        "train_data_batch_3",
        "train_data_batch_4",
        "train_data_batch_5",
        "train_data_batch_6",
        "train_data_batch_7",
        "train_data_batch_8",
        "train_data_batch_9",
        "train_data_batch_10"
    ]

    test_list = [
        "val_data"
    ]

    def __init__(self, root, num_class=1000, size=16, train=True,
                 transform=None, target_transform=None):
        super(ImageNetDownsampleDataset, self).__init__(root, transform=transform,
                                                        target_transform=target_transform)

        self.train = train  # training set or test set
        file_list = self.train_list if self.train else self.test_list

        self.num_class = num_class # the first `num_class` classes are kept

        len_ = 3 * size * size
        self.data = np.zeros((0, len_), dtype=np.uint8)
        self.targets = []

        for file_name in file_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f)
            if num_class < 1000:
                mask = np.array(entry["labels"]) <= num_class
                self.data = np.concatenate((self.data, entry["data"][mask]), axis=0)
                self.targets.extend(list((np.array(entry["labels"]) - 1)[mask]))
            else:
                self.data = np.concatenate((self.data, entry["data"]), axis=0)
                self.targets.extend(list(np.array(entry["labels"]) - 1))

        self.data = self.data.reshape(-1, 3, size, size).transpose((0, 2, 3, 1)) # HWC for PIL

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageNetDownsample(BaseDataset):
    NAME = "imagenet_downsample"

    def __init__(self, num_class=120, size=16, relative_dir=None, cutout=None):
        super(ImageNetDownsample, self).__init__(relative_dir=relative_dir)

        self.cutout = cutout
        self.num_class = num_class
        self.size = size

        # array([122.68245678, 116.65812896, 104.00708381])
        imgnet_mean = [0.48110767, 0.45748286, 0.40787092]
        imgnet_std = [0.229, 0.224, 0.225] # use imgnet

        train_transform = transforms.Compose([
            transforms.RandomCrop(16, padding=2), # follow NB201
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ])
        if self.cutout:
            train_transform.transforms.append(Cutout(self.cutout))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std),
        ])

        self.datasets = {}
        self.datasets["train"] = ImageNetDownsampleDataset(
            root=self.data_dir, num_class=self.num_class, size=self.size,
            train=True, transform=train_transform)
        self.datasets["train_testTransform"] = ImageNetDownsampleDataset(
            root=self.data_dir, num_class=self.num_class, size=self.size,
            train=True, transform=test_transform)
        self.datasets["test"] = ImageNetDownsampleDataset(
            root=self.data_dir, num_class=self.num_class, size=self.size,
            train=False, transform=test_transform)

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
        return ImageNetDownsample, (self.cutout,)

    def __getinitargs__(self):
        """
        Python 2
        getinitargs for pickling (mainly for use with async search see trainer/async_trainer.py)
        """
        return (self.cutout,)
