# -*- coding: utf-8 -*-
# refence to:
# https://github.com/dongzelian/T-NAS/blob/master/MiniImagenet.py
# https://github.com/dongzelian/T-NAS/blob/master/MiniImagenet_task.py

import os
import csv
import random
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from aw_nas.dataset.base import BaseDataset


class MiniImagenetDataset(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(
        self, root, mode, batch_size, n_way, k_shot, k_query, resize, transform=None
    ):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batch_size: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param transform: transform from filename to image
        """
        self.root = root
        self.path = os.path.join(self.root, "images")

        self.mode = mode
        self.batch_size = batch_size  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = (
            self.n_way * self.k_query
        )  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.transform = transform
        print(
            "shuffle Mini-ImageNet:%s, b:%d, %d-way, %d-shot, %d-query, resize:%d"
            % (mode, batch_size, n_way, k_shot, k_query, resize),
            end="",
        )

        # csvdata is a dict, len(csvdata.keys())=64  64*600=38400
        csvdata = self.loadCSV(os.path.join(root, mode + ".csv"))  # csv path

        self.data = []  # list of list
        self.img2label = {}  # img2label dict
        for idx, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img601, ...]]
            self.img2label[k] = idx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)  # 64

        if mode == "train":
            self.create_batch(self.batch_size)
            print("")
        else:
            support_path = os.path.join(
                root,
                mode
                + "_"
                + str(batch_size)
                + "_support_"
                + str(n_way)
                + "_way_"
                + str(k_shot)
                + "shot.json",
            )
            query_path = os.path.join(
                root,
                mode
                + "_"
                + str(batch_size)
                + "_query_"
                + str(n_way)
                + "way_"
                + str(k_shot)
                + "shot.json",
            )
            if not os.path.isfile(os.path.join(support_path)):
                self.create_batch(self.batch_size)
                with open(support_path, "w+") as f_support:
                    json.dump(self.support_x_batch, f_support)
                with open(query_path, "w+") as f_query:
                    json.dump(self.query_x_batch, f_query)
                print("")
            else:
                self.support_x_batch = json.load(open(support_path))
                self.query_x_batch = json.load(open(query_path))
                print("\tloaded from json file.")

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]

        return dictLabels

    def create_batch(self, batch_size):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batch_size):  # for each batch
            # 1.select n_way classes randomly, choose 5-way from 64 classes, no duplicate
            selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)

            np.random.shuffle(selected_cls)
            support_x = []  # list of list
            query_x = []  # list of list
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class

                selected_imgs_idx = np.random.choice(
                    len(self.data[cls]), self.k_shot + self.k_query, replace=False
                )
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(
                    selected_imgs_idx[: self.k_shot]
                )  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot :])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist()
                )  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)  # 5*1
            random.shuffle(query_x)  # 5*15

            self.support_x_batch.append(
                support_x
            )  # append set to current sets, 10000*5*1
            self.query_x_batch.append(
                query_x
            )  # append sets to current sets, 10000*5*15

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batch_size-1
        :param index:
        :return:
        """
        return self.getitem_from_batch(
            self.support_x_batch[index], self.query_x_batch[index]
        )

    def getitem_from_batch(self, support_x_batch, query_x_batch):
        support_x = torch.FloatTensor(self.set_size, 3, self.resize, self.resize)
        query_x = torch.FloatTensor(self.query_size, 3, self.resize, self.resize)

        flatten_support_x = [
            os.path.join(self.path, item)
            for sublist in support_x_batch
            for item in sublist
        ]
        support_y = np.array(
            [item[:9] for sublist in support_x_batch for item in sublist]
        )

        flatten_query_x = [
            os.path.join(self.path, item)
            for sublist in query_x_batch
            for item in sublist
        ]
        query_y = np.array([item[:9] for sublist in query_x_batch for item in sublist])

        # generate relative y in range(0, n_way)
        unique = np.unique(support_y)
        random.shuffle(unique)

        support_y_relative = np.zeros(self.set_size)
        query_y_relative = np.zeros(self.query_size)
        for idx, label in enumerate(unique):
            support_y_relative[support_y == label] = idx
            query_y_relative[query_y == label] = idx

        # apply transform, from path to image
        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        return (
            support_x,
            torch.LongTensor(support_y_relative),
            query_x,
            torch.LongTensor(query_y_relative),
        )

    def __len__(self):
        # as we have built up to batch_size of sets, you can sample some small batch size of sets.
        return self.batch_size


class MiniImageNet(BaseDataset):
    NAME = "miniimagenet"

    def __init__(
        self,
        batch_size,
        n_way,
        k_shot,
        k_query,
        train_shot=None,
        resize=84,
        load_val=True,
        batch_size_val=100,
        load_test=False,
        batch_size_test=1000,
    ):
        super(MiniImageNet, self).__init__()

        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.train_shot = train_shot if train_shot else k_shot
        self.k_query = k_query
        self.resize = resize
        self.load_val = load_val
        self.batch_size_val = batch_size_val
        self.load_test = load_test
        self.batch_size_test = batch_size_test

        self.datasets = {}

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            [
                self.load_image,
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.datasets["train"] = MiniImagenetDataset(
            root=self.data_dir,
            mode="train",
            batch_size=self.batch_size,
            n_way=self.n_way,
            k_shot=self.train_shot,
            k_query=self.k_query,
            resize=self.resize,
            transform=transform,
        )

        if self.load_val:
            self.datasets["val"] = MiniImagenetDataset(
                root=self.data_dir,
                mode="val",
                batch_size=self.batch_size_val,
                n_way=self.n_way,
                k_shot=self.k_shot,
                k_query=self.k_query,
                resize=self.resize,
                transform=transform,
            )
        if self.load_test:
            self.datasets["test"] = MiniImagenetDataset(
                root=self.data_dir,
                mode="test",
                batch_size=self.batch_size_test,
                n_way=self.n_way,
                k_shot=self.k_shot,
                k_query=self.k_query,
                resize=self.resize,
                transform=transform,
            )

    def load_image(self, x):
        return Image.open(x).convert("RGB")

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"

    def getitem_from_batch(self, support_x_batch, query_x_batch):
        return self.datasets["train"].getitem_from_batch(support_x_batch, query_x_batch)
