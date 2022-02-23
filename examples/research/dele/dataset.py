# -*- coding: utf-8 -*-
# pylint: disable-all

from typing import List

import numpy as np
from torch.utils.data import Dataset


class ArchDataset(Dataset):
    def __init__(self, data, low_fidelity_type: str, low_fidelity_normalize: bool = False):
        self.data = data
        self.low_fidelity_type = low_fidelity_type
        self.low_fidelity_normalize = low_fidelity_normalize
        
        if low_fidelity_normalize:
            low_fidelity_list = [_data[2][low_fidelity_type] for _data in data]
            self._low_fidelity_min = np.min(low_fidelity_list)
            self._low_fidelity_max = np.max(low_fidelity_list)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arch, acc, all_low_fidelity = self.data[idx]
        low_fidelity = all_low_fidelity[self.low_fidelity_type]
        if self.low_fidelity_normalize:
            low_fidelity = (low_fidelity - self._low_fidelity_min) / (self._low_fidelity_max - self._low_fidelity_min)
        return arch, acc, low_fidelity


class NasBench301Dataset(Dataset):
    def __init__(self, data, train: bool, low_fidelity_type: str, low_fidelity_normalize: bool = False):
        self.data = data
        self.train = train
        self.low_fidelity_type = low_fidelity_type
        self.low_fidelity_normalize = low_fidelity_normalize

        if self.low_fidelity_type == "loss":
            assert self.low_fidelity_normalize == True

        if low_fidelity_normalize and train:
            low_fidelity_list = [_data[2][low_fidelity_type] for _data in data]
            self._low_fidelity_min = np.min(low_fidelity_list)
            self._low_fidelity_max = np.max(low_fidelity_list)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train:
            arch, acc, all_low_fidelity = self.data[idx]
            low_fidelity = all_low_fidelity[self.low_fidelity_type]
            if self.low_fidelity_normalize:
                low_fidelity = (low_fidelity - self._low_fidelity_min) / (self._low_fidelity_max - self._low_fidelity_min)
            return arch, acc, low_fidelity
        return self.data[idx]


class MultiLFArchDataset(Dataset):
    def __init__(self, data, low_fidelity_type: List[str], low_fidelity_normalize: bool = False):
        self.data = data
        self.low_fidelity_type = low_fidelity_type
        self.low_fidelity_normalize = low_fidelity_normalize

        if self.low_fidelity_normalize:
            self.low_fidelity_min_max = {}
            for low_fidelity in self.low_fidelity_type:
                low_fidelity_list = [_data[2][low_fidelity] for _data in data]
                low_fidelity_min = np.min(low_fidelity_list)
                low_fidelity_max = np.max(low_fidelity_list)
                self.low_fidelity_min_max[low_fidelity] = {}
                self.low_fidelity_min_max[low_fidelity]["min"] = low_fidelity_min
                self.low_fidelity_min_max[low_fidelity]["max"] = low_fidelity_max

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arch, acc, all_low_fidelity = self.data[idx]
        low_fidelity_perfs = {}
        for low_fidelity in self.low_fidelity_type:
            low_fidelity_perf = all_low_fidelity[low_fidelity]
            if self.low_fidelity_normalize:
                _min = self.low_fidelity_min_max[low_fidelity]["min"]
                _max = self.low_fidelity_min_max[low_fidelity]["max"]
                low_fidelity_perf = (low_fidelity_perf - _min) / (_max - _min)
            low_fidelity_perfs[low_fidelity] = low_fidelity_perf
        return arch, acc, low_fidelity_perfs
