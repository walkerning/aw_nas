"""
Definitions of mnasnet OFA rollout and search space.
"""

import os
import copy
import glob
import shutil
import collections

import six
import yaml
import numpy as np

from aw_nas import utils
from aw_nas.utils import expect
from aw_nas.base import Component
from aw_nas.rollout.base import Rollout
from aw_nas.common import get_genotype_substr, genotype_from_str, SearchSpace
from aw_nas.utils import logger as _logger

class MNasNetOFASearchSpace(SearchSpace):
    NAME = "mnasnet_ofa"

    def __init__(self, width_choice=[4,5,6], depth_choice=[4,5,6],
                 num_cells=6, num_channels=[6,6,6,6,6,6]):
        super(MNasNetOFASearchSpace, self).__init__()
        self.genotype_type_name = "MnasnetOFAGenotype"
        self.block_names = sum(
            [["cell_{}".format(i) for i in range(num_cells)]] + \
            [["cell_{}_block_{}".format(i, j) for j in range(num_channels[i])] for i in range(num_cells)], [])
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, [])
        self.num_cells = num_cells
        self.num_channels = num_channels
        self.width_choice = width_choice
        self.depth_choice = depth_choice

    def __getstate__(self):
        state = super(MNasNetOFASearchSpace, self).__getstate__().copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        super(MNasNetOFASearchSpace, self).__setstate__(state)
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, [])

    def genotype(self, arch):
        geno_arch = arch[0] + arch[1]
        return self.genotype_type(**dict(zip(self.block_names, geno_arch)))

    def rollout_from_genotype(self, genotype):
        genotype_list = list(genotype._asdict().values())
        return MNasNetOFARollout(
            genotype_list, {}, self)      

    def plot_arch(self, genotypes, filename, label, **kwargs):
        raise NotImplementedError()

    def random_sample(self):
        return MNasNetOFARollout(MNasNetOFARollout.random_sample_arch(
               self.num_channels, self.num_cells,
               self.width_choice, self.depth_choice), info={}, search_space=self)

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    @classmethod
    def supported_rollout_types(cls):
        return ["mnasnet_ofa"]


class MNasNetOFARollout(Rollout):
    NAME = "mnasnet_ofa"
    channel = None

    @property
    def depth(self):
        return self.arch[0]

    @property
    def width(self):
        if self.channel == None:
            self.channel = []
            index = 0
            for c in self.search_space.num_channels:
                c_arch = []
                for i in range(c):
                    c_arch.append(self.arch[1][index])
                    index += 1
                self.channel.append(c_arch)
        return self.channel

    @classmethod
    def random_sample_arch(cls, num_channels, num_cells, width_choice, depth_choice):
        arch = []
        arch += [list(np.random.choice(depth_choice, size=num_cells))]
        arch += [list(np.random.choice(width_choice, size=sum(num_channels)))]
        return arch
            
        
         
