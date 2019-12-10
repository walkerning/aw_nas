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
from aw_nas.rollout.base import BaseRollout
from aw_nas.common import get_genotype_substr, genotype_from_str
from aw_nas.utils import logger as _logger

class MnasnetOFASearchSpace(SearchSpace):
    NAME = "mnasnet_ofa"

    def __init__(self, width_choice, depth_choice,
                 num_cells=6, num_channels=[3,3,3,2,4,1]):
        super(MnasnetOFASearchSpace, self).__init__()
        self.genotype_type_name = "MnasnetOFAGenotype"
        self.block_names = sum(
            [["cell_{}".format(i)] for i in range(num_cells)] + \
            [["cell_{}_block_{}".format(i, j)] for j in range(num_channels[i]) for i in range(num_cells)], [])
        self.genotype = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, [])
        self.num_cells = num_cells
        self.num_channels = num_channels
        self.width_choice = width_choice
        self.depth_choice = depth_choice

    def __getstate__(self):
        state = super(MnasnetOFASearchSpace, self).__getstate__().copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        super(MnasnetOFASearchSpace, self).__setstate__(state)
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, [])

    def genotype(self, arch):
        geno_arch = []
        for i_arch, n_cs in enumerate(arch):
            geno_arch.append(list(n_cs))
        return self.genotype_type(**dict(zip(self.block_names, extend_arch)))

    def rollout_from_genotype(self, genotype):
        genotype_list = list(genotype._asdict().values())
        return MnasnetOFARollout(
            genotype_list, {}, self)      

    def plot_arch(self, genotypes, filename, label, **kwargs):
        raise NotImplementedError()

    def random_sample(self):
        return MnasnetOFARollout(MnasnetOFARollout.random_sample_arch(
               self.num_channels, self.num_cells,
               self.width_choice, self.depth_choice))

    def distance(self, arch1, arch2):
        raise NotImplementedError()

class MnasnetOFARollout(Rollout):
    NAME = "mnasnet_ofa"
    self.width = None

    @property
    def depth(self):
        return self.arch[:self.search_space.num_cells]

    @property
    def width(self):
        if self.width == None:
            self.width = []
            index = 0
            for c in num_channels:
                c_arch = []
                for i in range(c):
                    c_arch.append(self.arch[self.search_space.num_cells + index])
                    index += 1
                self.width.append(c_arch)
        return self.width

    @classmethod
    def random_sample_arch(cls, num_channels, num_cells, width_choice, depth_choice):
        arch = []
        arch += [list(np.random.choice(depth_choice, size=num_cells))]
        arch += [list(np.random.randint(width_choice, size=sum(num_channels)))]
        return arch
            
        
         
