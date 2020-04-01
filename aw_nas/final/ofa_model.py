# -*- coding: utf-8 -*-
"""
A cell-based model whose architecture is described by a genotype.
"""

from __future__ import print_function

import os
from collections import defaultdict, OrderedDict

import numpy as np
import six
import torch
from torch import nn

from aw_nas import ops, utils
from aw_nas.common import genotype_from_str, group_and_sort_by_to_node
from aw_nas.final.base import FinalModel
from aw_nas.ops import *
from aw_nas.utils.common_utils import Context, timer
from aw_nas.utils.exception import ConfigException, expect


class OFAGenotypeModel(FinalModel):
    def __init__(self, search_space, device, genotypes,
                 num_classes=10, layer_channels=tuple(), strides=tuple(),
                schedule_cfg=None):
        super(OFAGenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotypes = list(genotype_from_str(genotypes, self.search_space)._asdict().values())
 
        self.num_classes = num_classes
        self.layer_channels = layer_channels
        self.strides = strides

        self.depth, self.width, self.kernel = self.parse(self.genotypes)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def set_hook(self):
        for name, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += 2* inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    outputs.size(2) * outputs.size(3) / module.groups
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def parse(self, genotype):
        depth = genotype[:len(self.search_space.num_cell_groups)]
        width = []
        kernel = []
        ind = len(self.search_space.num_cell_groups)
        for i, max_depth in zip(depth, self.search_space.num_cell_groups):
            width_list = []
            kernel_list = []
            for j in range(max_depth):
                if j < i:
                    try:
                        width_list.append(genotype[ind][0])
                        kernel_list.append(genotype[ind][1])
                    except Exception as identifier:
                        width_list.append(genotype[ind])
                        kernel_list.append(3)
                ind += 1
            width.append(width_list)
            kernel.append(kernel_list)
        return depth, width, kernel