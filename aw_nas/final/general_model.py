# -*- coding: utf-8 -*-
"""
A general model whose architecture is described by a genotype.
"""

from __future__ import print_function

import re
from collections import defaultdict

import copy
import six
import numpy as np
import torch
from torch import nn

from aw_nas import utils
from aw_nas.ops import get_op
from aw_nas.common import genotype_from_str, group_and_sort_by_to_node
from aw_nas.final.base import FinalModel
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.utils.common_utils import Context


class GeneralGenotypeModel(FinalModel):
    NAME = "general_final_model"

    def __init__(self, search_space, device, genotypes, schedule_cfg=None):
        super(GeneralGenotypeModel, self).__init__(schedule_cfg)
        self.search_space = search_space
        self.device = device

        if isinstance(genotypes, str):
            self.genotypes = list(genotype_from_str(genotypes, self.search_space))
        else:
            self.genotypes = copy.deepcopy(genotypes)
        model = []
        for geno in self.genotypes:
            op = geno.pop("prim_type")
            geno.pop("spatial_size")
            model += [get_op(op)(**geno)]
        self.model = nn.Sequential(*model)

        self.model.apply(utils.init_weight)

        self.model.to(self.device)

    def forward(self, inputs):
        return self.model(inputs)

    @classmethod
    def supported_data_types(cls):
        return ["image", "sequence"]

    def layer_idx_to_named_modules(self, idx):
        prefix = f"model.{idx}"
        m = self
        for name in prefix.split('.'):
            m = getattr(m, name)
        for n, sub_m in m.named_modules():
            if not n:
                yield prefix
            yield '.'.join([prefix, n])