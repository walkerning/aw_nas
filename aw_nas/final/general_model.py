# -*- coding: utf-8 -*-
"""
A general model whose architecture is described by a genotype.
"""

from __future__ import print_function

import re
import timeit
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
from aw_nas.utils.common_utils import Context, tick, Ticker


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
        for geno in copy.deepcopy(self.genotypes):
            op = geno.pop("prim_type")
            geno.pop("spatial_size")
            model += [get_op(op)(**geno)]

        self.model = nn.ModuleList(model)

        self.model.apply(utils.init_weight)
        self.model.to(self.device)

    @tick("_forward_elapse")
    def forward(self, inputs, callback=None):
        for sub_m in self.model:
            out = sub_m(inputs)
            if callback:
                callback(inputs, out, sub_m)
            inputs = out
        return out

    def analyze_elapses(self, inputs, use_cuda=True, forward_time=100):
        for _ in range(2):
            self.forward(inputs)
            torch.cuda.synchronize()

        def callback(inputs, out, model):
            if use_cuda:
                torch.cuda.synchronize()
            elapses.append(ticker.tick() * 1000)

        all_elapses = []
        async_elapse = 0.
        sync_elapse = 0.
        for _ in range(forward_time):
            ticker = Ticker("general_forward")
            elapses = []
            self.forward(inputs, callback=callback)
            self.forward(inputs)
            all_elapses.append(elapses)
            async_elapse += self._forward_elapse
            sync_elapse += ticker.total_time * 1000
        mean_elapse = np.array(all_elapses).mean(axis=0)
        async_elapse /= forward_time
        sync_elapse /= forward_time

        genotypes = [{"elapse": elapse, **geno} for elapse, geno in zip(mean_elapse, self.genotypes)]

        return {"primitives": genotypes, "async_elapse": async_elapse, "sync_elapse": sync_elapse}


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