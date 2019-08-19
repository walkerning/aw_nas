# -*- coding: utf-8 -*-
"""
A CNN model with densenet-like connections, whose architecture is described by a genotype.
"""

import numpy as np
from torch import nn

from aw_nas import ops
from aw_nas.ops.baseline_ops import DenseBlock, Transition
from aw_nas.final.base import FinalModel
from aw_nas.common import genotype_from_str

class DenseGenotypeModel(FinalModel):
    NAME = "dense_final_model"

    def __init__(self, search_space, device, genotypes,
                 num_classes=10,
                 dropout_rate=0.0,
                 schedule_cfg=None):
        super(DenseGenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotypes = list(genotype_from_str(genotypes, self.search_space)._asdict().values())
        self.num_classes = num_classes

        # training
        self.dropout_rate = dropout_rate

        self._num_blocks = self.search_space.num_dense_blocks
        # build model
        self.stem = nn.Conv2d(3, self.genotypes[0], kernel_size=3)

        self.dense_blocks = []
        self.trans_blocks = []
        last_channel = self.genotypes[0]
        for i_block in range(self._num_blocks):
            growths = self.genotypes[1 + i_block * 2]
            self.dense_blocks.append(self._new_dense_block(last_channel, growths))
            last_channel = int(last_channel + np.sum(growths))
            if i_block != self._num_blocks - 1:
                out_c = self.genotypes[2 + i_block * 2]
                self.trans_blocks.append(self._new_transition_block(last_channel, out_c))
                last_channel = out_c
        self.dense_blocks = nn.ModuleList(self.dense_blocks)
        self.trans_blocks = nn.ModuleList(self.trans_blocks)

        self.final_bn = nn.BatchNorm2d(last_channel)
        self.final_relu = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()

        self.classifier = nn.Linear(last_channel, self.num_classes)

        self.to(self.device)

    def _new_dense_block(self, last_channel, growths):
        mini_blocks = []
        for growth in growths:
            out_c = last_channel + growth
            mini_blocks.append(DenseBlock(last_channel, out_c, stride=1, affine=True,
                                          bc_mode=self.search_space.bc_mode,
                                          bc_ratio=self.search_space.bc_ratio))
            last_channel = out_c
        return nn.Sequential(*mini_blocks)

    def _new_transition_block(self, last_channel, out_c): #pylint: disable=no-self-use
        return Transition(last_channel, out_c, stride=2, affine=True)

    # ---- APIs ----
    def forward(self, inputs):
        out = self.stem(inputs)
        for i_block in range(self._num_blocks):
            out = self.dense_blocks[i_block](out)
            if i_block != self._num_blocks - 1:
                out = self.trans_blocks[i_block](out)

        out = self.final_relu(self.final_bn(out))
        out = self.dropout(self.global_pooling(out))
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]
