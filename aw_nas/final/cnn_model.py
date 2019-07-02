# -*- coding: utf-8 -*-
"""
A cell-based model whose architecture is described by a genotype.
"""

from __future__ import print_function

import re
from collections import defaultdict

import six
import numpy as np
import torch
from torch import nn

from aw_nas import ops, utils
from aw_nas.common import genotype_from_str, group_and_sort_by_to_node
from aw_nas.final.base import FinalModel
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.utils.common_utils import Context

def _defaultdict_dict():
    return defaultdict(dict)

class AuxiliaryHead(nn.Module):
    def __init__(self, C_in, num_classes):
        super(AuxiliaryHead, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, inputs): #pylint: disable=arguments-differ
        inputs = self.features(inputs)
        inputs = self.classifier(inputs.view(inputs.size(0), -1))
        return inputs

class CNNGenotypeModel(FinalModel):
    NAME = "cnn_final_model"

    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(self, search_space, device, genotypes,
                 num_classes=10, init_channels=36, cell_channels=tuple(), stem_multiplier=3,
                 dropout_rate=0.1, dropout_path_rate=0.2,
                 auxiliary_head=False, auxiliary_cfg=None,
                 use_stem=True, cell_use_preprocess=True,
                 cell_pool_batchnorm=False, cell_group_kwargs=None,
                 schedule_cfg=None):
        super(CNNGenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotypes = list(genotype_from_str(genotypes, self.search_space)._asdict().items())
        self.genotypes_grouped = [group_and_sort_by_to_node(g[1]) for g in self.genotypes\
                                  if "concat" not in g[0]]

        self.num_classes = num_classes
        self.init_channels = init_channels
        self.cell_channels = cell_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem
        self.cell_use_preprocess = cell_use_preprocess
        self.cell_group_kwargs = cell_group_kwargs

        # training
        self.dropout_rate = dropout_rate
        self.dropout_path_rate = dropout_path_rate
        self.auxiliary_head = auxiliary_head

        # search space configs
        self._num_init = self.search_space.num_init_nodes
        self._cell_layout = self.search_space.cell_layout
        self._reduce_cgs = self.search_space.reduce_cell_groups
        self._num_layers = self.search_space.num_layers
        self._out_multiplier = self.search_space.num_steps
        expect(len(self.genotypes) == self.search_space.num_cell_groups,
               ("Config genotype cell group number({}) "
                "does not match search_space cell group number({})")\
               .format(len(self.genotypes), self.search_space.num_cell_groups))

        ## initialize sub modules
        if self.use_stem:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = nn.Sequential(
                nn.Conv2d(3, c_stem, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_stem)
            )
        else:
            c_stem = 3

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = [c_stem] * self._num_init
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]
        if self.cell_channels:
            expect(len(self.cell_channels) == len(strides),
                   ("Config cell channels({}) does not match search_space num layers({})"\
                    .format(len(self.cell_channels), self.search_space.num_layers)),
                   ConfigException)
        for i_layer, stride in enumerate(strides):
            if self.cell_channels:
                num_channels = self.cell_channels[i_layer]
            elif stride > 1:
                num_channels *= stride
            if cell_group_kwargs is not None:
                # support passing in different kwargs when instantializing
                # cell class for different cell groups
                kwargs = {k: v for k, v in cell_group_kwargs[self._cell_layout[i_layer]].items()}
            else:
                kwargs = {}
            cg_idx = self.search_space.cell_layout[i_layer]
            # A patch: Can specificy input/output channels by hand in configuration,
            # instead of relying on the default
            # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
            _num_channels = num_channels if "C_in" not in kwargs \
                            else kwargs.pop("C_in")
            _num_out_channels = num_channels if "C_out" not in kwargs \
                                else kwargs.pop("C_out")
            cell = CNNGenotypeCell(self.search_space,
                                   self.genotypes_grouped[cg_idx],
                                   layer_index=i_layer,
                                   num_channels=_num_channels,
                                   num_out_channels=_num_out_channels,
                                   prev_num_channels=tuple(prev_num_channels),
                                   stride=stride,
                                   prev_strides=[1] * self._num_init + strides[:i_layer],
                                   use_preprocess=cell_use_preprocess,
                                   pool_batchnorm=cell_pool_batchnorm,
                                   **kwargs)
            prev_num_channels.append(_num_out_channels * self._out_multiplier)
            prev_num_channels = prev_num_channels[1:]
            self.cells.append(cell)

            if i_layer == (2 * self._num_layers) // 3 and self.auxiliary_head:
                self.auxiliary_net = AuxiliaryHead(prev_num_channels[-1],
                                                   num_classes, **(auxiliary_cfg or {}))

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(prev_num_channels[-1],
                                    self.num_classes)

        self.to(self.device)

    def forward(self, inputs): #pylint: disable=arguments-differ
        if self.use_stem:
            stemed = self.stem(inputs)
        else:
            stemed = inputs
        states = [stemed] * self._num_init

        for layer_idx, cell in enumerate(self.cells):
            states.append(cell(states, self.dropout_path_rate))
            states = states[1:]
            if layer_idx == 2 * self._num_layers // 3:
                if self.auxiliary_head and self.training:
                    logits_aux = self.auxiliary_net(states[-1])

        out = self.global_pooling(states[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.auxiliary_head and self.training:
            return logits, logits_aux
        return logits

    def forward_one_step(self, context, inputs, **kwargs):
        """
        NOTE: forward_one_step do not forward the auxliary head now!
        """
        assert (context is None) + (inputs is None) == 1
        # stem
        if inputs is not None:
            if self.use_stem:
                stemed = self.stem(inputs)
            else:
                stemed = inputs
            context = Context(previous_cells=[stemed], current_cell=[])
            return stemed, context

        cur_cell_ind, _ = context.next_step_index

        # final: pooling->dropout->classifier
        if cur_cell_ind == self._num_layers:
            out = self.global_pooling(context.previous_cells[-1])
            out = self.dropout(out)
            logits = self.classifier(out.view(out.size(0), -1))
            context.previous_cells.append(logits)
            return logits, context

        # cells
        return self.cells[cur_cell_ind].forward_one_step(context, self.dropout_path_rate, **kwargs)

    def forward_one_step_callback(self, inputs, callback):
        # forward stem
        stemed, context = self.forward_one_step(context=None, inputs=inputs)
        callback(stemed, context)

        # forward the cells
        for _ in range(0, self.search_space.num_layers):
            num_steps = self.search_space.num_steps + self.search_space.num_init_nodes + 1
            for _ in range(num_steps):
                while True: # call `forward_one_step` until this step ends
                    step_state, context = self.forward_one_step(context, inputs=None)
                    callback(step_state, context)
                    if context.is_end_of_step:
                        break
            # end of cell (every cell has the same number of num_steps)
        # final forward
        logits, context = self.forward_one_step(context, inputs=None)
        callback(logits, context)
        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _is_reduce(self, layer_idx):
        return self._cell_layout[layer_idx] in self._reduce_cgs

class CNNGenotypeCell(nn.Module):
    def __init__(self, search_space, genotype_grouped, layer_index, num_channels, num_out_channels,
                 prev_num_channels, stride, prev_strides, use_preprocess, pool_batchnorm,
                 **op_kwargs):
        super(CNNGenotypeCell, self).__init__()
        self.search_space = search_space
        self.genotype_grouped = genotype_grouped
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index
        self.use_preprocess = use_preprocess
        self.pool_batchnorm = pool_batchnorm
        self.op_kwargs = op_kwargs

        self._steps = self.search_space.num_steps
        self._num_init = self.search_space.num_init_nodes
        self._primitives = self.search_space.shared_primitives

        self.preprocess_ops = nn.ModuleList()
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[:len(prev_num_channels)])
        for prev_c, prev_s in zip(prev_num_channels, prev_strides):
            if not self.use_preprocess:
                # stride/channel not handled!
                self.preprocess_ops.append(ops.Identity())
                continue
            if prev_s > 1:
                # need skip connection, and is not the connection from the input image
                preprocess = ops.FactorizedReduce(C_in=prev_c,
                                                  C_out=num_channels,
                                                  stride=prev_s,
                                                  affine=True)
            else: # prev_c == _steps * num_channels or inputs
                preprocess = ops.ReLUConvBN(C_in=prev_c,
                                            C_out=num_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            affine=True)
            self.preprocess_ops.append(preprocess)
        assert len(self.preprocess_ops) == self._num_init

        self.edges = defaultdict(_defaultdict_dict)
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for _, conns in self.genotype_grouped:
            for op_type, from_, to_ in conns:
                stride = self.stride if from_ < self._num_init else 1
                op = ops.get_op(op_type)(
                    num_channels, num_out_channels, stride, True, **op_kwargs)
                if self.pool_batchnorm and "pool" in op_type:
                    op = nn.Sequential(op, nn.BatchNorm2d(num_out_channels, affine=False))
                if op_type not in self.edges[from_][to_]:
                    self.edges[from_][to_][op_type] = op
                    self.edge_mod.add_module("f_{}_t_{}-{}".format(from_, to_, op_type), op)

        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)-([a-z0-9_-]+)")

    def forward(self, inputs, dropout_path_rate): #pylint: disable=arguments-differ
        assert self._num_init == len(inputs)
        states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]

        for to_, connections in self.genotype_grouped:
            state_to_ = 0.
            for op_type, from_, _ in connections:
                op = self.edges[from_][to_][op_type]
                out = op(states[from_])
                if self.training and dropout_path_rate > 0:
                    if not isinstance(op, ops.Identity):
                        out = utils.drop_path(out, dropout_path_rate)
                state_to_ = state_to_ + out
            states.append(state_to_)

        return torch.cat(states[-self.search_space.num_steps:], dim=1)

    def forward_one_step(self, context, dropout_path_rate):
        to_ = cur_step = context.next_step_index[1]
        if cur_step < self._num_init: # `self._num_init` preprocess steps
            ind = len(context.previous_cells) - (self._num_init - cur_step)
            ind = max(ind, 0)
            state = self.preprocess_ops[cur_step](context.previous_cells[ind])
            context.current_cell.append(state)
        elif cur_step < self._num_init + self._steps: # the following steps
            conns = self.genotype_grouped[cur_step - self._num_init][1]
            op_ind, current_op = context.next_op_index
            if op_ind == len(conns):
                # all connections added to context.previous_ops, sum them up
                state = sum([st for st in context.previous_op])
                context.current_cell.append(state)
                context.previous_op = []
            else:
                op_type, from_, _ = conns[op_ind]
                op = self.edges[from_][to_][op_type]
                state, context = op.forward_one_step(
                    context=context,
                    inputs=context.current_cell[from_] if current_op == 0 else None)
                if self.training and dropout_path_rate > 0:
                    if not isinstance(op, ops.Identity):
                        context.last_state = state = utils.drop_path(state, dropout_path_rate)
        else: # final concat
            state = torch.cat(context.current_cell[-self.search_space.num_steps:], dim=1)
            context.current_cell = []
            context.previous_cells.append(state)
        return state, context

    def on_replicate(self):
        # Although this edges is easy to understand, when paralleized,
        # the reference relationship between `self.edge` and modules under `self.edge_mod`
        # will not get updated automatically.

        # So, after each replicate, we should initialize a new edges dict
        # and update the reference manually.
        self.edges = defaultdict(_defaultdict_dict)
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_, op_type = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)][op_type] = edge_mod
