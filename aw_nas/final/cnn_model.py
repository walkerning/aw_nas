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

def _defaultdict_1():
    return defaultdict(dict)

def _defaultdict_2():
    return defaultdict(_defaultdict_1)

def _defaultdict_3():
    return defaultdict(_defaultdict_2)

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
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, inputs): #pylint: disable=arguments-differ
        inputs = self.features(inputs)
        inputs = self.classifier(inputs.view(inputs.size(0), -1))
        return inputs

class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C_in, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class CNNGenotypeModel(FinalModel):
    NAME = "cnn_final_model"

    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(self, search_space, device, genotypes,
                 num_classes=10, init_channels=36, layer_channels=tuple(), stem_multiplier=3,
                 dropout_rate=0.1, dropout_path_rate=0.2,
                 auxiliary_head=False, auxiliary_cfg=None,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 no_fc=False,
                 cell_use_preprocess=True,
                 cell_pool_batchnorm=False, cell_group_kwargs=None,
                 cell_independent_conn=False,
                 cell_use_shortcut=False,
                 cell_shortcut_op_type="skip_connect",
                 cell_preprocess_stride="skip_connect",
                 cell_preprocess_normal="relu_conv_bn_1x1",
                 schedule_cfg=None):
        super(CNNGenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotypes = list(genotype_from_str(genotypes, self.search_space)._asdict().values())
        self.genotypes_grouped = list(zip(
            [group_and_sort_by_to_node(conns)
             for conns in self.genotypes[:self.search_space.num_cell_groups]],
            self.genotypes[self.search_space.num_cell_groups:]))
        # self.genotypes_grouped = [group_and_sort_by_to_node(g[1]) for g in self.genotypes\
        #                           if "concat" not in g[0]]

        self.num_classes = num_classes
        self.init_channels = init_channels
        self.layer_channels = layer_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem
        self.cell_use_preprocess = cell_use_preprocess
        self.cell_group_kwargs = cell_group_kwargs
        self.cell_independent_conn = cell_independent_conn
        self.no_fc = no_fc

        # training
        self.dropout_rate = dropout_rate
        self.dropout_path_rate = dropout_path_rate
        self.auxiliary_head = auxiliary_head

        # search space configs
        self._num_init = self.search_space.num_init_nodes
        self._cell_layout = self.search_space.cell_layout
        self._reduce_cgs = self.search_space.reduce_cell_groups
        self._num_layers = self.search_space.num_layers
        expect(len(self.genotypes_grouped) == self.search_space.num_cell_groups,
               ("Config genotype cell group number({}) "
                "does not match search_space cell group number({})")\
               .format(len(self.genotypes_grouped), self.search_space.num_cell_groups))

        ## initialize sub modules
        if not self.use_stem:
            c_stem = 3
            init_strides = [1] * self._num_init
        elif isinstance(self.use_stem, (list, tuple)):
            self.stems = []
            c_stem = self.stem_multiplier * self.init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stems.append(ops.get_op(stem_type)(
                    c_in, c_stem, stride=stem_stride, affine=stem_affine))
            self.stems = nn.ModuleList(self.stems)
            init_strides = [stem_stride] * self._num_init
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(3, c_stem,
                                                  stride=stem_stride, affine=stem_affine)
            init_strides = [1] * self._num_init

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = [c_stem] * self._num_init
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]
        if self.layer_channels:
            expect(len(self.layer_channels) == len(strides) + 1,
                   ("Config cell channels({}) does not match search_space num layers + 1 ({})"\
                    .format(len(self.layer_channels), self.search_space.num_layers + 1)),
                   ConfigException)
        for i_layer, stride in enumerate(strides):
            if self.layer_channels:
                # input and output channels of every layer is specified
                num_channels = self.layer_channels[i_layer]
                num_out_channels = self.layer_channels[i_layer + 1]
            else:
                if stride > 1:
                    num_channels *= stride
                num_out_channels = num_channels
            if cell_group_kwargs is not None:
                # support passing in different kwargs when instantializing
                # cell class for different cell groups
                # Can specificy input/output channels by hand in configuration,
                # instead of relying on the default
                # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
                kwargs = {k: v for k, v in cell_group_kwargs[self._cell_layout[i_layer]].items()}
                if "C_in" in kwargs:
                    num_channels = kwargs.pop("C_in")
                if "C_out" in kwargs:
                    num_out_channels = kwargs.pop("C_out")
            else:
                kwargs = {}
            cg_idx = self.search_space.cell_layout[i_layer]

            cell = CNNGenotypeCell(self.search_space,
                                   self.genotypes_grouped[cg_idx],
                                   layer_index=i_layer,
                                   num_channels=num_channels,
                                   num_out_channels=num_out_channels,
                                   prev_num_channels=tuple(prev_num_channels),
                                   stride=stride,
                                   prev_strides=init_strides + strides[:i_layer],
                                   use_preprocess=cell_use_preprocess,
                                   pool_batchnorm=cell_pool_batchnorm,
                                   independent_conn=cell_independent_conn,
                                   preprocess_stride=cell_preprocess_stride,
                                   preprocess_normal=cell_preprocess_normal,
                                   use_shortcut=cell_use_shortcut,
                                   shortcut_op_type=cell_shortcut_op_type,
                                   **kwargs)
            # TODO: support specify concat explicitly
            prev_num_channel = cell.num_out_channel()
            prev_num_channels.append(prev_num_channel)
            prev_num_channels = prev_num_channels[1:]
            self.cells.append(cell)

            if i_layer == (2 * self._num_layers) // 3 and self.auxiliary_head:
                if auxiliary_head == "imagenet":
                    self.auxiliary_net = AuxiliaryHeadImageNet(
                        prev_num_channels[-1], num_classes, **(auxiliary_cfg or {}))
                else:
                    self.auxiliary_net = AuxiliaryHead(
                        prev_num_channels[-1], num_classes, **(auxiliary_cfg or {}))

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        if self.no_fc:
            self.classifier = ops.Identity()
        else:
            self.classifier = nn.Linear(prev_num_channels[-1],
                                    self.num_classes)
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def set_hook(self):
        for name, module in self.named_modules():
            if "auxiliary" in name:
                continue
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

    def forward(self, inputs): #pylint: disable=arguments-differ
        if not self.use_stem:
            stemed = inputs
            states = [inputs] * self._num_init
        elif isinstance(self.use_stem, (list, tuple)):
            states = []
            stemed = inputs
            for stem in self.stems:
                stemed = stem(stemed)
                states.append(stemed)
        else:
            stemed = self.stem(inputs)
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

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops/1.e6)
            self._flops_calculated = True

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
            context = Context(self._num_init, self._num_layers,
                              previous_cells=[stemed], current_cell=[])
            context.last_conv_module = self.stem.get_last_conv_module()
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
        _, context = self.forward_one_step(context=None, inputs=inputs)
        callback(context.last_state, context)

        # forward the cells
        for i_layer in range(0, self.search_space.num_layers):
            num_steps = self.search_space.get_layer_num_steps(i_layer) + \
                        self.search_space.num_init_nodes + 1
            for _ in range(num_steps):
                while True: # call `forward_one_step` until this step ends
                    _, context = self.forward_one_step(context, inputs=None)
                    callback(context.last_state, context)
                    if context.is_end_of_cell or context.is_end_of_step:
                        break
            # end of cell (every cell has the same number of num_steps)
        # final forward
        _, context = self.forward_one_step(context, inputs=None)
        callback(context.last_state, context)
        return context.last_state

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _is_reduce(self, layer_idx):
        return self._cell_layout[layer_idx] in self._reduce_cgs

class CNNGenotypeCell(nn.Module):
    def __init__(self, search_space, genotype_grouped, layer_index, num_channels, num_out_channels,
                 prev_num_channels, stride, prev_strides, use_preprocess, pool_batchnorm,
                 independent_conn, preprocess_stride, preprocess_normal,
                 use_shortcut, shortcut_op_type,
                 **op_kwargs):
        super(CNNGenotypeCell, self).__init__()
        self.search_space = search_space
        self.conns_grouped, self.concat_nodes = genotype_grouped
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index
        self.use_preprocess = use_preprocess
        self.pool_batchnorm = pool_batchnorm
        self.independent_conn = independent_conn
        self.op_kwargs = op_kwargs
        self.use_shortcut = use_shortcut
        self.shortcut_op_type = shortcut_op_type

        self._steps = self.search_space.get_layer_num_steps(layer_index)
        self._num_init = self.search_space.num_init_nodes
        self._primitives = self.search_space.shared_primitives

        # initialize self.concat_op, self._out_multiplier (only used for discrete super net)
        self.concat_op = ops.get_concat_op(self.search_space.concat_op)
        if not self.concat_op.is_elementwise:
            expect(not self.search_space.loose_end,
                   "For shared weights weights manager, when non-elementwise concat op do not "
                   "support loose-end search space")
            self._out_multiplier = self._steps if not self.search_space.concat_nodes \
                                  else len(self.search_space.concat_nodes)
        else:
            # elementwise concat op. e.g. sum, mean
            self._out_multiplier = 1

        self.preprocess_ops = nn.ModuleList()
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[:len(prev_num_channels)])
        for prev_c, prev_s in zip(prev_num_channels, prev_strides):
            # print("cin: {}, cout: {}, stride: {}".format(prev_c, num_channels, prev_s))
            if not self.use_preprocess:
                # stride/channel not handled!
                self.preprocess_ops.append(ops.Identity())
                continue
            if prev_s > 1:
                # need skip connection, and is not the connection from the input image
                # ops.FactorizedReduce(C_in=prev_c,
                preprocess = ops.get_op(preprocess_stride)(
                    C=prev_c,
                    C_out=num_channels,
                    stride=int(prev_s),
                    affine=True)
            else: # prev_c == _steps * num_channels or inputs
                preprocess = ops.get_op(preprocess_normal)(
                    C=prev_c,
                    C_out=num_channels,
                    stride=1,
                    affine=True)
            self.preprocess_ops.append(preprocess)
        # only apply the inter-cell shortcut between last layer
        if self.use_shortcut:
            self.shortcut_reduction_op = ops.get_op(self.shortcut_op_type)(
                C=prev_num_channels[-1], C_out=self.num_out_channel(),
                stride=self.stride, affine=True)

        assert len(self.preprocess_ops) == self._num_init

        self.edges = _defaultdict_3()
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for _, conns in self.conns_grouped:
            for op_type, from_, to_ in conns:
                stride = self.stride if from_ < self._num_init else 1
                op = ops.get_op(op_type)(
                    num_channels, num_out_channels, stride, True, **op_kwargs)
                if self.pool_batchnorm and "pool" in op_type:
                    op = nn.Sequential(op, nn.BatchNorm2d(num_out_channels, affine=False))
                index = len(self.edges[from_][to_][op_type])
                if index == 0 or self.independent_conn:
                    # there is no this connection already established,
                    # or use indepdent connection for exactly the same (from, to, op_type)
                    self.edges[from_][to_][op_type][index] = op
                    self.edge_mod.add_module("f_{}_t_{}-{}-{}".format(
                        from_, to_, op_type, index), op)

        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)-([a-z0-9_-]+)-([0-9])")

    def num_out_channel(self):
        return self.num_out_channels * self._out_multiplier

    def forward(self, inputs, dropout_path_rate): #pylint: disable=arguments-differ
        assert self._num_init == len(inputs)
        states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]

        _num_conn = defaultdict(int)
        for to_, connections in self.conns_grouped:
            state_to_ = 0.
            for op_type, from_, _ in connections:
                conn_ind = 0 if not self.independent_conn else _num_conn[(from_, to_, op_type)]
                op = self.edges[from_][to_][op_type][conn_ind]
                _num_conn[(from_, to_, op_type)] += 1
                out = op(states[from_])
                if self.training and dropout_path_rate > 0:
                    if not isinstance(op, ops.Identity):
                        out = utils.drop_path(out, dropout_path_rate)
                state_to_ = state_to_ + out
            states.append(state_to_)

        out = self.concat_op([states[ind] for ind in self.concat_nodes])
        if self.use_shortcut and self.layer_index != 0:
            out = out + self.shortcut_reduction_op(inputs[-1])

        return out

    def forward_one_step(self, context, dropout_path_rate):
        to_ = cur_step = context.next_step_index[1]
        if cur_step == 0:
            context._num_conn[self] = defaultdict(int)
        if cur_step < self._num_init: # `self._num_init` preprocess steps
            ind = len(context.previous_cells) - (self._num_init - cur_step)
            ind = max(ind, 0)
            # state = self.preprocess_ops[cur_step](context.previous_cells[ind])
            # context.current_cell.append(state)
            # context.last_conv_module = self.preprocess_ops[cur_step].get_last_conv_module()
            current_op = context.next_op_index[1]
            state, context = self.preprocess_ops[cur_step].forward_one_step(
                context=context,
                inputs=context.previous_cells[ind] if current_op == 0 else None)
            if context.next_op_index[1] == 0: # this preprocess op finish, append to `current_cell`
                assert len(context.previous_op) == 1
                context.current_cell.append(context.previous_op[0])
                context.previous_op = []
                context.last_conv_module = self.preprocess_ops[cur_step].get_last_conv_module()
        elif cur_step < self._num_init + self._steps: # the following steps
            conns = self.conns_grouped[cur_step - self._num_init][1]
            op_ind, current_op = context.next_op_index
            if op_ind == len(conns):
                # all connections added to context.previous_ops, sum them up
                state = sum([st for st in context.previous_op])
                context.current_cell.append(state)
                context.previous_op = []
            else:
                op_type, from_, _ = conns[op_ind]
                conn_ind = 0 if not self.independent_conn else \
                           context._num_conn[self][(from_, to_, op_type)]
                op = self.edges[from_][to_][op_type][conn_ind]
                state, context = op.forward_one_step(
                    context=context,
                    inputs=context.current_cell[from_] if current_op == 0 else None)
                if self.training and dropout_path_rate > 0:
                    if not isinstance(op, ops.Identity):
                        context.last_state = state = utils.drop_path(state, dropout_path_rate)
                if context.next_op_index[0] != op_ind:
                    # this op finish
                    context._num_conn[self][(from_, to_, op_type)] += 1
        else: # final concat
            state = self.concat_op([context.current_cell[ind] for ind in self.concat_nodes])
            context.current_cell = []
            context.previous_cells.append(state)
        return state, context

    def on_replicate(self):
        # Although this edges is easy to understand, when paralleized,
        # the reference relationship between `self.edge` and modules under `self.edge_mod`
        # will not get updated automatically.

        # So, after each replicate, we should initialize a new edges dict
        # and update the reference manually.
        self.edges = _defaultdict_3()
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_, op_type, index = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)][op_type][int(index)] = edge_mod
