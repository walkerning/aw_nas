# -*- coding: utf-8 -*-

import re
from collections import defaultdict

import six
import numpy as np
import torch
from torch import nn

from aw_nas import ops
from aw_nas.weights_manager.base import BaseWeightsManager
from aw_nas.weights_manager.wrapper import BaseBackboneWeightsManager
from aw_nas.utils.common_utils import Context
from aw_nas.utils.exception import expect, ConfigException


class SharedNet(BaseBackboneWeightsManager, nn.Module):
    def __init__(self, search_space, device, rollout_type,
                 cell_cls, op_cls,
                 gpus=tuple(),
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 preprocess_op_type=None,
                 cell_use_preprocess=True,
                 cell_group_kwargs=None,
                 cell_use_shortcut=False,
                 bn_affine=False,
                 cell_shortcut_op_type="skip_connect"):
        super(SharedNet, self).__init__(search_space, device, rollout_type)
        nn.Module.__init__(self)

        # optionally data parallelism in SharedNet
        self.gpus = gpus

        self.num_classes = num_classes
        # init channel number of the first cell layers,
        # x2 after every reduce cell
        self.init_channels = init_channels
        # channels of stem conv / init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem
        # possible cell group kwargs
        self.cell_group_kwargs = cell_group_kwargs
        # possible inter-cell shortcut
        self.cell_use_shortcut = cell_use_shortcut
        self.cell_shortcut_op_type = cell_shortcut_op_type

        # training
        self.max_grad_norm = max_grad_norm
        self.dropout_rate = dropout_rate

        # search space configs
        self._num_init = self.search_space.num_init_nodes
        self._cell_layout = self.search_space.cell_layout
        self._reduce_cgs = self.search_space.reduce_cell_groups
        self._num_layers = self.search_space.num_layers

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
        self.all_num_channels = [c_stem]
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]

        for i_layer, stride in enumerate(strides):
            if stride > 1:
                num_channels *= stride
            if cell_group_kwargs is not None:
                # support passing in different kwargs when instantializing
                # cell class for different cell groups
                kwargs = {k: v for k, v in cell_group_kwargs[self._cell_layout[i_layer]].items()}
            else:
                kwargs = {}
            # A patch: Can specificy input/output channels by hand in configuration,
            # instead of relying on the default
            # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
            _num_channels = num_channels if "C_in" not in kwargs \
                            else kwargs.pop("C_in")
            _num_out_channels = num_channels if "C_out" not in kwargs \
                                else kwargs.pop("C_out")
            cell = cell_cls(op_cls,
                            self.search_space,
                            layer_index=i_layer,
                            num_channels=_num_channels,
                            num_out_channels=_num_out_channels,
                            prev_num_channels=tuple(prev_num_channels),
                            stride=stride,
                            prev_strides=init_strides + strides[:i_layer],
                            use_preprocess=cell_use_preprocess,
                            preprocess_op_type=preprocess_op_type,
                            use_shortcut=cell_use_shortcut,
                            shortcut_op_type=cell_shortcut_op_type,
                            bn_affine=bn_affine,
                            **kwargs)
            prev_num_channel = cell.num_out_channel()
            prev_num_channels.append(prev_num_channel)
            # currently, add all stem and cell outputs
            self.all_num_channels.append(prev_num_channel)
            prev_num_channels = prev_num_channels[1:]
            self.cells.append(cell)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(prev_num_channel, self.num_classes)

        self.to(self.device)

    def _is_reduce(self, layer_idx):
        return self._cell_layout[layer_idx] in self._reduce_cgs

    def set_device(self, device):
        self.device = device
        self.to(device)

    def get_feature_channel_num(self, feature_levels=None):
        if feature_levels is None:
            return self.all_num_channels
        return [self.all_num_channels[level] for level in feature_levels]

    def extract_features(self, inputs, genotypes, keep_all=True, **kwargs):
        # only support use genotypes/arch
        # the two subclasses (diff_)supernet support using rollout
        # * genotypes: for the commonly-used classification NAS API
        # * rollout (discrete) / arch (diff):
        #    used as the backbone weights manager in the wrapper weights manager

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

        if keep_all:
            all_states = [stemed]

        for cg_idx, cell in zip(self._cell_layout, self.cells):
            genotype = genotypes[cg_idx]
            states.append(cell(states, genotype, **kwargs))
            states = states[1:]
            if keep_all:
                all_states.append(states[-1])
        return all_states if keep_all else states

    def forward(self, inputs, genotypes, **kwargs): #pylint: disable=arguments-differ
        states = self.extract_features(inputs, genotypes, keep_all=False, **kwargs)
        # classification head
        # for compatibility of old checkpoints,
        # we do not reuse aw_nas.weights_manager.wrapper.ClassificationHead
        out = self.global_pooling(states[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def forward_one_step(self, context, inputs, genotypes, **kwargs):
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
        cell_genotype = genotypes[self._cell_layout[cur_cell_ind]]
        return self.cells[cur_cell_ind].forward_one_step(context, cell_genotype, **kwargs)

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def save(self, path):
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def finalize(self, rollout):
        # TODO: implement this
        raise NotImplementedError()


class SharedCell(nn.Module):
    def __init__(self, op_cls, search_space, layer_index, num_channels, num_out_channels,
                 prev_num_channels, stride, prev_strides, use_preprocess, preprocess_op_type,
                 use_shortcut, shortcut_op_type, bn_affine,
                 **op_kwargs):
        super(SharedCell, self).__init__()
        self.search_space = search_space
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index
        self.use_preprocess = use_preprocess
        self.preprocess_op_type = preprocess_op_type
        self.use_shortcut = use_shortcut
        self.shortcut_op_type = shortcut_op_type
        self.bn_affine = bn_affine
        self.op_kwargs = op_kwargs

        self._steps = self.search_space.get_layer_num_steps(layer_index)
        self._num_init = self.search_space.num_init_nodes
        if not self.search_space.cellwise_primitives:
            # the same set of primitives for different cg group
            self._primitives = self.search_space.shared_primitives
        else:
            # different set of primitives for different cg group
            self._primitives = \
                self.search_space.cell_shared_primitives[self.search_space.cell_layout[layer_index]]

        # initialize self.concat_op, self._out_multiplier (only used for discrete super net)
        self.concat_op = ops.get_concat_op(self.search_space.concat_op)
        if not self.concat_op.is_elementwise:
            expect(not self.search_space.loose_end,
                   "For shared weights weights manager, when non-elementwise concat op do not "
                   "support loose-end search space")
            self._out_multipler = self._steps if not self.search_space.concat_nodes \
                                  else len(self.search_space.concat_nodes)
        else:
            # elementwise concat op. e.g. sum, mean
            self._out_multipler = 1

        self.preprocess_ops = nn.ModuleList()
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[:len(prev_num_channels)])
        for prev_c, prev_s in zip(prev_num_channels, prev_strides):
            if not self.use_preprocess:
                # stride/channel not handled!
                self.preprocess_ops.append(ops.Identity())
                continue
            if self.preprocess_op_type is not None:
                # specificy other preprocess op
                preprocess = ops.get_op(self.preprocess_op_type)(
                    C=prev_c, C_out=num_channels, stride=int(prev_s), affine=bn_affine)
            else:
                if prev_s > 1:
                    # need skip connection, and is not the connection from the input image
                    preprocess = ops.FactorizedReduce(C_in=prev_c,
                                                      C_out=num_channels,
                                                      stride=prev_s,
                                                      affine=bn_affine)
                else: # prev_c == _steps * num_channels or inputs
                    preprocess = ops.ReLUConvBN(C_in=prev_c,
                                                C_out=num_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                affine=bn_affine)
            self.preprocess_ops.append(preprocess)
        assert len(self.preprocess_ops) == self._num_init

        if self.use_shortcut:
            self.shortcut_reduction_op = ops.get_op(self.shortcut_op_type)(
                C=prev_num_channels[-1], C_out=self.num_out_channel(),
                stride=self.stride, affine=True)

        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for i_step in range(self._steps):
            to_ = i_step + self._num_init
            for from_ in range(to_):
                self.edges[from_][to_] = op_cls(self.num_channels, self.num_out_channels,
                                                stride=self.stride if from_ < self._num_init else 1,
                                                primitives=self._primitives, bn_affine=bn_affine,
                                                **op_kwargs)
                self.edge_mod.add_module("f_{}_t_{}".format(from_, to_), self.edges[from_][to_])

        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

    def on_replicate(self):
        # Although this edges is easy to understand, when paralleized,
        # the reference relationship between `self.edge` and modules under `self.edge_mod`
        # will not get updated automatically.

        # So, after each replicate, we should initialize a new edges dict
        # and update the reference manually.
        self.edges = defaultdict(dict)
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_ = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)] = edge_mod


class SharedOp(nn.Module):
    """
    The operation on an edge, consisting of multiple primitives.
    """

    def __init__(self, C, C_out, stride, primitives, partial_channel_proportion=None,
                 bn_affine=False):
        super(SharedOp, self).__init__()

        self.primitives = primitives
        self.stride = stride
        self.partial_channel_proportion = partial_channel_proportion

        if self.partial_channel_proportion is not None:
            expect(C % self.partial_channel_proportion == 0,
                   "partial_channel_proportion must be divisible by #channels", ConfigException)
            expect(C_out % self.partial_channel_proportion == 0,
                   "partial_channel_proportion must be divisible by #channels", ConfigException)
            C = C // self.partial_channel_proportion
            C_out = C_out // self.partial_channel_proportion

        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = ops.get_op(primitive)(C, C_out, stride, bn_affine)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C_out, affine=bn_affine))

            self.p_ops.append(op)
