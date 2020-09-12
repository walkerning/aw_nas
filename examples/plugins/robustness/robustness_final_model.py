# -*- coding: utf-8 -*-

import re
from collections import defaultdict

import six
import numpy as np
import torch
from torch import nn

from aw_nas import utils, ops
from aw_nas.common import genotype_from_str
from aw_nas.ops import register_primitive
from aw_nas.final.base import FinalModel

skip_connect_2 = (
    lambda C, C_out, stride, affine: ops.FactorizedReduce(
        C, C_out, stride=stride, affine=affine
    )
    if stride == 2
    else ops.ReLUConvBN(C, C_out, 1, 1, 0)
)

# skip_connect_2 = (
#     lambda C, C_out, stride, affine: ops.FactorizedReduce(
#         C, C_out, stride=stride, affine=affine
#     )
#     if stride == 2
#     else (ops.Identity() if C == C_out else nn.Conv2d(C, C_out, 1, 1, 0))
# )

register_primitive("skip_connect_2", skip_connect_2)


class DenseRobFinalModel(FinalModel):
    NAME = "dense_rob_final_model"

    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(
        self,
        search_space,
        device,
        genotypes,
        num_classes=10,
        init_channels=36,
        stem_multiplier=1,
        dropout_rate=0.0,
        dropout_path_rate=0.0,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        schedule_cfg=None,
    ):
        super(DenseRobFinalModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        assert isinstance(genotypes, str)
        genotypes = genotype_from_str(genotypes, self.search_space)
        self.arch_list = self.search_space.rollout_from_genotype(genotypes).arch

        self.num_classes = num_classes
        self.init_channels = init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem

        # training
        self.dropout_rate = dropout_rate
        self.dropout_path_rate = dropout_path_rate

        # search space configs
        self._num_init = self.search_space.num_init_nodes
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
                self.stems.append(
                    ops.get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )
            self.stems = nn.ModuleList(self.stems)
            init_strides = [stem_stride] * self._num_init
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )
            init_strides = [1] * self._num_init

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = [c_stem] * self._num_init
        strides = [
            2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)
        ]

        for i_layer, stride in enumerate(strides):
            if stride > 1:
                num_channels *= stride
            num_out_channels = num_channels
            kwargs = {}
            cg_idx = self.search_space.cell_layout[i_layer]

            cell = DenseRobCell(
                self.search_space,
                self.arch_list[cg_idx],
                # num_channels=num_channels,
                num_input_channels=prev_num_channels,
                num_out_channels=num_out_channels,
                # prev_num_channels=tuple(prev_num_channels),
                prev_strides=init_strides + strides[:i_layer],
                stride=stride,
                **kwargs
            )

            prev_num_channel = cell.num_out_channel()
            prev_num_channels.append(prev_num_channel)
            prev_num_channels = prev_num_channels[1:]
            self.cells.append(cell)

        self.lastact = nn.Identity()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(prev_num_channels[-1], self.num_classes)
        self.to(self.device)

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
                self.total_flops += (
                    2
                    * inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * outputs.size(2)
                    * outputs.size(3)
                    / module.groups
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def forward(self, inputs):  # pylint: disable=arguments-differ
        if not self.use_stem:
            states = inputs
        elif isinstance(self.use_stem, (list, tuple)):
            stemed = inputs
            for stem in self.stems:
                stemed = stem(stemed)
            states = stemed
        else:
            stemed = self.stem(inputs)
            states = stemed
        states = [states] * self._num_init

        for layer_idx, cell in enumerate(self.cells):
            o_states = cell(states, self.dropout_path_rate)
            states.append(o_states)
            states = states[1:]

        out = self.lastact(states[-1])
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops / 1.0e6)
            self._flops_calculated = True

        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _is_reduce(self, i_layer):
        return (
            self.search_space.cell_layout[i_layer]
            in self.search_space.reduce_cell_groups
        )


class DenseRobCell(nn.Module):
    def __init__(
        self,
        search_space,
        cell_arch,
        num_input_channels,
        num_out_channels,
        stride,
        prev_strides,
    ):
        super(DenseRobCell, self).__init__()

        self.search_space = search_space
        self.arch = cell_arch
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_input_channels = num_input_channels
        self.num_out_channels = num_out_channels
        self.num_init_nodes = self.search_space.num_init_nodes

        self.preprocess_ops = nn.ModuleList()
        prev_strides = prev_strides[-self.num_init_nodes :]
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[: len(num_input_channels)])
        for prev_c, prev_s in zip(num_input_channels, prev_strides):
            preprocess = ops.get_op("skip_connect_2")(
                C=prev_c, C_out=num_out_channels, stride=prev_s, affine=True
            )
            self.preprocess_ops.append(preprocess)

        self._num_nodes = self.search_space._num_nodes
        self._primitives = self.search_space.primitives
        self.num_init_nodes = self.search_space.num_init_nodes

        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module()  # a stub wrapping module of all the edges
        for from_ in range(self._num_nodes):
            for to_ in range(max(self.num_init_nodes, from_ + 1), self._num_nodes):
                self.edges[from_][to_] = ops.get_op(
                    self._primitives[int(self.arch[to_][from_])]
                )(
                    # self.num_input_channels[from_] \
                    # if from_ < self.num_init_nodes else self.num_out_channels,
                    self.num_out_channels,
                    self.num_out_channels,
                    stride=self.stride if from_ < self.num_init_nodes else 1,
                    affine=False,
                )
                self.edge_mod.add_module(
                    "f_{}_t_{}".format(from_, to_), self.edges[from_][to_]
                )
        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

    def forward(self, inputs, dropout_path_rate):  # pylint: disable=arguments-differ
        states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]
        batch_size, _, height, width = states[0].shape
        o_height, o_width = height // self.stride, width // self.stride

        for to_ in range(self.num_init_nodes, self._num_nodes):
            state_to_ = torch.zeros(
                [batch_size, self.num_out_channels, o_height, o_width],
                device=states[0].device,
            )
            for from_ in range(to_):
                op_ = self.edges[from_][to_]
                if isinstance(op_, ops.Zero):
                    continue
                out = op_(states[from_])
                if self.training and dropout_path_rate > 0:
                    if not isinstance(op_, ops.Identity):
                        out = utils.drop_path(out, dropout_path_rate)
                state_to_ = state_to_ + out
            states.append(state_to_)

        # concat all internal nodes
        return torch.cat(states[self.num_init_nodes:], dim=1)

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

    def num_out_channel(self):
        return self.search_space.num_steps * self.num_out_channels
