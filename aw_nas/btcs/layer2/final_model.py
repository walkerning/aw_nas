"""
Layer 2 final model.
"""

import re
from collections import defaultdict

import six
import numpy as np
import torch
from torch import nn

from aw_nas import ops, utils
from aw_nas.final.base import FinalModel
from aw_nas.final.cnn_model import AuxiliaryHead, AuxiliaryHeadImageNet
from aw_nas.common import genotype_from_str


def _defaultdict_1():
    return defaultdict(dict)

def _defaultdict_2():
    return defaultdict(_defaultdict_1)

def _defaultdict_3():
    return defaultdict(_defaultdict_2)

class MicroDenseCell(FinalModel):
    NAME = "micro-dense-model"

    def __init__(self, search_space, arch, num_input_channels, num_out_channels, stride,
                 output_process_op="nor_conv_1x1",
                 use_shortcut=False,
                 shortcut_op_type="skip_connect",
                 schedule_cfg=None):
        super(MicroDenseCell, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.arch = arch
        self.num_input_channels = num_input_channels
        self.num_out_channels = num_out_channels
        self.stride = stride
        self.output_process_op = output_process_op

        self.use_shortcut = use_shortcut
        self.shortcut_op_type = shortcut_op_type

        if self.use_shortcut:
            self.shortcut_reduction_op = ops.get_op(self.shortcut_op_type)(
                C=num_input_channels, C_out=num_out_channels,
                stride=self.stride, affine=True,
            )

        self._num_nodes = self.search_space._num_nodes
        self._primitives = self.search_space.primitives
        self._num_init_nodes = self.search_space.num_init_nodes

        self.edges = _defaultdict_3()
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for from_ in range(self._num_nodes):
            for to_ in range(max(self._num_init_nodes, from_ + 1), self._num_nodes):
                num_input_channels = (self.num_input_channels \
                                      if from_ < self._num_init_nodes else self.num_out_channels)
                stride = self.stride if from_ < self._num_init_nodes else 1
                for op_ind in np.where(self.arch[to_, from_])[0]:
                    op_type = self._primitives[op_ind]
                    self.edges[from_][to_][op_type] = ops.get_op(op_type)(
                        num_input_channels,
                        self.num_out_channels,
                        stride=stride,
                        affine=False,
                    )
                    self.edge_mod.add_module(
                        "f_{}_t_{}_{}".format(from_, to_, op_type), self.edges[from_][to_][op_type]
                    )
        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)_([a-z0-9_-]+)")

        self.use_concat = self.search_space.concat_op == "concat"
        if self.use_concat:
            # currently, map the concatenated output to num_out_channel
            self.out_process_op = ops.get_op(self.output_process_op)(
                C=self.num_out_channels * self.search_space.num_steps,
                C_out=self.num_out_channels, stride=1, affine=False)

    def forward(self, inputs, dropout_path_rate):
        # TOOD: Add cell-shortcut
        states = [inputs]
        batch_size, _, height, width = states[0].shape
        o_height, o_width = height // self.stride, width // self.stride

        for to_ in range(self._num_init_nodes, self._num_nodes):
            state_to_ = torch.zeros(
                [batch_size, self.num_out_channels, o_height, o_width],
                device=states[0].device,
            )
            for from_ in range(to_):
                for op_ in self.edges[from_][to_].values():
                    if isinstance(op_, ops.Zero):
                        continue
                    out = op_(states[from_])
                    if self.training and dropout_path_rate > 0:
                        if not isinstance(op_, ops.Identity):
                            out = utils.drop_path(out, dropout_path_rate)
                    state_to_ = state_to_ + out
            states.append(state_to_)

        if self.use_concat:
            # concat all internal nodes
            out = torch.cat(states[self._num_init_nodes:], dim=1)
            out = self.out_process_op(out)
        else:
            out = sum(states[self._num_init_nodes:])

        if self.use_shortcut:
            out = out + self.shortcut_reduction_op(inputs)

        return out

    def on_replicate(self):
        self.edges = _defaultdict_3()
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_, op_type = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)][op_type] = edge_mod

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def num_out_channel(self):
        return self.num_out_channels


class MacroStagewiseFinalModel(FinalModel):
    """
    Accept layer2 layout.
    """

    NAME = "macro-stagewise-model"
    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(
            self,
            search_space, # layer2
            device,
            genotypes, # layer2
            micro_model_type="micro-dense-model",
            micro_model_cfg={},
            num_classes=10,
            init_channels=36,
            stem_multiplier=1,
            dropout_rate=0.0,
            dropout_path_rate=0.0,
            use_stem="conv_bn_3x3",
            stem_stride=1,
            stem_affine=True,
            auxiliary_head=False, auxiliary_cfg=None,
            schedule_cfg=None,
    ):
        super(MacroStagewiseFinalModel, self).__init__(schedule_cfg)

        self.macro_ss = search_space.macro_search_space
        self.micro_ss = search_space.micro_search_space
        self.device = device
        assert isinstance(genotypes, str)
        # TODO: remove this in __getstate__
        self.macro_g, self.micro_g = genotype_from_str(genotypes, search_space)

        # micro model (cell) class
        micro_model_cls = FinalModel.get_class_(micro_model_type) # cell type

        self.num_classes = num_classes
        self.init_channels = init_channels
        self.stem_multiplier = stem_multiplier
        self.stem_stride = stem_stride
        self.stem_affine = stem_affine
        self.use_stem = use_stem

        # training
        self.dropout_rate = dropout_rate
        self.dropout_path_rate = dropout_path_rate

        self.auxiliary_head = auxiliary_head

        self.overall_adj = self.macro_ss.parse_overall_adj(self.macro_g)

        # construct cells
        if not self.use_stem:
            c_stem = 3
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
            self.stem = nn.Sequential(self.stems)
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = c_stem
        self.micro_arch_list = self.micro_ss.rollout_from_genotype(self.micro_g).arch
        self.input_channel_list = []
        for i_layer in range(self.macro_ss.num_layers):
            stride = 2 if self._is_reduce(i_layer) else 1
            if stride > 1:
                num_channels *= stride
            self.input_channel_list.append(prev_num_channels)
            cg_idx = self.macro_ss.cell_layout[i_layer]
            # contruct micro cell
            cell = micro_model_cls(self.micro_ss, self.micro_arch_list[cg_idx],
                                   num_input_channels=prev_num_channels,
                                   num_out_channels=num_channels,
                                   stride=stride,
                                   **micro_model_cfg)
            # assume non-reduce cell does not change channel number
            prev_num_channels = cell.num_out_channel()
            self.cells.append(cell)
            # add auxiliary head
            if i_layer == (2 * self.macro_ss.num_layers) // 3 and self.auxiliary_head:
                if auxiliary_head == "imagenet":
                    self.auxiliary_net = AuxiliaryHeadImageNet(
                        prev_num_channels, num_classes, **(auxiliary_cfg or {}))
                else:
                    self.auxiliary_net = AuxiliaryHead(
                        prev_num_channels, num_classes, **(auxiliary_cfg or {}))

        self.lastact = nn.Identity()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(prev_num_channels, self.num_classes)
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self._set_hook()

    def forward(self, inputs):
        if not self.use_stem:
            states = [inputs]
        else:
            states = [self.stem(inputs)]

        batch_size, _, i_height, i_width = states[0].shape

        for layer_idx, cell in enumerate(self.cells):
            # construct all zero tensor, since the macro arch might be dis-connected
            input_ = torch.zeros(
                (batch_size, self.input_channel_list[layer_idx], i_height, i_width),
                device=inputs.device)
            if self._is_reduce(layer_idx):
                # calculate spatial size
                i_height = i_height // 2
                i_width = i_width // 2
            node_idx = layer_idx + 1
            for input_node in np.where(self.overall_adj[node_idx])[0]:
                input_ += states[input_node]
            output = cell(input_, dropout_path_rate=self.dropout_path_rate)
            states.append(output)
            # forward the auxiliary head
            if layer_idx == 2 * len(self.cells) // 3:
                if self.auxiliary_head and self.training:
                    logits_aux = self.auxiliary_net(states[-1])

        # final_processing
        input_ = torch.zeros(
            (batch_size, self.input_channel_list[layer_idx], i_height, i_width),
            device=inputs.device) 
        for input_node in np.where(self.overall_adj[node_idx+1])[0]:
            input_ += states[input_node]
        states.append(input_)

        out = self.global_pooling(states[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops/1.e6)
            self._flops_calculated = True

        if self.auxiliary_head and self.training:
            return logits, logits_aux

        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _set_hook(self):
        for _, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _is_reduce(self, i_layer):
        return (
            self.macro_ss.cell_layout[i_layer]
            in self.macro_ss.reduce_cell_groups
        )

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

