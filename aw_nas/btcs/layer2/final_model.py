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
    """
    The 1st version of the micro-dense cell
    use postprocess operation(may cause information loss rather than preprocess)
    the 'use next stage width' is applied to the last cell each stage to align its
    width for the latter stage
    """

    NAME = "micro-dense-model"

    def __init__(
        self,
        search_space,
        arch,
        num_input_channels,
        num_out_channels,
        stride,
        postprocess=False,  # default use preprocess
        process_op_type="nor_conv_1x1",
        use_shortcut=True,
        shortcut_op_type="skip_connect",
        # applied on every cell at the end of the stage, before the reduction cell, 
        # to ensure x2 ch in reduction
        use_next_stage_width=None,  # applied on every cell at the end of the stage, 
        # before the reduction cell, to ensure x2 ch in reduction
        is_last_cell=False,
        is_first_cell=False,
        skip_cell=False,
        schedule_cfg=None,
    ):
        super(MicroDenseCell, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.arch = arch
        self.stride = stride
        self.postprocess = postprocess
        self.process_op_type = process_op_type

        self.use_shortcut = use_shortcut
        self.shortcut_op_type = shortcut_op_type

        self.num_steps = self.search_space.num_steps

        self._num_nodes = self.search_space._num_nodes
        self._primitives = self.search_space.primitives
        self._num_init_nodes = self.search_space.num_init_nodes

        self.is_last_cell = is_last_cell
        self.is_first_cell = is_first_cell
        self.skip_cell = skip_cell

        if use_next_stage_width is not None:
            self.use_next_stage_width = use_next_stage_width.item()
        else:
            self.use_next_stage_width = use_next_stage_width

        if self.use_next_stage_width:
            # when use_next_stage_width, should apply to normal_cell, in_c == out_c
            assert num_input_channels == num_out_channels

        self.num_input_channels = num_input_channels
        self.num_out_channels = num_out_channels

        if self.use_shortcut:
            """
            no 'use-next-stage-width' is applied in to the cell-wise shortcut,
            since the 'use-next-stage-width' only happens in last cell before reduction,
            the shortcut is usually plain shortcut and could not handle ch disalignment

            when using preprocess, the shortcut is of 4C width;
            when using postprocess, the shortcut is of C witdh;
            """
            if not self.postprocess:
                self.shortcut_reduction_op = ops.get_op(self.shortcut_op_type)(
                    C=num_input_channels * self.num_steps,
                    C_out=num_out_channels * self.num_steps,
                    stride=self.stride,
                    affine=True,
                )
            else:
                self.shortcut_reduction_op = ops.get_op(self.shortcut_op_type)(
                    C=num_input_channels,
                    C_out=num_out_channels,
                    stride=self.stride,
                    affine=True,
                )

        self.edges = _defaultdict_3()
        self.edge_mod = torch.nn.Module()  # a stub wrapping module of all the edges
        for from_ in range(self._num_nodes):
            for to_ in range(max(self._num_init_nodes, from_ + 1), self._num_nodes):
                num_input_channels = (
                    self.num_input_channels
                    if from_ < self._num_init_nodes
                    else self.num_out_channels
                )
                stride = self.stride if from_ < self._num_init_nodes else 1
                for op_ind in np.where(self.arch[to_, from_])[0]:
                    op_type = self._primitives[op_ind]
                    self.edges[from_][to_][op_type] = ops.get_op(op_type)(
                        # when applying the preprocess and cell `use-next-stage-width` 
                        # the op width should also align with next stage width
                        self.use_next_stage_width
                        if (
                            self.use_next_stage_width is not None
                            and not self.postprocess
                        )
                        else num_input_channels,
                        self.use_next_stage_width
                        if (
                            self.use_next_stage_width is not None
                            and not self.postprocess
                        )
                        else self.num_out_channels,
                        stride=stride,
                        affine=False,
                    )
                    self.edge_mod.add_module(
                        "f_{}_t_{}_{}".format(from_, to_, op_type),
                        self.edges[from_][to_][op_type],
                    )
        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)_([a-z0-9_-]+)")

        self.use_concat = self.search_space.concat_op == "concat"
        if self.use_concat:
            if not self.postprocess:
                # currently, map the concatenated output to num_out_channels
                self.process_op = ops.get_op(self.process_op_type)(
                    C=self.num_input_channels * self.search_space.num_steps,
                    # change outprocess op's output-width to align with next stage's width
                    # ensuring the reduction cell meets in_c*2 == out_c
                    C_out=self.use_next_stage_width
                    if self.use_next_stage_width is not None
                    else self.num_input_channels,
                    stride=1,
                    affine=True,
                )
            else:
                self.process_op = ops.get_op(self.process_op_type)(
                    C=self.num_out_channels * self.search_space.num_steps,
                    # change outprocess op's output-width to align with next stage's width
                    # ensuring the reduction cell meets in_c*2 == out_c
                    C_out=self.use_next_stage_width
                    if self.use_next_stage_width is not None
                    else self.num_out_channels,
                    stride=1,
                    affine=True,
                )

    def forward(self, inputs, dropout_path_rate):
        if self.skip_cell:
            out = self.shortcut_reduction_op(inputs)
            return out

        if not self.postprocess:
            states = [self.process_op(inputs)]
        else:
            states = [inputs]

        batch_size, _, height, width = states[0].shape
        o_height, o_width = height // self.stride, width // self.stride

        for to_ in range(self._num_init_nodes, self._num_nodes):
            state_to_ = torch.zeros(
                [
                    batch_size,
                    # if preprocess & use_next_stage_width, op width
                    self.use_next_stage_width
                    if (not self.postprocess and self.use_next_stage_width)
                    else self.num_out_channels,
                    o_height,
                    o_width,
                ],
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
        """
        (Maybe not so elegant)
        extra operations are applied here to align the width when 'use-next-stage-width'
        since the last cell is the normal cell, the cell-wise shortcut is the identity(),
        which could not change the dimension. but the conv inside should use-next-stage-width,
        which causes disalignment in width. for cell-wise shortcut, if ch_shortcut < ch_cell_out,
        only applying shortcut to former chs.
        """
        if self.use_concat:
            # concat all internal nodes
            out = torch.cat(states[self._num_init_nodes :], dim=1)
            if self.use_shortcut:
                assert (
                    self.is_last_cell and self.use_next_stage_width
                ) is not True, (
                    "is_last_cell and use_next_stage_width should not happen together"
                )
                if self.use_next_stage_width:
                    # if use-stage-width, cannot apply shortcut
                    shortcut = self.shortcut_reduction_op(inputs)
                    if self.postprocess:
                        out = self.process_op(out)
                    no_shortcut = False
                    if no_shortcut:
                        pass
                    else:
                        if shortcut.shape[1] > out.shape[1]:
                            out += shortcut[:, : out.shape[1], :, :]
                        else:
                            out[:, : shortcut.shape[1], :, :] += shortcut
                else:
                    if self.postprocess:
                        out = self.process_op(out)
                    out = out + self.shortcut_reduction_op(inputs)
            else:
                out = self.process_op(out)
        else:
            out = sum(states[self._num_init_nodes :])
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
        search_space,  # layer2
        device,
        genotypes,  # layer2
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
        auxiliary_head=False,
        auxiliary_cfg=None,
        schedule_cfg=None,
    ):
        super(MacroStagewiseFinalModel, self).__init__(schedule_cfg)
        self.macro_ss = search_space.macro_search_space
        self.micro_ss = search_space.micro_search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotypes_str = genotypes
        self.macro_g, self.micro_g = genotype_from_str(genotypes, search_space)

        # micro model (cell) class
        micro_model_cls = FinalModel.get_class_(micro_model_type)  # cell type

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
        self.layer_widths = [float(w) for w in self.macro_g.width.split(",")]

        # sort channels out
        assert self.stem_multiplier == 1, "Cannot handle stem_multiplier != 1 now"
        self.input_channel_list = [self.init_channels]
        for i in range(1, self.macro_ss.num_layers):
            self.input_channel_list.append(
                self.input_channel_list[i - 1] * 2
                if self._is_reduce(i - 1)
                else self.input_channel_list[i - 1]
            )
        for i in range(self.macro_ss.num_layers):
            self.input_channel_list[i] = int(
                self.input_channel_list[i] * self.layer_widths[i]
                if not self._is_reduce(i)
                else self.input_channel_list[i] * self.layer_widths[i - 1]
            )

        self.output_channel_list = self.input_channel_list[1:] + [
            self.input_channel_list[-1]
        ]

        # construct cells
        if not self.use_stem:
            raise NotImplementedError
            c_stem = 3
        elif isinstance(self.use_stem, (list, tuple)):
            raise NotImplementedError
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
            self.stem = ops.get_op(self.use_stem)(
                3, self.input_channel_list[0], stride=stem_stride, affine=stem_affine
            )

        self.cells = nn.ModuleList()
        self.micro_arch_list = self.micro_ss.rollout_from_genotype(self.micro_g).arch
        for i_layer in range(self.macro_ss.num_layers):
            # print(i_layer, self._is_reduce(i_layer))
            stride = 2 if self._is_reduce(i_layer) else 1
            cg_idx = self.macro_ss.cell_layout[i_layer]
            # contruct micro cell
            # FIXME: Currently MacroStageWiseFinalModel doesnot support postprocess = False
            micro_model_cfg["postprocess"] = True
            cell = micro_model_cls(
                self.micro_ss,
                self.micro_arch_list[cg_idx],
                num_input_channels=self.input_channel_list[i_layer],
                num_out_channels=self.output_channel_list[i_layer],
                stride=stride,
                **micro_model_cfg
            )
            # assume non-reduce cell does not change channel number
            self.cells.append(cell)
            # add auxiliary head
            if i_layer == (2 * self.macro_ss.num_layers) // 3 and self.auxiliary_head:
                if auxiliary_head == "imagenet":
                    self.auxiliary_net = AuxiliaryHeadImageNet(
                        self.output_channel_list[i_layer],
                        num_classes,
                        **(auxiliary_cfg or {})
                    )
                else:
                    self.auxiliary_net = AuxiliaryHead(
                        self.output_channel_list[i_layer],
                        num_classes,
                        **(auxiliary_cfg or {})
                    )

        self.lastact = nn.Identity()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(self.output_channel_list[-1], self.num_classes)
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self._set_hook()

    def __getstate__(self):
        state = super(MacroStagewiseFinalModel, self).__getstate__().copy()
        del state["macro_g"]
        del state["micro_g"]
        return state

    def __setstate(self, state):
        super(MacroStagewiseFinalModel, self).__setstate(state)
        self.macro_g, self.micro_g = genotype_from_str(
            self.genotypes_str, self.search_space
        )

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
                device=inputs.device,
            )
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

        # final processing
        input_ = torch.zeros(
            (batch_size, self.input_channel_list[layer_idx], i_height, i_width),
            device=inputs.device,
        )
        for input_node in np.where(self.overall_adj[node_idx + 1])[0]:
            input_ += states[input_node]
        states.append(input_)

        out = self.global_pooling(states[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops / 1.0e6)
            self._flops_calculated = True

        if self.auxiliary_head and self.training:
            return logits, logits_aux

        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _set_hook(self):
        for _, module in self.named_modules():
            if "aux" in _:
                continue
            module.register_forward_hook(self._hook_intermediate_feature)

    def _is_reduce(self, i_layer):
        return self.macro_ss.cell_layout[i_layer] in self.macro_ss.reduce_cell_groups

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


class MacroSinkConnectFinalModel(MacroStagewiseFinalModel):
    NAME = "macro-sinkconnect-model"
    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(
        self,
        search_space,  # layer2
        device,
        genotypes,  # layer2
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
        auxiliary_head=False,
        auxiliary_cfg=None,
        schedule_cfg=None,
    ):
        super(MacroStagewiseFinalModel, self).__init__(schedule_cfg)
        self.macro_ss = search_space.macro_search_space
        self.micro_ss = search_space.micro_search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotypes_str = genotypes
        self.macro_g, self.micro_g = genotype_from_str(genotypes, search_space)

        # micro model (cell) class
        micro_model_cls = FinalModel.get_class_(micro_model_type)  # cell type

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
        self.layer_widths = [float(w) for w in self.macro_g.width.split(",")]

        self.micro_model_cfg = micro_model_cfg
        if "postprocess" in self.micro_model_cfg.keys():
            self.cell_use_postprocess = self.micro_model_cfg["postprocess"]
        else:
            self.cell_use_postprocess = False

        # sort channels out
        assert self.stem_multiplier == 1, "Cannot handle stem_multiplier != 1 now"
        self.input_channel_list = [self.init_channels]
        for i in range(1, self.macro_ss.num_layers):
            self.input_channel_list.append(
                self.input_channel_list[i - 1] * 2
                if self._is_reduce(i - 1)
                else self.input_channel_list[i - 1]
            )
        for i in range(self.macro_ss.num_layers):
            self.input_channel_list[i] = int(
                self.input_channel_list[i] * self.layer_widths[i]
                if not self._is_reduce(i)
                else self.input_channel_list[i] * self.layer_widths[i - 1]
            )

        self.output_channel_list = self.input_channel_list[1:] + [
            self.input_channel_list[-1]
        ]

        # construct cells
        if not self.use_stem:
            raise NotImplementedError
            c_stem = 3
        elif isinstance(self.use_stem, (list, tuple)):
            raise NotImplementedError
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
            self.stem = ops.get_op(self.use_stem)(
                3, self.input_channel_list[0], stride=stem_stride, affine=stem_affine
            )

        self.extra_stem = ops.get_op("nor_conv_1x1")(
            self.input_channel_list[0],
            self.input_channel_list[0] * self.micro_ss.num_steps,
            stride=1,
            affine=True,
        )

        # For sink-connect, don't init all cells, just init connected cells

        connected_cells = []
        for cell_idx in range(1, self.macro_ss.num_layers + 2):
            if len(self.overall_adj[cell_idx].nonzero()[0]) > 0:
                connected_cells.append(self.overall_adj[cell_idx].nonzero()[0])
        # -1 to make the 1st element 0
        self.connected_cells = np.concatenate(connected_cells)[1:] - 1

        """
        ininitialize cells, only connected cells are initialized
        also use `use_next_stage_width` to handle the disalignment of width due to width search
        """
        self.cells = nn.ModuleList()
        self.micro_arch_list = self.micro_ss.rollout_from_genotype(self.micro_g).arch
        for i_layer in range(self.macro_ss.num_layers):
            stride = 2 if self._is_reduce(i_layer) else 1
            connected_is_reduce = [self._is_reduce(i) for i in self.connected_cells]
            # the layer-idx to use next stage's width: the last cell 
            # before the redudction cell in each stage
            use_next_stage_width_layer_idx = self.connected_cells[
                np.argwhere(np.array(connected_is_reduce)).reshape(-1) - 1
            ]
            reduction_layer_idx = self.connected_cells[
                np.argwhere(np.array(connected_is_reduce)).reshape(-1)
            ]  #  find reudction cells are the 1-th in connected cells
            if not self.cell_use_postprocess:
                next_stage_widths = (
                    np.array(self.output_channel_list)[self.macro_ss.stages_begin[1:]]
                    // 2
                )  # preprocess, so no //2
            else:
                next_stage_widths = (
                    np.array(self.output_channel_list)[self.macro_ss.stages_begin[1:]]
                    // 2
                )  # the width to use for `ues_next_stage_width`, the reduction cell is of expansion 2, so //2
            use_next_stage_width = (
                next_stage_widths[
                    np.argwhere(use_next_stage_width_layer_idx == i_layer).reshape(-1)
                ]
                if np.argwhere(use_next_stage_width_layer_idx == i_layer).size > 0
                else None
            )
            input_channel_list_n = np.array(self.input_channel_list)
            input_channel_list_n[
                reduction_layer_idx
            ] = next_stage_widths  # input of the reduction should be half of the next stage's width

            cg_idx = self.macro_ss.cell_layout[i_layer]
            if i_layer not in self.connected_cells:
                continue
            # contruct micro cell
            cell = micro_model_cls(
                self.micro_ss,
                self.micro_arch_list[cg_idx],
                num_input_channels=int(
                    input_channel_list_n[i_layer]
                ),  # TODO: input_channel_list is of type: np.int64
                num_out_channels=self.output_channel_list[i_layer],
                stride=stride,
                use_next_stage_width=use_next_stage_width,
                is_last_cell=True if i_layer == self.connected_cells[-1] else False,
                is_first_cell=True if i_layer == self.connected_cells[0] else False,
                skip_cell=False,
                **micro_model_cfg
            )
            # assume non-reduce cell does not change channel number

            self.cells.append(cell)
            # add auxiliary head

        # connected_cells has 1 more element [0] than the self.cells
        if self.auxiliary_head:
            self.where_aux_head = self.connected_cells[(2 * len(self.cells)) // 3]
            extra_expansion_for_aux = (
                1 if self.cell_use_postprocess else self.micro_ss.num_steps
            )  # if use preprocess, aux head's input ch num should change accordingly
            # aux head is connected to last cell's output
            if auxiliary_head == "imagenet":
                self.auxiliary_net = AuxiliaryHeadImageNet(
                    input_channel_list_n[self.where_aux_head] * extra_expansion_for_aux,
                    num_classes,
                    **(auxiliary_cfg or {})
                )
            else:
                self.auxiliary_net = AuxiliaryHead(
                    input_channel_list_n[self.where_aux_head] * extra_expansion_for_aux,
                    num_classes,
                    **(auxiliary_cfg or {})
                )

        self.lastact = nn.Identity()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()

        if not self.cell_use_postprocess:
            self.classifier = nn.Linear(
                self.output_channel_list[-1] * self.micro_ss.num_steps, self.num_classes
            )
        else:
            self.classifier = nn.Linear(self.output_channel_list[-1], self.num_classes)
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

        """when using preprocess, extra stem is used for 4xch the stem output"""
        if not self.cell_use_postprocess:
            states[0] = self.extra_stem(states[0])

        for idx, cell in enumerate(self.cells):
            layer_idx = self.connected_cells[idx]
            # construct all zero tensor, since the macro arch might be dis-connected
            # should not happen when sink-connect macro is used
            # input_ = torch.zeros(
            #     (batch_size, self.input_channel_list[layer_idx], i_height, i_width),
            #     device=inputs.device)
            if self._is_reduce(layer_idx):
                # calculate spatial size
                i_height = i_height // 2
                i_width = i_width // 2
            # since only connected cells are initialized, so we donnot 
            # need to use adj-matrix to identify macro connection pattern
            # node_idx = layer_idx + 1
            # for input_node in np.where(self.overall_adj[node_idx])[0]:
            #     input_ += states[input_node]
            input_ = states[idx]
            output = cell(input_, dropout_path_rate=self.dropout_path_rate)
            states.append(output)
            # forward the auxiliary head
            if idx == 2 * len(self.cells) // 3 - 1:
                if self.auxiliary_head and self.training:
                    logits_aux = self.auxiliary_net(states[-1])

        out = self.global_pooling(states[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops / 1.0e6)
            self._flops_calculated = True

        if self.auxiliary_head and self.training:
            return logits, logits_aux

        return logits
