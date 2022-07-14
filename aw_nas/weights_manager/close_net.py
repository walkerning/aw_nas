# -*- coding: utf-8 -*-
"""
Curriculum Learning On Sharing Extent, CLOSE.
"""

from __future__ import print_function

import numpy as np
import itertools
from collections import OrderedDict
import contextlib
import six
import random

import torch
from torch import nn
from torch.nn.parameter import Parameter

from aw_nas import ops, utils
from aw_nas.common import assert_rollout_type, group_and_sort_by_to_node, BaseRollout
from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp
from aw_nas.utils import data_parallel, use_params
from aw_nas.utils import getLogger
from aw_nas.utils.parallel_utils import _check_support_candidate_member_mask
from aw_nas.weights_manager.super_net import SubCandidateNet, DiscreteSharedOp
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.evaluator.arch_network import GCNFlowArchEmbedder
from functools import partial

__all__ = ["CloseCandidateNet", "CloseNet"]


class CloseCandidateNet(SubCandidateNet):
    def __init__(
        self,
        close_net,
        rollout,
        member_mask,
        gpus=tuple(),
        cache_named_members=False,
        virtual_parameter_only=True,
        eval_no_grad=True,
        for_eval=False,
    ):
        super(CloseCandidateNet, self).__init__(
            super_net=close_net,
            rollout=rollout,
            member_mask=member_mask,
            gpus=gpus,
            cache_named_members=cache_named_members,
            virtual_parameter_only=virtual_parameter_only,
            eval_no_grad=eval_no_grad,
        )
        self.for_eval = for_eval
        self.gate = rollout.gate
        self.rollout = rollout

    def _forward(self, inputs, **kwargs):
        genotypes = list(zip(self.genotypes_grouped, self.rollout.arch, self.gate))
        return self.super_net.forward(inputs, genotypes, is_training=not self.for_eval)


class CloseNet(SharedNet):
    NAME = "closenet"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="discrete",
        gpus=tuple(),
        num_classes=10,
        init_channels=16,
        stem_multiplier=3,
        max_grad_norm=5.0,
        dropout_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        preprocess_op_type=None,
        cell_use_preprocess=True,
        cell_group_kwargs=None,
        cell_use_shortcut=False,
        cell_shortcut_op_type="skip_connect",
        bn_affine=False,
        candidate_member_mask=True,
        candidate_cache_named_members=False,
        candidate_virtual_parameter_only=False,
        candidate_eval_no_grad=True,
        # For CLOSE
        num_curriculums=8,
        cl_schedule=[1, 101, 201, 301, 401, 501, 601, 701],
        # For GATES
        op_dim=32,
        node_dim=32,
        hidden_dim=32,
        gcn_out_dims=[64, 64, 64, 64],
        gcn_kwargs=None,
        use_bn=False,
        mlp_dims=[200],
        mlp_dropout=0.1,
        gate_type="arch-level",
        reconstruct_only_last=False,
        reconstruct_inherit=False,
        wit_from="prev",
        eval_arch=False,
    ):
        _check_support_candidate_member_mask(gpus, candidate_member_mask, self.NAME)

        super(CloseNet, self).__init__(
            search_space,
            device,
            rollout_type,
            cell_cls=partial(CloseCell, num_curriculums=num_curriculums),
            op_cls=DiscreteSharedOp,
            gpus=gpus,
            num_classes=num_classes,
            init_channels=init_channels,
            stem_multiplier=stem_multiplier,
            max_grad_norm=max_grad_norm,
            dropout_rate=dropout_rate,
            use_stem=use_stem,
            stem_stride=stem_stride,
            stem_affine=stem_affine,
            preprocess_op_type=preprocess_op_type,
            cell_use_preprocess=cell_use_preprocess,
            cell_group_kwargs=cell_group_kwargs,
            cell_use_shortcut=cell_use_shortcut,
            cell_shortcut_op_type=cell_shortcut_op_type,
            bn_affine=bn_affine,
        )

        # candidate net with/without parameter mask
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only
        self.candidate_eval_no_grad = candidate_eval_no_grad
        self.set_hook()
        self._flops_calculated = False
        self.total_flops = 0

        # CL configs
        self.num_positions = (
            self.search_space.num_steps * self.search_space.num_node_inputs
        )
        self.num_curriculums = num_curriculums
        self.cl_schedule = cl_schedule
        self.curriculum_level = 1

        # MLP configs
        self.mlp_dims = mlp_dims
        self.mlp_dropout = mlp_dropout
        self.reconstruct_only_last = reconstruct_only_last
        self.reconstruct_inherit = reconstruct_inherit

        # construct predictor (GATES & MLP)
        self.embedder = GCNFlowArchEmbedder(
            search_space=search_space,
            node_dim=node_dim,
            op_dim=op_dim,
            hidden_dim=hidden_dim,
            gcn_out_dims=gcn_out_dims,
            gcn_kwargs=gcn_kwargs,
            use_bn=use_bn,
        )

        self.mlp_in_dim = self.embedder.out_dim
        self.mlp_out_dim = (
            self.num_positions * self.num_curriculums
            if gate_type == "arch-level"
            else self.num_curriculums
        )
        self.mlp = self.construct_mlp(
            self.mlp_dims, self.mlp_in_dim, self.mlp_out_dim, self.mlp_dropout
        )

        self.gate_type = gate_type
        self.eval_arch = eval_arch
        self.wit_from = wit_from
        self.to(self.device)

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def set_hook(self):
        for name, module in self.named_modules():
            if "auxiliary" in name:
                continue
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += (
                    inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * inputs[0].size(2)
                    * inputs[0].size(3)
                    / (module.stride[0] * module.stride[1] * module.groups)
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("discrete"), assert_rollout_type("nb301")]

    def construct_mlp(self, mlp_dims, in_dim, out_dim, mlp_dropout):
        mlp = []
        for dim in mlp_dims:
            mlp.append(
                nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout),
                )
            )
            in_dim = dim
        mlp.append(nn.Linear(in_dim, out_dim))
        mlp = nn.Sequential(*mlp)
        return mlp

    def get_genotypes_grouped(self, rollout):
        genotype_list = rollout.genotype_list()
        genotypes = [g[1] for g in genotype_list]
        genotypes_grouped = list(
            zip(
                [
                    group_and_sort_by_to_node(conns)
                    for conns in genotypes[: self.search_space.num_cell_groups]
                ],
                genotypes[self.search_space.num_cell_groups :],
            )
        )
        return genotypes_grouped

    def random_gate(self, rollout):
        arch = np.array(rollout.arch)
        node_embs = self.embedder(np.expand_dims(arch, 0), return_all=True).squeeze(0)
        genotypes_grouped = self.get_genotypes_grouped(rollout)

        normal_gate = []
        conns_grouped, _ = genotypes_grouped[0]
        for to_, connections in conns_grouped:
            for op_type, from_, _ in connections:
                index = torch.randint(0, self.curriculum_level, (1,))
                logit = torch.nn.functional.one_hot(index, self.curriculum_level)[0]
                normal_gate.append(logit)

        reduce_gate = []
        conns_grouped, _ = genotypes_grouped[1]
        for to, connections in conns_grouped:
            for op_type, from_, _ in connections:
                index = torch.randint(0, self.curriculum_level, (1,))
                logit = torch.nn.functional.one_hot(index, self.curriculum_level)[0]
                reduce_gate.append(logit)

        return (normal_gate, reduce_gate)

    def node_level_gate(self, rollout, softmax=True):
        arch = np.array(rollout.arch)
        node_embs = self.embedder(np.expand_dims(arch, 0), return_all=True).squeeze(0)
        genotypes_grouped = self.get_genotypes_grouped(rollout)

        normal_gate = []
        conns_grouped, _ = genotypes_grouped[0]
        for to_, connections in conns_grouped:
            for op_type, from_, _ in connections:
                logit = self.mlp(torch.cat((node_embs[0][from_], node_embs[0][to_])))
                logit = logit[: self.curriculum_level]
                if softmax == False:
                    normal_gate.append(logit)
                    continue
                prob, _ = utils.gumbel_softmax(logit, temperature=5.0, hard=False)
                normal_gate.append(utils.straight_through(prob))

        reduce_gate = []
        conns_grouped, _ = genotypes_grouped[1]
        for to, connections in conns_grouped:
            for op_type, from_, _ in connections:
                logit = self.mlp(torch.cat((node_embs[1][from_], node_embs[1][to_])))
                logit = logit[: self.curriculum_level]
                if softmax == False:
                    reduce_gate.append(logit)
                    continue
                prob, _ = utils.gumbel_softmax(logit, temperature=5.0, hard=False)
                reduce_gate.append(utils.straight_through(prob))

        return (normal_gate, reduce_gate)

    def arch_level_gate(self, rollout):
        arch = np.array(rollout.arch)
        score = self.mlp(self.embedder(np.expand_dims(arch, 0), return_all=False))
        score = score.squeeze(0).reshape(self.num_positions, self.num_curriculums)
        gate = []
        for i in range(score.shape[0]):
            logit = score[i][: self.curriculum_level]
            prob, _ = utils.gumbel_softmax(logit, temperature=5.0, hard=False)
            gate.append(utils.straight_through(prob))
        return (gate, gate)

    def assemble_candidate(self, rollout, for_eval=False, softmax=True):
        if self.gate_type == "arch-level":
            rollout.gate = self.arch_level_gate(rollout)
        elif self.gate_type == "node-level":
            rollout.gate = self.node_level_gate(rollout, softmax=softmax)
        elif self.gate_type == "random":
            rollout.gate = self.random_gate(rollout)

        return CloseCandidateNet(
            self,
            rollout,
            gpus=self.gpus,
            member_mask=self.candidate_member_mask,
            cache_named_members=self.candidate_cache_named_members,
            virtual_parameter_only=self.candidate_virtual_parameter_only,
            eval_no_grad=self.candidate_eval_no_grad,
            for_eval=for_eval,
        )

    def new_curriculum(self, level, tp):
        assert tp in ["random", "prev"]
        for cell in self.cells:
            cell.new_curriculum(level, tp)
        self.logger.info("Successfully add a new curriculum of level {}".format(level))

    def inherit(self, w, x):
        w[x] += w[x - 1]
        w.requires_grad = True
        return w

    def on_epoch_start(self, epoch):
        super(CloseNet, self).on_epoch_start(epoch)
        self.curriculum_level = np.where(np.array(self.cl_schedule) <= epoch)[0][-1] + 1
        if (epoch in self.cl_schedule) and (not self.eval_arch):
            if self.reconstruct_inherit:
                if self.curriculum_level > 1:
                    last_layer = list(self.mlp.children())[-1]
                    new_mlp = list(self.mlp.children())[:-1]
                    w = self.inherit(
                        last_layer.weight.clone().detach(), self.curriculum_level - 1
                    )
                    b = self.inherit(
                        last_layer.bias.clone().detach(), self.curriculum_level - 1
                    )
                    last_layer.weight = Parameter(w)
                    last_layer.bias = Parameter(b)
                    new_mlp.append(last_layer)
                    self.mlp = nn.Sequential(*new_mlp)
                self.logger.info(
                    "Sucessfully inherit paramaters of the last layer of the MLP!!!"
                )
            elif self.reconstruct_only_last:
                new_mlp = list(self.mlp.children())[:-1]
                new_mlp.append(nn.Linear(self.mlp_dims[-1], self.mlp_out_dim))
                self.mlp = nn.Sequential(*new_mlp)
                self.logger.info(
                    "Successfully reinit the parameters of the last layer of the MLP!!!"
                )
            else:
                self.mlp = self.construct_mlp(
                    self.mlp_dims, self.mlp_in_dim, self.mlp_out_dim, self.mlp_dropout
                )
                self.logger.info("Successfully reinit the parameters of the MLP!!!")
            self.to(self.device)
            self.new_curriculum(self.curriculum_level, tp=self.wit_from)


class CloseCell(nn.Module):
    def __init__(
        self,
        op_cls,
        search_space,
        layer_index,
        num_channels,
        num_out_channels,
        prev_num_channels,
        stride,
        prev_strides,
        use_preprocess,
        preprocess_op_type,
        use_shortcut,
        shortcut_op_type,
        bn_affine,
        num_curriculums,
        **op_kwargs
    ):
        super(CloseCell, self).__init__()
        self._logger = None
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
        self.num_curriculums = num_curriculums

        self._steps = self.search_space.get_layer_num_steps(layer_index)
        self._num_init = self.search_space.num_init_nodes
        if not self.search_space.cellwise_primitives:
            # the same set of primitives for different cg group
            self._primitives = self.search_space.shared_primitives
        else:
            # different set of primitives for different cg group
            self._primitives = self.search_space.cell_shared_primitives[
                self.search_space.cell_layout[layer_index]
            ]

        # initialize self.concat_op, self._out_multiplier (only used for discrete super net)
        self.concat_op = ops.get_concat_op(self.search_space.concat_op)
        if not self.concat_op.is_elementwise:
            expect(
                not self.search_space.loose_end,
                "For shared weights weights manager, when non-elementwise concat op do not "
                "support loose-end search space",
            )
            self._out_multipler = (
                self._steps
                if not self.search_space.concat_nodes
                else len(self.search_space.concat_nodes)
            )
        else:
            # elementwise concat op. e.g. sum, mean
            self._out_multipler = 1

        self.preprocess_ops = nn.ModuleList()
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[: len(prev_num_channels)])
        for prev_c, prev_s in zip(prev_num_channels, prev_strides):
            if not self.use_preprocess:
                # stride/channel not handled!
                self.preprocess_ops.append(ops.Identity())
                continue
            if self.preprocess_op_type is not None:
                # specificy other preprocess op
                preprocess = ops.get_op(self.preprocess_op_type)(
                    C=prev_c, C_out=num_channels, stride=int(prev_s), affine=bn_affine
                )
            else:
                if prev_s > 1:
                    # need skip connection, and is not the connection from the input image
                    preprocess = ops.FactorizedReduce(
                        C_in=prev_c, C_out=num_channels, stride=prev_s, affine=bn_affine
                    )
                else:  # prev_c == _steps * num_channels or inputs
                    preprocess = ops.ReLUConvBN(
                        C_in=prev_c,
                        C_out=num_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        affine=bn_affine,
                    )
            self.preprocess_ops.append(preprocess)
        assert len(self.preprocess_ops) == self._num_init

        if self.use_shortcut:
            self.shortcut_reduction_op = ops.get_op(self.shortcut_op_type)(
                C=prev_num_channels[-1],
                C_out=self.num_out_channel(),
                stride=self.stride,
                affine=True,
            )

        self.curriculums = []
        for i in range(self.num_curriculums):
            curriculum = op_cls(
                self.num_channels,
                self.num_out_channels,
                stride=1,
                primitives=self._primitives,
                bn_affine=bn_affine,
                **op_kwargs
            )
            self.curriculums.append(curriculum)
            self.add_module("c_normal_{}".format(i + 1), curriculum)

        if self.stride != 1:
            for i in range(self.num_curriculums):
                curriculum = op_cls(
                    self.num_channels,
                    self.num_out_channels,
                    stride=self.stride,
                    primitives=self._primitives,
                    bn_affine=bn_affine,
                    **op_kwargs
                )
                self.curriculums.append(curriculum)
                self.add_module("c_reduce_{}".format(i + 1), curriculum)

    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def _new_curriculum(self, level, tp, fst):
        assert tp in ["random", "prev"]
        if level > 1:
            to_ = level - 1
            from_ = random.randint(fst, to_ - 1) if tp == "random" else to_ - 1

            params_1 = self.curriculums[from_].state_dict()
            params_2 = self.curriculums[to_].state_dict()
            assert len(params_1) == len(params_2)

            for par1, par2 in zip(params_1, params_2):
                params_2[par2] += params_1[par1]

            self.curriculums[to_].load_state_dict(params_2)
            self.logger.info(
                "Successfully add a new global op weight from {} to {}".format(
                    from_, to_
                )
            )

    def new_curriculum(self, level, tp):
        self._new_curriculum(level, tp, 0)
        if (level > 1) and (self.stride != 1):
            self._new_curriculum(level + self.num_curriculums, tp, self.num_curriculums)

    def _forward(self, inputs, op, gate, fst, is_training):
        if is_training:
            return sum(
                [
                    gate[i] * self.curriculums[fst + i](inputs, op)
                    for i in range(gate.shape[0])
                ]
            )
        else:
            _, ind = torch.max(gate, 0)
            return self.curriculums[fst + ind.item()](inputs, op)

    def num_out_channel(self):
        return self.num_out_channels * self._out_multipler

    def forward(
        self, inputs, genotype_grouped, is_training
    ):  # pylint: disable=arguments-differ
        genotype_grouped, arch, gate = genotype_grouped
        conns_grouped, concat_nodes = genotype_grouped
        assert self._num_init == len(inputs)
        if self.use_preprocess:
            states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]
        else:
            states = [s for s in inputs]

        ind = -1
        for to_, connections in conns_grouped:
            state_to_ = 0.0
            for op_type, from_, _ in connections:
                ind = ind + 1
                fst = (
                    self.num_curriculums
                    if (from_ < self.search_space.num_init_nodes) and (self.stride != 1)
                    else 0
                )
                state_to_ = state_to_ + self._forward(
                    states[from_], op_type, gate[ind], fst, is_training
                )
            states.append(state_to_)

        out = self.concat_op([states[ind] for ind in concat_nodes])
        if self.use_shortcut and self.layer_index != 0:
            out = out + self.shortcut_reduction_op(inputs[-1])
        return out
