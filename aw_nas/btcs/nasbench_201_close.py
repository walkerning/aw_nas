import os
import re
import copy
import random
import pickle
import itertools
import collections
from collections import defaultdict, OrderedDict
import contextlib

import yaml
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from aw_nas.utils import getLogger
from aw_nas.btcs.nasbench_201 import *

from functools import partial


class NB201CloseCandidateNet(NB201CandidateNet):
    def __init__(
        self,
        super_net,
        rollout,
        member_mask,
        gpus=tuple(),
        cache_named_members=False,
        eval_no_grad=True,
        for_eval=False,
    ):
        super(NB201CloseCandidateNet, self).__init__(
            super_net, rollout, member_mask, gpus, cache_named_members, eval_no_grad
        )

        self.for_eval = for_eval
        self.gate = rollout.gate
        self.rollout = rollout

    def _forward(self, inputs, **kwargs):
        return self.super_net.forward(
            inputs, (self.genotype_arch, self.gate), is_training=not self.for_eval
        )


class NB201CloseNet(BaseNB201SharedNet):
    NAME = "nasbench-201-close"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-201",
        gpus=tuple(),
        num_classes=10,
        init_channels=16,
        stem_multiplier=1,
        max_grad_norm=5.0,
        dropout_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        reduce_affine=True,
        cell_bn_affine=False,
        candidate_member_mask=True,
        candidate_cache_named_members=False,
        candidate_eval_no_grad=True,
        iso_mapping_file=None,
        # For CLOSE
        num_curriculums=6,
        cl_schedule=[1, 101, 201, 401, 601, 801],
        # For GATES
        op_embedding_dim=48,
        node_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        share_op_attention=False,
        gcn_kwargs=None,
        use_bn=False,
        use_final_only=False,
        share_self_op_emb=False,
        dropout=0.0,
        schedule_cfg=None,
        mlp_dims=[200],
        mlp_dropout=0.1,
        reconstruct_only_last=False,
        reconstruct_inherit=False,
        gate_type="arch-level",
        wit_from="prev",
        eval_arch=False,
    ):
        super(NB201CloseNet, self).__init__(
            search_space,
            device,
            cell_cls=partial(NB201CloseCell, num_curriculums=num_curriculums),
            op_cls=NB201SharedOp,
            rollout_type=rollout_type,
            gpus=gpus,
            num_classes=num_classes,
            init_channels=init_channels,
            stem_multiplier=stem_multiplier,
            max_grad_norm=max_grad_norm,
            dropout_rate=dropout_rate,
            use_stem=use_stem,
            stem_stride=stem_stride,
            stem_affine=stem_affine,
            reduce_affine=reduce_affine,
            cell_bn_affine=cell_bn_affine,
            candidate_member_mask=candidate_member_mask,
            candidate_cache_named_members=candidate_cache_named_members,
            candidate_eval_no_grad=candidate_eval_no_grad,
            iso_mapping_file=iso_mapping_file,
        )

        # CL configs
        self.num_positions = 6
        self.num_curriculums = num_curriculums
        self.cl_schedule = cl_schedule
        self.curriculum_level = 1

        # MLP configs
        self.mlp_dims = mlp_dims
        self.mlp_dropout = mlp_dropout
        self.reconstruct_only_last = reconstruct_only_last
        self.reconstruct_inherit = reconstruct_inherit

        # construct predictor (GATES & MLP)
        self.embedder = NasBench201FlowArchEmbedder(
            search_space,
            op_embedding_dim,
            node_embedding_dim,
            hid_dim,
            gcn_out_dims,
            share_op_attention,
            gcn_kwargs,
            use_bn,
            use_final_only,
            share_self_op_emb,
            dropout,
            schedule_cfg,
        )

        self.mlp_in_dim = (
            self.embedder.out_dim
            if gate_type == "arch-level"
            else self.embedder.out_dim * 2
        )
        self.mlp_out_dim = (
            self.num_positions * self.num_curriculums
            if gate_type == "arch-level"
            else self.num_curriculums
        )
        self.mlp = self.construct_mlp(
            self.mlp_dims, self.mlp_in_dim, self.mlp_out_dim, self.mlp_dropout
        )

        self.gate_type = gate_type
        self.wit_from = wit_from
        self.eval_arch = eval_arch
        self.to(self.device)

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

    # ---- APIs ----
    def random_gate(self, rollout):
        node_embs = self.embedder(
            np.expand_dims(rollout.arch.astype(np.float32), 0), return_all=True
        ).squeeze(0)
        gate = []
        for to_ in range(1, self._num_vertices):
            for from_ in range(to_):
                index = torch.randint(0, self.curriculum_level, (1,))
                logit = torch.nn.functional.one_hot(index, self.curriculum_level)[0]
                gate.append(logit)
        return gate

    def node_level_gate(self, rollout, softmax):
        node_embs = self.embedder(
            np.expand_dims(rollout.arch.astype(np.float32), 0), return_all=True
        ).squeeze(0)
        gate = []
        for to_ in range(1, self._num_vertices):
            for from_ in range(to_):
                logit = self.mlp(torch.cat((node_embs[from_], node_embs[to_]), dim=-1))
                logit = logit[: self.curriculum_level]
                if softmax == False:
                    gate.append(logit)
                    continue
                prob, _ = utils.gumbel_softmax(logit, temperature=5.0, hard=False)
                gate.append(utils.straight_through(prob))
        return gate

    def arch_level_gate(self, rollout):
        score = self.mlp(
            self.embedder(np.expand_dims(rollout.arch.astype(np.float32), 0))
        )
        score = score.squeeze(0).reshape(self.num_positions, self.num_curriculums)
        gate = []
        for i in range(score.shape[0]):
            logit = score[i][: self.curriculum_level]
            prob, _ = utils.gumbel_softmax(logit, temperature=5.0, hard=False)
            gate.append(utils.straight_through(prob))
        return gate

    def assemble_candidate(self, rollout, for_eval=False, softmax=True):
        if self.gate_type == "arch-level":
            rollout.gate = self.arch_level_gate(rollout)
        elif self.gate_type == "node-level":
            rollout.gate = self.node_level_gate(rollout, softmax=softmax)
        elif self.gate_type == "random":
            rollout.gate = self.random_gate(rollout)

        return NB201CloseCandidateNet(
            self,
            rollout,
            gpus=self.gpus,
            member_mask=self.candidate_member_mask,
            cache_named_members=self.candidate_cache_named_members,
            eval_no_grad=self.candidate_eval_no_grad,
            for_eval=for_eval,
        )

    def new_curriculum(self, level, tp):
        assert tp in ["random", "prev"]
        for cell in self.cells:
            if isinstance(cell, NB201CloseCell):
                cell.new_curriculum(level, tp)
        self.logger.info("Successfully add a new curriculum of level {}".format(level))

    def inherit(self, w, x):
        w[x] += w[x - 1]
        w.requires_grad = True
        return w

    def on_epoch_start(self, epoch):
        super(NB201CloseNet, self).on_epoch_start(epoch)
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


class NB201CloseCell(nn.Module):
    def __init__(
        self,
        op_cls,
        search_space,
        layer_index,
        num_channels,
        num_out_channels,
        stride,
        bn_affine=False,
        num_curriculums=6,
    ):
        super(NB201CloseCell, self).__init__()
        self._logger = None

        self.search_space = search_space
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index
        self.num_curriculums = num_curriculums

        self._vertices = self.search_space.num_vertices
        self._primitives = self.search_space.ops_choices

        self.curriculums = []
        for i in range(self.num_curriculums):
            curriculum = op_cls(
                self.num_channels,
                self.num_out_channels,
                stride=self.stride,
                primitives=self._primitives,
                bn_affine=bn_affine,
            )
            self.curriculums.append(curriculum)
            self.add_module("c_{}".format(i + 1), curriculum)

    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def new_curriculum(self, level, tp):
        assert tp in ["random", "prev"]
        if level > 1:
            to_ = level - 1
            from_ = random.randint(0, to_ - 1) if tp == "random" else to_ - 1

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

    def _forward(self, inputs, op, gate, is_training):
        op = int(op)
        if is_training:
            return sum(
                [
                    gate[i] * self.curriculums[i](inputs, op)
                    for i in range(gate.shape[0])
                ]
            )
        else:
            _, ind = torch.max(gate, 0)
            return self.curriculums[ind.item()](inputs, op)

    def forward(self, inputs, genotype, is_training):
        assert len(genotype) == 2
        genotype, gate = genotype

        valid_input = [0]
        for to_ in range(1, self.search_space.num_vertices):
            for input_ in valid_input:
                if genotype[to_][input_] > 0:
                    valid_input.append(to_)
                    break

        valid_output = [self.search_space.num_vertices - 1]
        for from_ in range(self.search_space.num_vertices - 2, -1, -1):
            for output_ in valid_output:
                if genotype[output_][from_] > 0:
                    valid_output.append(from_)

        states_ = [inputs]
        ind = -1

        for to_ in range(1, self._vertices):
            state_ = torch.zeros(inputs.shape).to(inputs.device)
            for from_ in range(to_):
                ind = ind + 1
                if from_ in valid_input and to_ in valid_output:
                    out = self._forward(
                        states_[from_], genotype[to_][from_], gate[ind], is_training
                    )
                    state_ = state_ + out
            states_.append(state_)
        return states_[-1]
