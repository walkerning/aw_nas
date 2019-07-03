# -*- coding: utf-8 -*-
"""
Discrete shared weights RNN super net.
"""

#pylint: disable=invalid-name

import itertools
import six
import torch

from aw_nas import assert_rollout_type, utils
from aw_nas.weights_manager.rnn_shared import RNNSharedNet, RNNSharedCell, RNNSharedOp
from aw_nas.weights_manager import SubCandidateNet # candidateNet implementation can be reused

__all__ = ["RNNSubCandidateNet", "RNNSuperNet"]

class RNNSubCandidateNet(SubCandidateNet):
    def forward(self, inputs, hiddens): #pylint: disable=arguments-differ
        # make a copy of the hiddens and forward
        hiddens_copy = hiddens.clone()
        logits, raw_outs, outs, next_hiddens \
            = self.super_net.forward(inputs, self.genotypes, hiddens=hiddens_copy)
        # update hiddens in place
        hiddens.data.copy_(next_hiddens.data)
        return logits, raw_outs, outs, next_hiddens

class RNNSuperNet(RNNSharedNet):
    """
    A rnn super network
    """
    NAME = "rnn_supernet"

    def __init__(
            self, search_space, device, num_tokens,
            rollout_type="discrete",
            num_emb=300, num_hid=300,
            tie_weight=True, decoder_bias=True,
            share_primitive_weights=False, share_from_weights=False,
            batchnorm_step=False,
            batchnorm_edge=False, batchnorm_out=True,
            # training
            max_grad_norm=5.0,
            # dropout probs
            dropout_emb=0., dropout_inp0=0., dropout_inp=0., dropout_hid=0., dropout_out=0.,
            candidate_member_mask=True, candidate_cache_named_members=False,
            candidate_virtual_parameter_only=False):
        super(RNNSuperNet, self).__init__(
            search_space, device, rollout_type,
            cell_cls=RNNDiscreteSharedCell, op_cls=RNNDiscreteSharedOp,
            num_tokens=num_tokens, num_emb=num_emb, num_hid=num_hid,
            tie_weight=tie_weight, decoder_bias=decoder_bias,
            share_primitive_weights=share_primitive_weights, share_from_weights=share_from_weights,
            batchnorm_step=batchnorm_step,
            batchnorm_edge=batchnorm_edge, batchnorm_out=batchnorm_out,
            max_grad_norm=max_grad_norm,
            dropout_emb=dropout_emb, dropout_inp0=dropout_inp0, dropout_inp=dropout_inp,
            dropout_hid=dropout_hid, dropout_out=dropout_out)

        # candidate net with/without parameter mask
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return RNNSubCandidateNet(self, rollout,
                                  member_mask=self.candidate_member_mask,
                                  cache_named_members=self.candidate_cache_named_members,
                                  virtual_parameter_only=self.candidate_virtual_parameter_only)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("discrete")]

    def sub_named_members(self, genotypes,
                          prefix="", member="parameters",
                          check_visited=False):
        prefix = prefix + ("." if prefix else "")

        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "cells":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix+mod_name):
                yield n, v

        # currently, as only `num_init_nodes==1` is supported,
        # and the connection pattern between cell layers is sequential,
        # all cells will certainly be forwarded
        for idx, cell in enumerate(self.cells):
            for n, v in cell.sub_named_members(genotypes,
                                               prefix=prefix + "cells.{}".format(idx),
                                               member=member):
                yield n, v


class RNNDiscreteSharedCell(RNNSharedCell):
    def forward(self, inputs, hidden, x_mask, h_mask, genotypes): #pylint: disable=arguments-differ
        """
        Cell forward, forward for one timestep.
        """
        genotype, concat_ = genotypes

        s0 = self._compute_init_state(inputs, hidden, x_mask, h_mask)
        if self.batchnorm_step:
            s0 = self.bn_steps[0](s0)

        states = {0: s0}

        len_genotype = len(genotype)

        for i, (op_type, from_, to_) in enumerate(genotype):
            s_prev = states[from_]
            s_inputs = s_prev
            if self.training:
                s_inputs = s_prev * h_mask

            # In discrete supernet, as for now, `num_node_inputs==1`;
            # so, no matter `share_from_w` or not, the number of calculation is the same.
            # if `share_from_w==True and num_node_inputs>1`, for efficiency, should group genotype
            # connections by `to_`, and merge `from_` states before multiply by step weights.
            # HOLD for now. Will be implemented if `num_node_inputs>1` is considered in the future.
            out = self.edges[from_][to_](s_inputs, op_type, s_prev)

            if to_ in states:
                states[to_] = states[to_] + out
            else:
                states[to_] = out

            to_finish = i == len_genotype-1 or genotype[i+1][2] != to_
            if self.batchnorm_step and to_finish:
                states[to_] = self.bn_steps[to_](states[to_])


        # average the ends
        output = torch.mean(torch.stack([states[i] for i in concat_]), 0)
        if self.batchnorm_out:
            # batchnorm
            output = self.bn_out(output)
        return output

    def sub_named_members(self, genotypes,
                          prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")

        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "edge_mod":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix+mod_name):
                yield n, v

        genotype = genotypes[0]

        for op_type, from_, to_ in genotype:
            edge_share_op = self.edges[from_][to_]
            for n, v in edge_share_op.sub_named_members(
                    op_type,
                    prefix=prefix + "edge_mod.f_{}_t_{}".format(from_, to_),
                    member=member):
                yield n, v


class RNNDiscreteSharedOp(RNNSharedOp):
    def forward(self, inputs, op_type, s_prev): #pylint: disable=arguments-differ
        op_ind = self.primitives.index(op_type)
        ch = (self.W if self.share_w else self.Ws[op_ind])(inputs)
        if self.batch_norm:
            ch = self.bn(ch)
        c, h = torch.split(ch, self.num_hid, dim=-1)
        c = c.sigmoid()
        h = self.p_ops[op_ind](h)
        s = s_prev + c * (h - s_prev)
        return s

    def sub_named_members(self, op_type,
                          prefix="", member="parameters"):
        # the common modules that will be forwarded by every candidate
        return itertools.chain(utils.submodule_named_members(
            self, member, prefix + ("." if prefix else ""),
            not_include=("Ws", "p_ops")
        ), self._sub_named_members(op_type, prefix, member))

    def _sub_named_members(self, op_type,
                           prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        op_ind = self.primitives.index(op_type)
        if not self.share_w:
            for n, v in getattr(self.Ws[op_ind], "named_" + member)(prefix="{}Ws.{}"\
                                                                    .format(prefix, op_ind)):
                yield n, v
        for n, v in getattr(self.p_ops[op_ind], "named_" + member)(prefix="{}p_ops.{}"\
                                                                   .format(prefix, op_ind)):
            # for now, p_ops do not have any parameters
            yield n, v
