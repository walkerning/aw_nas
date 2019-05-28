# -*- coding: utf-8 -*-
"""
Discrete shared weights RNN super net.
"""

#pylint: disable=invalid-name

import six
import torch

from aw_nas import utils, assert_rollout_type

from aw_nas.weights_manager.rnn_shared import RNNSharedNet, RNNSharedCell, RNNSharedOp
from aw_nas.weights_manager import SubCandidateNet # candidateNet implementation can be reused

__all__ = ["RNNSubCandidateNet", "RNNSuperNet"]

class RNNSubCandidateNet(SubCandidateNet):
    def forward(self, inputs, hiddens): #pylint: disable=arguments-differ
        # make a copy of the hiddens and forward
        hiddens_copy = [hid.clone() for hid in hiddens]
        logits, raw_outs, outs, next_hiddens \
            = self.super_net.forward(inputs, self.genotypes, hiddens=hiddens_copy)
        # update hiddens in place
        for hid, n_hid in zip(hiddens, next_hiddens):
            hid.data.copy_(n_hid.data)
        return logits, raw_outs, outs, next_hiddens

class RNNSuperNet(RNNSharedNet):
    """
    A rnn super network
    """
    NAME = "rnn_supernet"

    def __init__(
            self, search_space, device,
            num_tokens, num_emb=300, num_hid=300,
            tie_weight=True, decoder_bias=True,
            share_primitive_weights=False, batchnorm_edge=False, batchnorm_out=True,
            # training
            max_grad_norm=5.0,
            # dropout probs
            dropout_emb=0., dropout_inp0=0., dropout_inp=0., dropout_hid=0., dropout_out=0.,
            candidate_member_mask=True, candidate_cache_named_members=False,
            candidate_virtual_parameter_only=False):
        super(RNNSuperNet, self).__init__(
            search_space, device,
            cell_cls=RNNDiscreteSharedCell, op_cls=RNNDiscreteSharedOp,
            num_tokens=num_tokens, num_emb=num_emb, num_hid=num_hid,
            tie_weight=tie_weight, decoder_bias=decoder_bias,
            share_primitive_weights=share_primitive_weights,
            batchnorm_edge=batchnorm_edge, batchnorm_out=batchnorm_out,
            max_grad_norm=max_grad_norm,
            dropout_emb=dropout_emb, dropout_inp0=dropout_inp0, dropout_inp=dropout_inp,
            dropout_hid=dropout_hid, dropout_out=dropout_out)

        # candidate net with/without parameter mask
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only

    def forward(self, inputs, genotypes, hiddens): #pylint: disable=arguments-differ
        """
        Returns:
            logits: Tensor(bptt_steps, batch_size, num_tokens)
            raw_outs: Tensor(bptt_steps, batch_size, num_hid)
            droped_outs: Tensor(bptt_steps, batch_size, num_hid)
            next_hiddens: List(Tensor(num_hid))[num_layers]
        """
        batch_size = inputs.size(1)
        time_steps = inputs.size(0)
        # embedding the inputs
        emb = self.encoder(inputs)
        emb = self.lockdrop(emb, self.dropout_inp0)

        # variational dropout masks for inputs/hidden for every layer
        if self.training:
            x_masks = [utils.mask2d(batch_size, self.num_emb,
                                    keep_prob=1.-self.dropout_inp, device=inputs.device)
                       for _ in range(self._num_layers)]
            h_masks = [utils.mask2d(batch_size, self.num_hid,
                                    keep_prob=1.-self.dropout_hid, device=inputs.device)
                       for _ in range(self._num_layers)]
        else:
            x_masks = h_masks = [None] * self._num_layers

        all_outs = []
        for t in range(time_steps):
            next_hiddens = []
            for i_layer in range(self._num_layers):
                if i_layer == 0:
                    layer_inputs = emb[t]
                else:
                    layer_inputs = next_hiddens[-1]
                next_hidden = self.cells[i_layer](layer_inputs, hiddens[i_layer],
                                                  x_masks[i_layer], h_masks[i_layer], genotypes)
                next_hiddens.append(next_hidden)
            # output of this time step
            all_outs.append(next_hiddens[-1])
            # new hiddens
            hiddens = next_hiddens

        raw_outs = torch.stack(all_outs)
        # dropout output
        droped_outs = self.lockdrop(raw_outs, self.dropout_out)

        # decoder to logits
        logits = self.decoder(droped_outs.view(-1, self.num_hid))\
                     .view(-1, batch_size, self.num_tokens)

        # next hidden states: list of size num_layers, tensors of size (num_hid)
        next_hiddens = [next_hid.detach_() for next_hid in next_hiddens]
        return logits, raw_outs, droped_outs, next_hiddens

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return RNNSubCandidateNet(self, rollout,
                                  member_mask=self.candidate_member_mask,
                                  cache_named_members=self.candidate_cache_named_members,
                                  virtual_parameter_only=self.candidate_virtual_parameter_only)

    @classmethod
    def rollout_type(cls):
        return assert_rollout_type("discrete")

    def sub_named_members(self, genotypes,
                          prefix="", member="parameters"):
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
    def _compute_init_state(self, x, h, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h], dim=-1)
        xh_prev = self.w_prev(xh_prev)
        if self.batchnorm_edge:
            xh_prev = self.bn_prev(xh_prev)

        c0, h0 = torch.split(xh_prev, self.num_hid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h + c0 * (h0 - h)
        return s0

    def forward(self, inputs, hidden, x_mask, h_mask, genotypes): #pylint: disable=arguments-differ
        """
        Cell forward, forward for one timestep.
        """
        genotype, concat_ = genotypes

        s0 = self._compute_init_state(inputs, hidden, x_mask, h_mask)
        states = {0: s0}

        for op_type, from_, to_ in genotype:
            s_prev = states[from_]
            if self.training:
                s_prev = s_prev * h_mask

            out = self.edges[from_][to_](s_prev, op_type)

            if to_ in states:
                states[to_] = states[to_] + out
            else:
                states[to_] = out

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
    def forward(self, s_prev, op_type): #pylint: disable=arguments-differ
        op_ind = self.primitives.index(op_type)
        if self.share_w:
            ch = self.W(s_prev)
        else:
            ch = self.Ws[op_ind](s_prev)
        if self.batch_norm:
            ch = self.bn(ch)
        c, h = torch.split(ch, self.num_hid, dim=-1)
        c = c.sigmoid()
        h = self.p_ops[op_ind](h)
        s = s_prev + c * (h - s_prev)
        return s

    def sub_named_members(self, op_type,
                          prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name in {"Ws", "p_ops"}:
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix+mod_name):
                yield n, v

        op_ind = self.primitives.index(op_type)
        if not self.share_w:
            for n, v in getattr(self.Ws[op_ind], "named_" + member)(prefix="{}Ws.{}"\
                                                                    .format(prefix, op_ind)):
                yield n, v
        for n, v in getattr(self.p_ops[op_ind], "named_" + member)(prefix="{}p_ops.{}"\
                                                                   .format(prefix, op_ind)):
            # for now, p_ops do not have any parameters
            yield n, v
