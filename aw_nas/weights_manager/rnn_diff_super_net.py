"""
RNN supernet for differentiable rollouts.
"""

#pylint: disable=invalid-name

import torch
from torch import nn

from aw_nas import assert_rollout_type, ops
from aw_nas.rollout.base import DartsArch
from aw_nas.weights_manager.diff_super_net import DiffSubCandidateNet
from aw_nas.weights_manager.rnn_shared import (
    RNNSharedNet, RNNSharedCell, RNNSharedOp, INIT_RANGE
)
from aw_nas.utils.exception import expect, ConfigException

__all__ = ["RNNDiffSubCandidateNet", "RNNDiffSuperNet"]

# TODO: fix RNN diff super net

class RNNDiffSubCandidateNet(DiffSubCandidateNet):
    def forward(self, inputs, hiddens, detach_arch=True): #pylint: disable=arguments-differ
        if detach_arch:
            arch = [
                DartsArch(
                    op_weights=op_weights.detach(),
                    edge_norms=edge_norms.detach() if edge_norms is not None else None
                ) for op_weights, edge_norms in self.arch
            ]
        else:
            arch = self.arch
        # make a copy of the hiddens and forward
        hiddens_copy = hiddens.clone()
        logits, raw_outs, outs, next_hiddens \
            = self.super_net.forward(inputs, arch, hiddens=hiddens_copy, detach_arch=detach_arch)
        # update hiddens in place
        hiddens.data.copy_(next_hiddens.data)
        return logits, raw_outs, outs, next_hiddens


class RNNDiffSuperNet(RNNSharedNet):
    """
    A rnn super network
    """
    NAME = "rnn_diff_supernet"

    def __init__(
            self, search_space, device, num_tokens,
            rollout_type="differentiable", num_emb=300, num_hid=300,
            tie_weight=True, decoder_bias=True,
            share_primitive_weights=False, share_from_weights=False,
            batchnorm_step=False,
            batchnorm_edge=False, batchnorm_out=True,
            # training
            max_grad_norm=5.0,
            # dropout probs
            dropout_emb=0., dropout_inp0=0., dropout_inp=0., dropout_hid=0., dropout_out=0.,
            candidate_virtual_parameter_only=False):
        expect(not search_space.loose_end,
               "Differentiable NAS searching do not support loose-ended search_space",
               ConfigException)
        if share_from_weights:
            # darts
            cell_cls = RNNDiffSharedFromCell
        else:
            cell_cls = RNNDiffSharedCell
        super(RNNDiffSuperNet, self).__init__(
            search_space, device, rollout_type,
            cell_cls=cell_cls, op_cls=RNNDiffSharedOp,
            num_tokens=num_tokens, num_emb=num_emb, num_hid=num_hid,
            tie_weight=tie_weight, decoder_bias=decoder_bias,
            share_primitive_weights=share_primitive_weights, share_from_weights=share_from_weights,
            batchnorm_step=batchnorm_step,
            batchnorm_edge=batchnorm_edge, batchnorm_out=batchnorm_out,
            max_grad_norm=max_grad_norm,
            dropout_emb=dropout_emb, dropout_inp0=dropout_inp0, dropout_inp=dropout_inp,
            dropout_hid=dropout_hid, dropout_out=dropout_out)

        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return RNNDiffSubCandidateNet(self, rollout,
                                      virtual_parameter_only=self.candidate_virtual_parameter_only)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("differentiable")]

class RNNDiffSharedFromCell(nn.Module):
    def __init__(self, search_space, device, op_cls, num_emb, num_hid,
                 share_primitive_w, batchnorm_step, batchnorm_out, **kwargs):
        super(RNNDiffSharedFromCell, self).__init__()

        self.num_emb = num_emb
        self.num_hid = num_hid
        self.batchnorm_step = batchnorm_step
        self.batchnorm_out = batchnorm_out
        self._steps = search_space.num_steps
        self._num_init = search_space.num_init_nodes
        self._primitives = search_space.shared_primitives

        # the first step, convert input x and previous hidden
        self.w_prev = nn.Linear(num_emb + num_hid, 2 * num_hid, bias=False)
        self.w_prev.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)

        if self.batchnorm_step:
            # batchnorm after every step (just as in darts's implementation)
            # self.bn_steps = nn.ModuleList([nn.BatchNorm1d(num_hid, affine=False)
            #                                for _ in range(self._steps+1)])

            ## darts: (but seems odd...)
            self.bn_step = nn.BatchNorm1d(num_hid, affine=False)
            self.bn_steps = [self.bn_step] * (self._steps + 1)

        if self.batchnorm_out:
            # the out bn
            self.bn_out = nn.BatchNorm1d(num_hid, affine=True)

        self.step_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(num_hid, 2*num_hid)\
                         .uniform_(-INIT_RANGE, INIT_RANGE))
            for _ in range(self._steps)])
        [mod.weight.data.uniform_(-INIT_RANGE, INIT_RANGE) for mod in self.step_weights]

        self.p_ops = nn.ModuleList()
        for primitive in self._primitives:
            op = ops.get_op(primitive)()
            self.p_ops.append(op)


    def forward(self, inputs, hidden, x_mask, h_mask, archs, detach_arch): #pylint: disable=arguments-differ
        """
        Cell forward, forward for one timestep.
        """
        arch = archs[0] # only one element now, for only one cell group

        s0 = self._compute_init_state(inputs, hidden, x_mask, h_mask)
        if self.batchnorm_step:
            s0 = self.bn_steps[0](s0)

        states = s0.unsqueeze(0)
        offset = 0
        for i in range(self._steps):
            to_ = i + self._num_init # =1
            if self.training:
                inp_states = states * h_mask.unsqueeze(0)
            else:
                inp_states = states
            ch = inp_states.view(-1, self.num_hid).mm(self.step_weights[i])\
                                                  .view(to_, -1, 2*self.num_hid)
            cs, hs = torch.split(ch, self.num_hid, dim=-1)
            cs = cs.sigmoid()
            # weights: (num_from_nodes, num_pritmives);
            # states: (num_from_nodes, batch_size, num_hid);
            weights = arch[offset:offset+to_].op_weights
            # unweighted: (num_from_nodes, batch_size, num_hid, num_primitives)
            unweighted = states.unsqueeze(-1) + cs.unsqueeze(-1) * \
                         torch.stack([op(hs) - states for op in self.p_ops], dim=-1)
            new_state = torch.sum(torch.sum(unweighted * weights.view(to_, 1, 1, -1), dim=-1),
                                  dim=0)
            if self.batchnorm_step:
                new_state = self.bn_steps[to_](new_state)
            offset += len(states)
            states = torch.cat([states, new_state.unsqueeze(0)], dim=0)

        # average the ends, in differential searching, all the intermediate ends are averaged
        output = torch.mean(states[self._num_init:], dim=0)
        if self.batchnorm_out:
            # batchnorm
            output = self.bn_out(output)
        return output

    def _compute_init_state(self, x, h, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h], dim=-1)
        xh_prev = self.w_prev(xh_prev)
        c0, h0 = torch.split(xh_prev, self.num_hid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h + c0 * (h0 - h)
        return s0

class RNNDiffSharedCell(RNNSharedCell):
    def forward(self, inputs, hidden, x_mask, h_mask, archs, detach_arch): #pylint: disable=arguments-differ
        """
        Cell forward, forward for one timestep.
        """
        arch = archs[0] # only one element now, for only one cell group

        s0 = self._compute_init_state(inputs, hidden, x_mask, h_mask)
        states = s0.unsqueeze(0)
        offset = 0

        for i in range(self._steps):
            to_ = i + self._num_init # =1
            if self.training:
                inp_states = states * h_mask.unsqueeze(0)
            else:
                inp_states = states
            act_lst = [
                self.edges[from_][to_](
                    inp,
                    arch.op_weights[offset+from_],
                    s_prev,
                    detach_arch=detach_arch
                )
                for from_, (inp, s_prev) in enumerate(zip(inp_states, states))
            ]
            new_state = sum(act_lst)
            offset += len(states)
            states = torch.cat([states, new_state.unsqueeze(0)], dim=0)

        # average the ends, in differential searching, all the intermediate ends are averaged
        output = torch.mean(states[self._num_init:], dim=0)
        if self.batchnorm_out:
            # batchnorm
            output = self.bn_out(output)
        return output

class RNNDiffSharedOp(RNNSharedOp):
    def forward(self, inputs, weights, s_prev, detach_arch): #pylint: disable=arguments-differ
        s_update = 0.
        for op_ind, (w, op) in enumerate(zip(weights, self.p_ops)):
            if detach_arch and w.item() == 0:
                continue
            ch = (self.W if self.share_w else self.Ws[op_ind])(inputs)
            if self.batch_norm:
                ch = self.bn(ch)
            c, h = torch.split(ch, self.num_hid, dim=-1)
            c = c.sigmoid()
            h = op(h)
            update = c * (h - s_prev)
            if w.item() == 0:
                update.detach_()
            s_update += w * update
        return s_prev + s_update
