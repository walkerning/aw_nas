"""
RNN supernet for differentiable rollouts.
"""

#pylint: disable=invalid-name

import torch

from aw_nas import assert_rollout_type
from aw_nas.weights_manager.diff_super_net import DiffSubCandidateNet
from aw_nas.weights_manager.rnn_shared import RNNSharedNet, RNNSharedCell, RNNSharedOp
from aw_nas.utils.exception import expect, ConfigException

class RNNDiffSubCandidateNet(DiffSubCandidateNet):
    def forward(self, inputs, hiddens, detach_arch=True): #pylint: disable=arguments-differ
        arch = [a.detach() for a in self.arch] if detach_arch else self.arch
        # make a copy of the hiddens and forward
        hiddens_copy = [hid.clone() for hid in hiddens]
        logits, raw_outs, outs, next_hiddens \
            = self.super_net.forward(inputs, arch, hiddens=hiddens_copy, detach_arch=detach_arch)
        # update hiddens in place
        for hid, n_hid in zip(hiddens, next_hiddens):
            hid.data.copy_(n_hid.data)
        return logits, raw_outs, outs, next_hiddens


class RNNDiffSuperNet(RNNSharedNet):
    """
    A rnn super network
    """
    NAME = "rnn_diff_supernet"

    def __init__(
            self, search_space, device,
            num_tokens, num_emb=300, num_hid=300,
            tie_weight=True, decoder_bias=True,
            share_primitive_weights=False, batchnorm_edge=False, batchnorm_out=True,
            # training
            max_grad_norm=5.0,
            # dropout probs
            dropout_emb=0., dropout_inp0=0., dropout_inp=0., dropout_hid=0., dropout_out=0.,
            candidate_virtual_parameter_only=False):
        expect(not search_space.loose_end,
               "Differentiable NAS searching do not support loose-ended search_space",
               ConfigException)
        super(RNNDiffSuperNet, self).__init__(
            search_space, device,
            cell_cls=RNNDiffSharedCell, op_cls=RNNDiffSharedOp,
            num_tokens=num_tokens, num_emb=num_emb, num_hid=num_hid,
            tie_weight=tie_weight, decoder_bias=decoder_bias,
            share_primitive_weights=share_primitive_weights,
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
    def rollout_type(cls):
        return assert_rollout_type("differentiable")


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
            to_ = i + self._num_init
            if self.training:
                inp_states = states * h_mask.unsqueeze(0)
            act_lst = [self.edges[from_][to_](inp, arch[offset+from_],
                                              detach_arch=detach_arch)
                       for from_, inp in enumerate(inp_states)]
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
    def forward(self, s_prev, weights, detach_arch): #pylint: disable=arguments-differ
        s_update = 0.
        for op_ind, (w, op) in enumerate(zip(weights, self.p_ops)):
            if detach_arch and w.item() == 0:
                continue
            ch = (self.W if self.share_w else self.Ws[op_ind])(s_prev)
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
