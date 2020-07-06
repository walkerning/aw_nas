#pylint: disable=invalid-name

import numpy as np
import torch
from torch import nn

from aw_nas import ops
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.weights_manager.rnn_shared import RNNSharedNet, INIT_RANGE

class RNNGenotypeModel(RNNSharedNet):
    REGISTRY = "final_model"
    NAME = "rnn_model"
    def __init__(self, search_space, device, genotypes,
                 num_tokens, num_emb=300, num_hid=300,
                 tie_weight=True, decoder_bias=True,
                 share_primitive_weights=False, share_from_weights=False,
                 batchnorm_step=False,
                 batchnorm_edge=False, batchnorm_out=True,
                 # training
                 max_grad_norm=5.0,
                 # dropout probs
                 dropout_emb=0., dropout_inp0=0., dropout_inp=0., dropout_hid=0., dropout_out=0.):

        self.genotypes = genotypes
        if isinstance(genotypes, str):
            self.genotypes = eval("search_space.genotype_type({})".format(self.genotypes)) # pylint: disable=eval-used
            self.genotypes = list(self.genotypes._asdict().values())
        # check tos:
        _tos = [conn[2] for conn in self.genotypes[0]]
        (np.argsort(_tos) == np.arange(len(_tos))).all()
        expect((np.argsort(_tos) == np.arange(len(_tos))).all(),
               "genotype must be ordered in the way that `to_node` monotonously increase",
               ConfigException)

        super(RNNGenotypeModel, self).__init__(
            search_space, device,
            cell_cls=RNNGenotypeCell, op_cls=None,
            num_tokens=num_tokens, num_emb=num_emb, num_hid=num_hid,
            tie_weight=tie_weight, decoder_bias=decoder_bias,
            share_primitive_weights=share_primitive_weights, share_from_weights=share_from_weights,
            batchnorm_step=batchnorm_step,
            batchnorm_edge=batchnorm_edge, batchnorm_out=batchnorm_out,
            max_grad_norm=max_grad_norm,
            dropout_emb=dropout_emb, dropout_inp0=dropout_inp0, dropout_inp=dropout_inp,
            dropout_hid=dropout_hid, dropout_out=dropout_out,
            genotypes=self.genotypes) # this genotypes will be used for construction/forward

        self.logger.info("Genotype: %s", self.genotypes)

    def forward(self, inputs, hiddens): #pylint: disable=arguments-differ
        # this genotypes will not be used
        return RNNSharedNet.forward(self, inputs, self.genotypes, hiddens)

    @classmethod
    def supported_rollout_types(cls):
        # this should not be called
        # assert 0, "should not be called"
        return []

    def assemble_candidate(self, *args, **kwargs): #pylint: disable=arguments-differ
        # this will not be called
        assert 0, "should not be called"

class RNNGenotypeCell(nn.Module):
    def __init__(self, search_space, device, op_cls, num_emb, num_hid,
                 share_from_weights, batchnorm_step,
                 batchnorm_edge, batchnorm_out, genotypes, **kwargs):
        super(RNNGenotypeCell, self).__init__()
        self.genotypes = genotypes

        self.search_space = search_space

        self.num_emb = num_emb
        self.num_hid = num_hid
        self.batchnorm_step = batchnorm_step
        self.batchnorm_edge = batchnorm_edge
        self.batchnorm_out = batchnorm_out
        self.share_from_w = share_from_weights
        self._steps = search_space.num_steps
        self._num_init = search_space.num_init_nodes

        # the first step, convert input x and previous hidden
        self.w_prev = nn.Linear(num_emb + num_hid, 2 * num_hid, bias=False)
        self.w_prev.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)

        if self.batchnorm_edge:
            # batchnorm on each edge/connection
            # when `num_node_inputs==1`, there is `step + 1` edges
            # the first bn
            self.bn_prev = nn.BatchNorm1d(num_emb + num_hid, affine=True)
            # other bn
            self.bn_edges = nn.ModuleList([nn.BatchNorm1d(num_emb + num_hid, affine=True)
                                           for _ in range(len(self.genotypes[0]))])

        if self.batchnorm_step:
            # batchnorm after every step (as in darts's implementation)
            self.bn_steps = nn.ModuleList([nn.BatchNorm1d(num_hid, affine=False)
                                           for _ in range(self._steps+1)])

        if self.batchnorm_out:
            # the out bn
            self.bn_out = nn.BatchNorm1d(num_hid, affine=True)

        if self.share_from_w:
            # actually, as `num_node_inputs==1`, thus only one from node is used each step
            # `share_from_w==True/False` are equivalent in final training...
            self.step_weights = nn.ModuleList([
                nn.Linear(num_hid, 2*num_hid, bias=False)
                for _ in range(self._steps)])
            [mod.weight.data.uniform_(-INIT_RANGE, INIT_RANGE) for mod in self.step_weights]

        # initiatiate op on edges
        self.Ws = nn.ModuleList()
        self.ops = nn.ModuleList()
        genotype_, _ = self.genotypes

        for op_type, _, _ in genotype_:
            # edge weights
            op = ops.get_op(op_type)()
            self.ops.append(op)
            if not self.share_from_w:
                W = nn.Linear(self.num_hid, 2 * self.num_hid, bias=False)
                W.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)
                self.Ws.append(W)

    def forward(self, inputs, hidden, x_mask, h_mask, genotypes): #pylint: disable=arguments-differ
        """
        Cell forward, forward for one timestep.
        """
        genotype, concat_ = self.genotypes # self.genotypes == genotypes

        s0 = self._compute_init_state(inputs, hidden, x_mask, h_mask)
        if self.batchnorm_step:
            s0 = self.bn_steps[0](s0)

        states = {0: s0}

        for i, (_, from_, to_) in enumerate(genotype):
            s_prev = states[from_]
            s_inputs = s_prev
            if self.training:
                s_inputs = s_prev * h_mask
            w = self.step_weights[to_-1] if self.share_from_w else self.Ws[i]
            ch = w(s_inputs)
            if self.batchnorm_edge:
                ch = self.bn_edges[i](ch)
            c, h = torch.split(ch, self.num_hid, dim=-1)
            c = c.sigmoid()
            h = self.ops[i](h)
            out = s_prev + c * (h - s_prev)
            if to_ in states:
                states[to_] = states[to_] + out
            else:
                states[to_] = out

            to_finish = i == len(genotype)-1 or genotype[i+1][2] != to_
            if self.batchnorm_step and to_finish:
                # if the calculation of the `to_` step finished, batch norm it
                states[to_] = self.bn_steps[to_](states[to_])

        # average the ends
        output = torch.mean(torch.stack([states[i] for i in concat_]), 0)
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
        if self.batchnorm_edge:
            xh_prev = self.bn_prev(xh_prev)

        c0, h0 = torch.split(xh_prev, self.num_hid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h + c0 * (h0 - h)
        return s0
