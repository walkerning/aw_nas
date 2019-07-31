# -*- coding: utf-8 -*-
#pylint: disable=invalid-name

import re
from collections import defaultdict

import six
import torch
from torch import nn

from aw_nas import ops, utils
from aw_nas.weights_manager.base import BaseWeightsManager

# INIT_RANGE = 0.1
INIT_RANGE = 0.04

class RNNSharedNet(BaseWeightsManager, nn.Module):
    def __init__(
            self, search_space, device, rollout_type, cell_cls, op_cls,
            num_tokens, num_emb, num_hid,
            tie_weight, decoder_bias,
            share_primitive_weights, share_from_weights,
            batchnorm_step, batchnorm_edge, batchnorm_out,
            # training
            max_grad_norm,
            # dropout probs
            dropout_emb, dropout_inp0, dropout_inp, dropout_hid, dropout_out,
            # kwargs that will be passed to cell init
            **kwargs):
        super(RNNSharedNet, self).__init__(search_space, device, rollout_type)
        nn.Module.__init__(self)

        # model configs
        self.num_tokens = num_tokens
        self.num_emb = num_emb
        self.num_hid = num_hid
        self.tie_weight = tie_weight
        # search space
        self._num_layers = self.search_space.num_layers

        # training configs
        self.max_grad_norm = max_grad_norm
        self.dropout_emb = dropout_emb
        self.dropout_inp0 = dropout_inp0
        self.dropout_inp = dropout_inp
        self.dropout_hid = dropout_hid
        self.dropout_out = dropout_out

        # modules
        self.encoder = ops.EmbeddingDropout(num_tokens, num_emb, dropout=dropout_emb)
        self.decoder = nn.Linear(num_emb, num_tokens, bias=decoder_bias)
        self.lockdrop = ops.LockedDropout()

        if tie_weight:
            self.decoder.weight = self.encoder.weight

        # support independent emb size/hid size even `tie_weight` is true
        # make last layer's hidden size to be `num_emb` when `tie_weight` is true
        self.cells = nn.ModuleList([cell_cls(
            search_space, device, op_cls,
            num_emb, num_emb if i_layer == self._num_layers-1 and self.tie_weight else num_hid,
            share_primitive_w=share_primitive_weights,
            share_from_weights=share_from_weights,
            batchnorm_step=batchnorm_step,
            batchnorm_edge=batchnorm_edge, batchnorm_out=batchnorm_out,
            **kwargs
        ) for i_layer in range(self._num_layers)])

        self._init_weights()
        self.to(self.device)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, inputs, genotypes, hiddens, **kwargs): #pylint: disable=arguments-differ
        """
        Returns:
            logits: Tensor(bptt_steps, batch_size, num_tokens)
            raw_outs: Tensor(bptt_steps, batch_size, num_hid)
            droped_outs: Tensor(bptt_steps, batch_size, num_hid)
            next_hiddens: Tensor(num_layers, batch_size, num_hid)
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
                                                  x_masks[i_layer], h_masks[i_layer], genotypes,
                                                  **kwargs)
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
        next_hiddens = torch.stack([next_hid.detach_() for next_hid in next_hiddens])
        return logits, raw_outs, droped_outs, next_hiddens

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def save(self, path):
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    @classmethod
    def supported_data_types(cls):
        return ["sequence"]

    # ---- only rnn weights manager that handle sequence data need this ----
    def init_hidden(self, batch_size):
        """
        Initialize a hidden state. Only rnn that handled sequence data need this.
        """
        return torch.stack([torch.zeros(batch_size, self.num_hid, device=self.device)
                            for _ in range(self._num_layers)])

    def _init_weights(self):
        self.encoder.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)
        if self.decoder.bias is not None:
            self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)

class RNNSharedCell(nn.Module):
    def __init__(self, search_space, device, op_cls, num_emb, num_hid,
                 share_primitive_w, share_from_weights, batchnorm_step,
                 batchnorm_edge, batchnorm_out):
        super(RNNSharedCell, self).__init__()

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
            # the first bn
            self.bn_prev = nn.BatchNorm1d(num_emb + num_hid, affine=True)
        if self.batchnorm_step:
            # batchnorm after every step (just as in darts's implementation)
            self.bn_steps = nn.ModuleList([nn.BatchNorm1d(num_hid, affine=False)
                                           for _ in range(self._steps+1)])
        if self.batchnorm_out:
            # the out bn
            self.bn_out = nn.BatchNorm1d(num_hid, affine=True)

        if self.share_from_w:
            self.step_weights = nn.ModuleList([
                nn.Linear(num_hid, 2*num_hid, bias=False)
                for _ in range(self._steps)])
            [mod.weight.data.uniform_(-INIT_RANGE, INIT_RANGE) for mod in self.step_weights]

        # initiatiate op on edges
        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for from_ in range(self._steps):
            for to_ in range(from_+1, self._steps+1):
                self.edges[from_][to_] = op_cls(self.num_hid,
                                                search_space.shared_primitives,
                                                share_w=share_primitive_w,
                                                shared_module=self.step_weights[to_-1] \
                                                if self.share_from_w else None,
                                                batch_norm=batchnorm_edge)
                self.edge_mod.add_module("f_{}_t_{}".format(from_, to_), self.edges[from_][to_])

        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

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

class RNNSharedOp(nn.Module):
    def __init__(self, num_hid, primitives, share_w, batch_norm, shared_module, **kwargs):
        #pylint: disable=invalid-name
        super(RNNSharedOp, self).__init__()
        self.num_hid = num_hid
        self.primitives = primitives
        self.p_ops = nn.ModuleList()
        self.share_w = share_w
        self.batch_norm = batch_norm
        if shared_module is None:
            if share_w: # share weights between different activation function
                self.W = nn.Linear(num_hid, 2 * num_hid, bias=False)
                self.W.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)
            else:
                self.Ws = nn.ModuleList([nn.Linear(num_hid, 2 * num_hid, bias=False)
                                         for _ in range(len(self.primitives))])
                [mod.weight.data.uniform_(-INIT_RANGE, INIT_RANGE) for mod in self.Ws]
        else:
            self.W = shared_module
            self.share_w = True

        if batch_norm:
            self.bn = nn.BatchNorm1d(2 * num_hid, affine=True)

        for primitive in self.primitives:
            op = ops.get_op(primitive)(**kwargs)
            self.p_ops.append(op)
