# -*- coding: utf-8 -*-

from collections import defaultdict

import torch
from torch import nn

from aw_nas import ops
from aw_nas.weights_manager.base import BaseWeightsManager

class RNNSharedNet(BaseWeightsManager, nn.Module):
    def __init__(
            self, search_space, device, cell_cls, op_cls,
            num_tokens, num_emb, num_hid,
            tie_weight, share_primitive_weights, edge_batch_norm,
            # training
            max_grad_norm,
            # dropout probs
            dropout_emb, dropout_inp, dropout_hid, dropout_out):
        super(RNNSharedNet, self).__init__(search_space, device)
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
        self.dropout_inp = dropout_inp
        self.dropout_hid = dropout_hid
        self.dropout_out = dropout_out

        # modules
        self.encoder = ops.EmbeddingDropout(num_tokens, num_emb, dropout=dropout_emb)
        self.decoder = nn.Linear(num_hid, num_tokens)
        self.lockdrop = ops.LockedDropout()

        if tie_weight:
            assert num_hid == num_emb, \
                "if `tie_weight` is true, `num_hid` must equal `num_emb` ({} VS {})"\
                    .format(num_hid, num_emb)
            self.decoder.weight = self.encoder.weight

        self.cells = nn.ModuleList([cell_cls(search_space, device, op_cls,
                                             num_emb, num_hid, share_w=share_primitive_weights,
                                             edge_batch_norm=edge_batch_norm)
                                    for _ in range(self._num_layers)])
        self.to(self.device)

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # clip the gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def save(self, path):
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])

    def supported_data_types(self):
        return ["sequence"]

    def init_hidden(self, batch_size):
        """
        Initialize a hidden state. Only rnn that handled sequence data need this.
        """
        return [torch.zeros(batch_size, self.num_hid).to(self.device)
                for _ in range(self._num_layers)]


class RNNSharedCell(nn.Module):
    def __init__(self, search_space, device, op_cls, num_emb, num_hid,
                 share_w, edge_batch_norm):
        super(RNNSharedCell, self).__init__()

        self.num_emb = num_emb
        self.num_hid = num_hid
        self.edge_batch_norm = edge_batch_norm
        self._steps = search_space.num_steps

        # the first step, convert input x and previous hidden
        self.w_prev = nn.Linear(num_emb + num_hid, 2 * num_hid)

        if self.edge_batch_norm:
            self.bn_prev = nn.BatchNorm1d(num_emb + num_hid, affine=True)
            self.bn_out = nn.BatchNorm1d(num_hid, affine=True)

        # initiatiate op on edges
        self.edges = defaultdict(dict)
        for from_ in range(self._steps):
            for to_ in range(from_+1, self._steps+1):
                self.edges[from_][to_] = op_cls(num_hid, search_space.shared_primitives,
                                                share_w=share_w, batch_norm=edge_batch_norm)
                self.add_module("f_{}_t_{}".format(from_, to_), self.edges[from_][to_])


class RNNSharedOp(nn.Module):
    def __init__(self, num_hid, primitives, share_w, batch_norm, **kwargs):
        #pylint: disable=invalid-name
        super(RNNSharedOp, self).__init__()
        self.num_hid = num_hid
        self.primitives = primitives
        self.p_ops = nn.ModuleList()
        self.share_w = share_w
        self.batch_norm = batch_norm

        if share_w: # share weights between different activation function
            self.W = nn.Linear(num_hid, 2 * num_hid, bias=False)
        else:
            self.Ws = nn.ModuleList([nn.Linear(num_hid, 2 * num_hid, bias=False)
                                     for _ in range(len(self.primitives))])

        if batch_norm:
            self.bn = nn.BatchNorm1d(2 * num_hid, affine=True)

        for primitive in self.primitives:
            op = ops.get_op(primitive)(**kwargs)
            self.p_ops.append(op)
