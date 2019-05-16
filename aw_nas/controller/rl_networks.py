# -*- coding: utf-8 -*-
"""
Implementations of various controller rnn networks.
"""
import abc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import Component
from aw_nas import utils

class BaseRLControllerNet(Component, nn.Module):
    REGISTRY = "controller_network"

    def __init__(self, search_space, device, schedule_cfg=None):
        super(BaseRLControllerNet, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

        self.search_space = search_space
        self.device = device

    def forward(self, *args, **kwargs): #pylint: disable=arguments-differ
        return self.sample(*args, **kwargs)

    @abc.abstractmethod
    def sample(self, batch_size, prev_hidden):
        """Setup and initialize tasks of your ControlNet"""

    @abc.abstractmethod
    def save(self, path):
        """Save the network state to disk."""

    @abc.abstractmethod
    def load(self, path):
        """Load the network state from disk."""


class AnchorControlNet(BaseRLControllerNet):
    """
    Ref:
        https://github.com/melodyguan/enas/
        https://github.com/carpedm20/ENAS-pytorch/
    """

    NAME = "anchor_lstm"
    SCHEDULABLE_ATTRS = [
        "softmax_temperature"
    ]

    def __init__(self, search_space, device, num_lstm_layers=1,
                 controller_hid=100, attention_hid=100,
                 softmax_temperature=None, tanh_constant=None,
                 schedule_cfg=None):
        """
        Args:
            num_lstm_layers (int): Number of lstm layers.
            controller_hid (int): Dimension of lstm hidden state.
            attention_hid (int): Dimension of the attention layer.
            softmax_temperature (float): Softmax temperature before each
                decision softmax.
            tanh_constant (float):
        """
        super(AnchorControlNet, self).__init__(search_space, device, schedule_cfg)

        self.num_lstm_layers = num_lstm_layers
        self.controller_hid = controller_hid
        self.attention_hid = attention_hid
        self.softmax_temperature = softmax_temperature
        self.tanh_constant = tanh_constant

        self._num_primitives = len(self.search_space.shared_primitives)
        self.func_names = self.search_space.shared_primitives

        # will use anchors instead of embedding, see below
        # self.encoder = torch.nn.Embedding(self._num_primitives +
        #                                   self.search_space.num_init_nodes +
        #                                   self.search_space.num_steps - 1,
        #                                   self.controller_hid)

        self.lstm = nn.ModuleList()
        for _ in range(self.num_lstm_layers):
            self.lstm.append(torch.nn.LSTMCell(self.controller_hid, self.controller_hid))

        self.tanh = nn.Tanh()

        ## Embeddings
        # used as inputs to the decision block of each nodes
        self.g_emb = nn.Embedding(1, self.controller_hid)
        self.op_emb = nn.Embedding(self._num_primitives, self.controller_hid)

        ## Attention mapping
        # used for anchor attention mapping
        self.anchor_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        # used for (current node) hidden state attention mapping
        self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)

        ## Mapping before softmax
        self.w_node_soft = nn.Linear(self.attention_hid, 1, bias=False)
        self.w_op_soft = nn.Linear(self.controller_hid, self._num_primitives, bias=False)

        self.to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def stack_lstm(self, x, hidden):
        prev_h, prev_c = hidden
        next_h, next_c = [], []
        for layer_id, (_h, _c) in enumerate(zip(prev_h, prev_c)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_h, curr_c = self.lstm[layer_id](inputs, (_h, _c))
            next_h.append(curr_h)
            next_c.append(curr_c)
        return next_h, next_c

    def forward_op(self, inputs, hidden):
        hx, cx = self.stack_lstm(inputs, hidden)
        logits = self.w_op_soft(hx[-1])

        logits = self._handle_logits(logits)

        return logits, (hx, cx)

    def forward_node(self, inputs, hidden,
                     node_idx, anchors_w_1):
        hx, cx = self.stack_lstm(inputs, hidden)

        # attention mechanism: match with anchors of previous nodes
        query = torch.stack(anchors_w_1[:node_idx+self.search_space.num_init_nodes])
        query = self.tanh(query + self.hid_attn(hx[-1]))
        logits = self.w_node_soft(query).squeeze(-1).transpose(0, 1)

        # logits: (batch_size, num_choices)
        logits = self._handle_logits(logits)

        return logits, (hx, cx)

    def sample(self, batch_size=1, prev_hidden=None):
        """
        Args:
            batch_size (int): Number of samples to generate.
            prev_hidden (Tuple(List(torch.Tensor), List(torch.Tensor))): The previous hidden
                states of the stacked lstm cells.
        Returns:
        """
        entropies = []
        log_probs = []
        prev_nodes = []
        prev_ops = []
        anchors = torch.zeros((0, batch_size, self.controller_hid)).to(self.device)
        anchors_w_1 = []
        prev_layers = []

        # initialise anchors for the init nodes
        inputs = self.g_emb(torch.LongTensor([0] * batch_size).to(self.device))
        if prev_hidden is None:
            hidden = self._init_hidden(batch_size)
        else:
            hidden = prev_hidden
        for idx in range(self.search_space.num_init_nodes):
            hx, cx = self.stack_lstm(inputs, hidden)
            hidden = (hx, cx)
            anchors = torch.cat([anchors, torch.zeros_like(hx[-1])[None, :]], dim=0)
            anchors_w_1.append(self.anchor_attn(hx[-1]))

        # begin sample
        for idx in range(self.search_space.num_steps):
            # for every step, sample `self.search_space.num_node_inputs` input nodes and ops
            # from nodes
            for _ in range(self.search_space.num_node_inputs):
                logits, hidden = self.forward_node(inputs, hidden, idx, anchors_w_1)

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(log_prob * probs).sum(dim=-1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(1, action)[:, 0]
                action = action[:, 0]

                entropies.append(entropy)
                log_probs.append(selected_log_prob)
                prev_nodes.append(action.cpu().numpy())

                # calcualte next inputs
                # anchors: (step, batch_size, controller_hid);
                # action: (batch_size);
                prev_layers.append(anchors[action, range(batch_size)])
                inputs = prev_layers[-1]

            # operation on edges
            for _ in range(self.search_space.num_node_inputs):
                logits, hidden = self.forward_op(inputs, hidden)

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(log_prob * probs).sum(dim=-1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(1, action)[:, 0]
                action = action[:, 0]

                entropies.append(entropy)
                log_probs.append(selected_log_prob)
                prev_ops.append(action.cpu().numpy())

                # calcualte next inputs
                inputs = self.op_emb(action)

            # calculate anchor for this node
            next_h, next_c = self.stack_lstm(inputs, hidden)
            anchors = torch.cat([anchors, next_h[-1][None, :]], dim=0)
            anchors = torch.cat([anchors, ], dim=0)
            anchors_w_1.append(self.anchor_attn(next_h[-1]))
            hidden = (next_h, next_c)
            inputs = self.g_emb(torch.LongTensor([0] * batch_size)\
                                .to(self.device))

        # stack prev_nodes / prev_ops of all steps, and tranpose to (batch_size, steps)
        prev_nodes = list(np.stack(prev_nodes).transpose())
        prev_ops = list(np.stack(prev_ops).transpose())
        arch = list(zip(prev_nodes, prev_ops))
        return arch, torch.stack(log_probs, dim=-1), \
            torch.stack(entropies, dim=-1), hidden

    def save(self, path):
        """Save the network state to disk."""
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path):
        """Load the network state from disk."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])

    def _handle_logits(self, logits):
        if self.softmax_temperature is not None:
            logits /= self.softmax_temperature

        # exploration
        # if self.mode == "train": # foxfi: ?
        if self.tanh_constant is not None:
            logits = (self.tanh_constant * F.tanh(logits))

        return logits

    def _init_hidden(self, batch_size):
        hxs = [utils.get_variable(torch.zeros(batch_size,
                                              self.controller_hid),
                                  self.device, requires_grad=False)
               for _ in range(self.num_lstm_layers)]
        cxs = [v.clone() for v in hxs]
        return (hxs, cxs)
