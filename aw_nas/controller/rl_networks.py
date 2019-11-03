# -*- coding: utf-8 -*-
"""
Implementations of various controller rnn networks.
"""
import abc
import collections

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import Component
from aw_nas import utils
from aw_nas.utils.exception import expect

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
    def sample(self, batch_size, prev_hidden, cell_index):
        """Setup and initialize tasks of your ControlNet"""

    @abc.abstractmethod
    def save(self, path):
        """Save the network state to disk."""

    @abc.abstractmethod
    def load(self, path):
        """Load the network state from disk."""

class BaseLSTM(BaseRLControllerNet):
    def __init__(self, search_space, device, cell_index,
                 num_lstm_layers=1, controller_hid=64,
                 softmax_temperature=None, tanh_constant=1.1,
                 op_tanh_reduce=2.5, force_uniform=False,
                 schedule_cfg=None):
        super(BaseLSTM, self).__init__(search_space, device, schedule_cfg)

        self.cell_index = cell_index
        self.num_lstm_layers = num_lstm_layers
        self.controller_hid = controller_hid
        self.softmax_temperature = softmax_temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.force_uniform = force_uniform

        if self.cell_index is None:
            # parameters/inference for all cell group in one controller network
            if not self.search_space.cellwise_primitives:
                # the same set of primitives for different cg group
                self._primitives = self.search_space.shared_primitives
            else:
                # different set of primitives for different cg group
                _primitives = collections.OrderedDict()
                for csp in self.search_space.cell_shared_primitives:
                    for name in csp:
                        if name not in _primitives:
                            _primitives[name] = len(_primitives)
                self._primitives = list(_primitives.keys())
                self._cell_primitive_indexes = [[_primitives[name] for name in csp] \
                                                for csp in self.search_space.cell_shared_primitives]
            self._num_steps = self.search_space.num_steps
            expect(isinstance(self._num_steps, int),
                   "Shared RL network do not support using different steps in "
                   "different cell groups")
        else:
            self._primitives = self.search_space.cell_shared_primitives[self.cell_index]
            self._num_steps = self.search_space.get_num_steps(self.cell_index)

        self._num_primitives = len(self._primitives)

        self.lstm = nn.ModuleList()
        for _ in range(self.num_lstm_layers):
            self.lstm.append(nn.LSTMCell(self.controller_hid, self.controller_hid))

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def forward_op(self, inputs, hidden, cell_index):
        hx, cx = self.stack_lstm(inputs, hidden)
        if self.cell_index is not None or not self.search_space.cellwise_primitives or \
           tuple(self._cell_primitive_indexes[cell_index]) == tuple(range(self._num_primitives)):
            # cell_index is not None: this network is not shared
            logits = self.w_op_soft(hx[-1])
        else:
            # choose subset of op/primitive weight
            w_op_soft_weight = self.w_op_soft.weight[self._cell_primitive_indexes[cell_index], :]
            logits = torch.matmul(hx[-1], torch.transpose(w_op_soft_weight, 0, 1))

        logits = self._handle_logits(logits, mode="op")

        return logits, (hx, cx)

    def stack_lstm(self, x, hidden):
        prev_h, prev_c = hidden
        next_h, next_c = [], []
        for layer_id, (_h, _c) in enumerate(zip(prev_h, prev_c)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_h, curr_c = self.lstm[layer_id](inputs, (_h, _c))
            next_h.append(curr_h)
            next_c.append(curr_c)
        return next_h, next_c

    def save(self, path):
        """Save the network state to disk."""
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        """Load the network state from disk."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    def _check_cell_index(self, cell_index):
        if self.cell_index is not None:
            # this controller network only handle one cell group
            expect(cell_index is None or cell_index == self.cell_index,
                   ("This controller network handle only cell group {},"
                    " cell group {} not handlable").format(self.cell_index, cell_index))
            cell_index = self.cell_index
        else:
            # this controller network handle all cell groups
            if self.search_space.num_cell_groups == 1:
                cell_index = 0
            expect(not self.search_space.cellwise_primitives or cell_index is not None,
                   "When controller network handle all cell groups, and cell-wise shared"
                   " primitives are different. "
                   "must specifiy `cell_index` when calling `c_net.sample`")
        return cell_index

    def _handle_logits(self, logits, mode):
        if self.force_uniform:
            logits = F.softmax(torch.zeros_like(logits), dim=-1)
        else:
            if self.softmax_temperature is not None:
                logits /= self.softmax_temperature

            # exploration
            # if self.mode == "train": # foxfi: ?
            if self.tanh_constant is not None:
                tanh_constant = self.tanh_constant
                if mode == "op":
                    tanh_constant /= self.op_tanh_reduce
                logits = tanh_constant * torch.tanh(logits)

        return logits

    def _get_default_hidden(self, batch_size):
        hxs = [utils.get_variable(torch.zeros(batch_size,
                                              self.controller_hid,
                                              device=self.device),
                                  self.device, requires_grad=False)
               for _ in range(self.num_lstm_layers)]
        cxs = [v.clone() for v in hxs]
        return (hxs, cxs)


class AnchorControlNet(BaseLSTM):
    """
    if `cell_index` is specified, controller network will use
    `search_space.cell_shared_primitives[cell_index]` to construct
    op-related embeddings; Otherwise, when `cell_index is None`
    will construct using all share op/primtivie embedding
    primitives between different cell groups

    Ref:
        https://github.com/melodyguan/enas/
        https://github.com/carpedm20/ENAS-pytorch/
    """

    NAME = "anchor_lstm"
    SCHEDULABLE_ATTRS = [
        "softmax_temperature",
        "force_uniform"
    ]

    def __init__(self, search_space, device, cell_index,
                 num_lstm_layers=1,
                 controller_hid=64, attention_hid=64,
                 softmax_temperature=None, tanh_constant=1.1,
                 op_tanh_reduce=2.5, force_uniform=False,
                 schedule_cfg=None):
        """
        Args:
            num_lstm_layers (int): Number of lstm layers.
            controller_hid (int): Dimension of lstm hidden state.
            attention_hid (int): Dimension of the attention layer.
            softmax_temperature (float): Softmax temperature before each
                decision softmax.
            tanh_constant (float):
            op_tanh_reduce (float):
            force_uniform (bool):
        """
        super(AnchorControlNet, self).__init__(search_space, device, cell_index,
                                               num_lstm_layers=num_lstm_layers,
                                               controller_hid=controller_hid,
                                               softmax_temperature=softmax_temperature,
                                               tanh_constant=tanh_constant,
                                               op_tanh_reduce=op_tanh_reduce,
                                               force_uniform=force_uniform,
                                               schedule_cfg=schedule_cfg)

        self.attention_hid = attention_hid
        self.tanh = nn.Tanh()

        ## Embeddings
        # used as inputs to the decision block of each nodes
        self.static_hidden = utils.keydefaultdict(self._get_default_hidden)
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

    def forward_node(self, inputs, hidden,
                     node_idx, anchors_w_1):
        hx, cx = self.stack_lstm(inputs, hidden)

        # attention mechanism: match with anchors of previous nodes
        # query: (num_node_choice, batch_size, attention_hid)
        query = torch.stack(anchors_w_1[:node_idx+self.search_space.num_init_nodes])

        query = self.tanh(query + self.hid_attn(hx[-1]))
        logits = self.w_node_soft(query).squeeze(-1).transpose(0, 1)

        # logits: (batch_size, num_choices)
        logits = self._handle_logits(logits, mode="node")

        return logits, (hx, cx)

    def sample(self, batch_size=1, prev_hidden=None, cell_index=None):
        """
        Args:
            batch_size (int): Number of samples to generate.
            prev_hidden (Tuple(List(torch.Tensor), List(torch.Tensor))): The previous hidden
                states of the stacked lstm cells.
        Returns:
        """
        cell_index = self._check_cell_index(cell_index)

        entropies = []
        log_probs = []
        prev_nodes = []
        prev_ops = []
        anchors = torch.zeros(0, batch_size, self.controller_hid, device=self.device)
        anchors_w_1 = []
        prev_layers = []

        # initialise anchors for the init nodes
        inputs = self.g_emb(torch.zeros(batch_size, dtype=torch.long, device=self.device))
        if prev_hidden is None:
            hidden = self.static_hidden[batch_size]
        else:
            hidden = prev_hidden
        for idx in range(self.search_space.num_init_nodes):
            hx, cx = self.stack_lstm(inputs, hidden)
            hidden = (hx, cx)
            anchors = torch.cat([anchors, torch.zeros_like(hx[-1])[None, :]], dim=0)
            anchors_w_1.append(self.anchor_attn(hx[-1]))

        # begin sample
        for idx in range(self._num_steps):
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
                inputs = prev_layers[-1] # (batch_size, controler_hid)

            # operation on edges
            for _ in range(self.search_space.num_node_inputs):
                logits, hidden = self.forward_op(inputs, hidden, cell_index)

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
            inputs = self.g_emb(torch.zeros(batch_size, dtype=torch.long, device=self.device))

        # stack prev_nodes / prev_ops of all steps, and tranpose to (batch_size, steps)
        prev_nodes = list(np.stack(prev_nodes).transpose())
        prev_ops = list(np.stack(prev_ops).transpose())
        arch = list(zip(prev_nodes, prev_ops))
        return arch, torch.stack(log_probs, dim=-1), \
            torch.stack(entropies, dim=-1), hidden


class EmbedControlNet(BaseLSTM):

    NAME = "embed_lstm"
    SCHEDULABLE_ATTRS = [
        "softmax_temperature",
        "force_uniform"
    ]

    def __init__(self, search_space, device, cell_index,
                 num_lstm_layers=1,
                 controller_hid=64, attention_hid=64,
                 softmax_temperature=None, tanh_constant=1.1,
                 op_tanh_reduce=2.5, force_uniform=False,
                 schedule_cfg=None):
        """
        Args:
            num_lstm_layers (int): Number of lstm layers.
            controller_hid (int): Dimension of lstm hidden state.
            softmax_temperature (float): Softmax temperature before each
                decision softmax.
            tanh_constant (float):
            op_tanh_reduce (float):
            force_uniform (bool):
        """
        super(EmbedControlNet, self).__init__(search_space, device, cell_index,
                                              num_lstm_layers=num_lstm_layers,
                                              controller_hid=controller_hid,
                                              softmax_temperature=softmax_temperature,
                                              tanh_constant=tanh_constant,
                                              op_tanh_reduce=op_tanh_reduce,
                                              force_uniform=force_uniform,
                                              schedule_cfg=schedule_cfg)

        self.attention_hid = attention_hid
        self.tanh = nn.Tanh()

        ## Embeddings
        self.static_inputs = utils.keydefaultdict(self._get_default_inputs)
        self.static_hidden = utils.keydefaultdict(self._get_default_hidden)
        self.op_emb = nn.Embedding(self._num_primitives, self.controller_hid)
        _n_node_input = self.search_space.num_init_nodes +\
                        self._num_steps - 1
        self.node_emb = nn.Embedding(_n_node_input, self.controller_hid)

        ## Attention mapping
        # used for embedding attention mapping
        self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        # used for (current node) hidden state attention mapping
        self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        query_index = torch.LongTensor(range(0, _n_node_input))
        self.register_buffer("query_index", query_index)

        ## Mapping before softmax
        self.w_node_soft = nn.Linear(self.attention_hid, 1, bias=False)
        self.w_op_soft = nn.Linear(self.controller_hid, self._num_primitives, bias=False)

        self.to(self.device)

        self.reset_parameters()

    def __getstate__(self):
        state = super(EmbedControlNet, self).__getstate__()
        del state["static_inputs"]
        del state["static_hidden"]
        return state

    def __setstate__(self, state):
        super(EmbedControlNet, self).__setstate__(state)
        # reset these two cache dict
        self.static_inputs = utils.keydefaultdict(self._get_default_inputs)
        self.static_hidden = utils.keydefaultdict(self._get_default_hidden)

    def forward_node(self, inputs, hidden, node_idx):
        hx, cx = self.stack_lstm(inputs, hidden)

        # attention mechanism: match with embedding of previous nodes
        # query: (num_node_choices, 1, controller_hid)
        query = self.node_emb(self.query_index[:node_idx+self.search_space.num_init_nodes])\
                    .unsqueeze(1)
        # hx[-1]: (batch_size, controller_hid)
        query = self.tanh(self.emb_attn(query) + self.hid_attn(hx[-1]))
        logits = self.w_node_soft(query).squeeze(-1).transpose(0, 1)

        # logits: (batch_size, num_choices)
        logits = self._handle_logits(logits, mode="node")

        return logits, (hx, cx)

    def sample(self, batch_size=1, prev_hidden=None, cell_index=None):
        """
        Args:
            batch_size (int): Number of samples to generate.
            prev_hidden (Tuple(List(torch.Tensor), List(torch.Tensor))): The previous hidden
                states of the stacked lstm cells.
        Returns:
        """
        cell_index = self._check_cell_index(cell_index)

        entropies = []
        log_probs = []
        prev_nodes = []
        prev_ops = []

        inputs = self.static_inputs[batch_size]  # zeros (batch_size, controller_hid)
        if prev_hidden is None:
            hidden = self.static_hidden[batch_size]
        else:
            hidden = prev_hidden

        # begin sample
        for idx in range(self._num_steps):
            # for every step, sample `self.search_space.num_node_inputs` input nodes and ops
            # from nodes
            for _ in range(self.search_space.num_node_inputs):
                logits, hidden = self.forward_node(inputs, hidden, idx)

                probs = F.softmax(logits, dim=-1)
                log_prob = torch.log(probs)
                entropy = -(log_prob * probs).sum(dim=-1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(1, action)[:, 0]
                action = action[:, 0]

                entropies.append(entropy)
                log_probs.append(selected_log_prob)
                prev_nodes.append(action.cpu().numpy())

                # calcualte next inputs
                inputs = self.node_emb(action)

            # operation on edges
            for _ in range(self.search_space.num_node_inputs):
                logits, hidden = self.forward_op(inputs, hidden, cell_index)

                probs = F.softmax(logits, dim=-1)
                log_prob = torch.log(probs)
                entropy = -(log_prob * probs).sum(dim=-1, keepdim=False)

                action = probs.multinomial(num_samples=1).data
                selected_log_prob = log_prob.gather(1, action)[:, 0]
                action = action[:, 0]

                entropies.append(entropy)
                log_probs.append(selected_log_prob)
                prev_ops.append(action.cpu().numpy())

                # calcualte next inputs
                inputs = self.op_emb(action)

        # stack prev_nodes / prev_ops of all steps, and tranpose to (batch_size, steps)
        prev_nodes = list(np.stack(prev_nodes).transpose())
        prev_ops = list(np.stack(prev_ops).transpose())
        arch = list(zip(prev_nodes, prev_ops))
        return arch, torch.stack(log_probs, dim=-1), \
            torch.stack(entropies, dim=-1), hidden

    def _get_default_inputs(self, batch_size):
        return utils.get_variable(
            torch.zeros(batch_size, self.controller_hid, device=self.device),
            self.device, requires_grad=False)
