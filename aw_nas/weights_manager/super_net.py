# -*- coding: utf-8 -*-
"""
Shared weights super net.
"""

from __future__ import print_function

import contextlib
import six

import numpy as np
import torch
from torch import nn

from aw_nas.weights_manager.base import (
    BaseWeightsManager,
    CandidateNet
)
from aw_nas import ops

__all__ = ["SubCandidateNet", "SuperNet"]


class SubCandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """

    def __init__(self, super_net, rollout):
        super(SubCandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.search_space = super_net.search_space
        self.rollout = rollout
        self.genotypes = list(self.search_space.genotype(rollout.arch)._asdict().values())

    @contextlib.contextmanager
    def begin_virtual(self):
        """
        On entering, store the current states (parameters/buffers) of the network
        On exiting, restore the stored states.
        Needed for surrogate steps of each candidate network,
        as different SubCandidateNet share the same set of SuperNet weights.
        """

        w_clone = {k: v.clone() for k, v in self.named_parameters()}

        yield

        for n, v in self.named_parameters():
            v.data.copy_(w_clone[n])
        del w_clone

    def get_device(self):
        return self._device

    def forward(self, inputs): #pylint: disable=arguments-differ
        return self.super_net.forward(inputs, self.genotypes)

    def plot_arch(self):
        return self.super_net.search_space.plot_arch(self.genotypes)

    def named_parameters(self, prefix="", recurse=True):
        """
        Get the generator of name-parameter pairs active
        in this candidate network. Always recursive.
        """
        for n, v in self.super_net.sub_named_members(self.genotypes,
                                                     prefix=prefix,
                                                     member="parameters"):
            yield n, v

    def named_buffers(self, prefix="", recurse=True):
        """
        Get the generator of name-buffer pairs active
        in this candidate network.
        """
        for n, v in self.super_net.sub_named_members(self.genotypes,
                                                     prefix=prefix,
                                                     member="buffers"):
            yield n, v


class SuperNet(BaseWeightsManager, nn.Module):
    """
    A cell-based super network
    """
    NAME = "supernet"

    def __init__(self, search_space, device,
                 num_classes=10, init_channels=16, stem_multiplier=3):
        super(SuperNet, self).__init__(search_space, device)
        nn.Module.__init__(self)

        self.num_classes = num_classes
        # init channel number of the first cell layers,
        # x2 after every reduce cell
        self.init_channels = init_channels
        # channels of stem conv / init_channels
        self.stem_multiplier = stem_multiplier

        self._num_init = self.search_space.num_init_nodes
        self._cell_layout = self.search_space.cell_layout
        self._reduce_cgs = self.search_space.reduce_cell_groups
        self._num_layers = self.search_space.num_layers
        self._out_multiplier = self.search_space.num_steps

        ## initialize sub modules
        c_stem = self.stem_multiplier * self.init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_stem)
        )

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = [c_stem] * self._num_init
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]

        for i_layer, stride in enumerate(strides):
            if stride > 1:
                num_channels *= stride

            cell = SharedCell(self.search_space,
                              layer_index=i_layer,
                              num_channels=num_channels,
                              prev_num_channels=tuple(prev_num_channels),
                              stride=stride,
                              prev_strides=[1] * self._num_init + strides[:i_layer])

            prev_num_channels.append(num_channels * self._out_multiplier)
            prev_num_channels = prev_num_channels[1:]
            self.cells.append(cell)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_channels * self._out_multiplier,
                                    self.num_classes)

        self.to(self.device)

    def forward(self, inputs, genotypes): #pylint: disable=arguments-differ
        states = [self.stem(inputs) for _ in range(self._num_init)]
        for cg_idx, cell in zip(self._cell_layout, self.cells):
            genotype = genotypes[cg_idx]
            states.append(cell(states, genotype))
            states = states[1:]

        out = self.global_pooling(states[-1])
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

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

        # only a subset of modules under `self.cells` will be forwarded
        # from the last output, parse the dependency backward
        visited = set()
        cell_idxes = [len(self.cells)-1]
        depend_nodes_lst = [{edge[1] for edge in genotype}.intersection(range(self._num_init))\
                            for genotype in genotypes]
        while cell_idxes:
            cell_idx = cell_idxes.pop()
            visited.update([cell_idx])
            # cg_idx is the cell group of the cell i
            cg_idx = self._cell_layout[cell_idx]
            depend_nodes = depend_nodes_lst[cg_idx]
            depend_cell_idxes = [cell_idx - self._num_init + node_idx for node_idx in depend_nodes]
            depend_cell_idxes = [i for i in depend_cell_idxes if i >= 0 and i not in visited]
            cell_idxes += depend_cell_idxes

        for cell_idx in sorted(visited):
            cell = self.cells[cell_idx]
            genotype = genotypes[self._cell_layout[cell_idx]]
            for n, v in cell.sub_named_members(genotype,
                                               prefix=prefix + "cells.{}".format(cell_idx),
                                               member=member):
                yield n, v

    def _is_reduce(self, layer_idx):
        return self._cell_layout[layer_idx] in self._reduce_cgs

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return SubCandidateNet(self, rollout)

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        optimizer.step() # apply the gradients

    def save(self, path):
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])

class SharedCell(nn.Module):
    def __init__(self, search_space, layer_index, num_channels,
                 prev_num_channels, stride, prev_strides):
        super(SharedCell, self).__init__()
        self.search_space = search_space
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.layer_index = layer_index

        self._steps = self.search_space.num_steps
        self._num_init = self.search_space.num_init_nodes
        self._primitives = self.search_space.shared_primitives

        self.preprocess_ops = nn.ModuleList()
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[:len(prev_num_channels)])
        for prev_c, prev_s in zip(prev_num_channels, prev_strides):
            if prev_s > 1:
                # need skip connection, and is not the connection from the input image
                preprocess = ops.FactorizedReduce(C_in=prev_c,
                                                  C_out=num_channels,
                                                  stride=prev_s,
                                                  affine=False)
            else: # prev_c == _steps * num_channels or inputs
                preprocess = ops.ReLUConvBN(C_in=prev_c,
                                            C_out=num_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            affine=False)
            self.preprocess_ops.append(preprocess)
        assert len(self.preprocess_ops) == self._num_init

        self.edges = nn.ModuleList()
        for i in range(self._steps):
            for j in range(i + self._num_init):
                op = SharedOp(self.num_channels, stride=self.stride if j < self._num_init else 1,
                              primitives=self._primitives)
                self.edges.append(op)

    def forward(self, inputs, genotype): #pylint: disable=arguments-differ
        assert self._num_init == len(inputs)
        states = {
            i: op(inputs) for i, (op, inputs) in \
            enumerate(zip(self.preprocess_ops, inputs))
        }

        for op_type, from_, to_ in genotype:
            edge_id = int((to_ - self._num_init) * \
                          (self._num_init + to_ - 1) / 2 + from_)
            # print("from_: {} ({}) ; to_: {} ; op_type: {} ; stride: {} , edge_id: {}"\
            #       .format(from_, states[from_].shape[2], to_, op_type,
            #               self.edges[edge_id].stride, edge_id))
            out = self.edges[edge_id](states[from_], op_type)
            if to_ in states:
                states[to_] = states[to_] + out
            else:
                states[to_] = out

        return torch.cat([states[i] for i in \
                          range(self._num_init,
                                self._steps + self._num_init)],
                         dim=1)

    def sub_named_members(self, genotype,
                          prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        all_from = {edge[1] for edge in genotype}
        for i, pre_op in enumerate(self.preprocess_ops):
            if i in all_from:
                for n, v in getattr(pre_op, "named_" + member)\
                    (prefix=prefix+"preprocess_ops."+str(i)):
                    yield n, v

        for op_type, from_, to_ in genotype:
            edge_id = int((to_ - self._num_init) * \
                          (self._num_init + to_ - 1) / 2 + from_)
            edge_share_op = self.edges[edge_id]
            for n, v in edge_share_op.sub_named_members(op_type,
                                                        prefix=prefix +"edges." + str(edge_id),
                                                        member=member):
                yield n, v

class SharedOp(nn.Module):
    """
    An operation on an edge, consisting of multiple primitives.
    """

    def __init__(self, C, stride, primitives):
        super(SharedOp, self).__init__()
        self.primitives = primitives
        self.stride = stride
        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = ops.get_op(primitive)(C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C,
                                                      affine=False))
            self.p_ops.append(op)

    def forward(self, x, op_type): #pylint: disable=arguments-differ
        index = self.primitives.index(op_type)
        return self.p_ops[index](x)

    def sub_named_members(self, op_type,
                          prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        index = self.primitives.index(op_type)
        for n, v in getattr(self.p_ops[index], "named_" + member)(prefix="{}p_ops.{}"\
                                                .format(prefix, index)):
            yield n, v


#pylint: disable=invalid-name,ungrouped-imports
if __name__ == "__main__":
    from torch import optim

    from aw_nas.common import get_search_space, Rollout

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    net = SuperNet(search_space, device)

    NUM_TEST = 10
    LR = 0.1
    for i in range(NUM_TEST):
        print("TEST {}/{}".format(i+1, NUM_TEST))
        arch = search_space.random_sample()
        # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
        # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

        rollout = Rollout(arch, info={}, search_space=search_space)
        cand_net = net.assemble_candidate(rollout)

        # test named_parameters/named_buffers
        print("Supernet parameter num: {} ; buffer num: {}"\
              .format(len(list(net.parameters())), len(list(net.buffers()))))
        print("candidatenet parameter num: {} ; buffer num: {}"\
              .format(len(list(cand_net.named_parameters())), len(list(cand_net.buffers()))))
        c_params = dict(cand_net.named_parameters())
        s_names = set(dict(net.named_parameters()).keys())
        c_names = set(c_params.keys())
        assert len(s_names.intersection(c_names)) == len(c_names)

        # test forward
        data = (torch.tensor(np.random.rand(1, 3, 28, 28)).float(), torch.tensor([0]).long()) #pylint: disable=not-callable

        logits = cand_net.forward_data(data[0])
        assert logits.shape[-1] == 10

        # names = sorted(set(c_names).difference([g[0] for g in grads]))

        # test `gradient`, `begin_virtual`
        w_prev = {k: v.clone() for k, v in six.iteritems(c_params)}
        with cand_net.begin_virtual():
            grads = cand_net.gradient(data)
            assert len(grads) == len(c_names)
            optimizer = optim.SGD(cand_net.parameters(), lr=LR)
            optimizer.step()
            EPS = 1e-5
            for n, grad in grads:
                assert (w_prev[n] - grad * LR - c_params[n]).abs().sum().item() < EPS
            grads_2 = dict(cand_net.gradient(data))
            assert len(grads) == len(c_names)
            optimizer.step()
            for n, grad in grads:
                assert (w_prev[n] - (grad + grads_2[n]) * LR - c_params[n]).abs().sum().item() < EPS

        for n in c_params:
            assert (w_prev[n] - c_params[n]).abs().sum().item() < EPS
#pylint: enable=invalid-name,ungrouped-imports
