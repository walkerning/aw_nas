# -*- coding: utf-8 -*-
"""
Shared weights super net.
"""

from __future__ import print_function

import contextlib
import six

import torch

from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp

__all__ = ["SubCandidateNet", "SuperNet"]


class SubCandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """

    def __init__(self, super_net, rollout, member_mask, cache_named_members=False,
                 virtual_parameter_only=False):
        super(SubCandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.search_space = super_net.search_space
        self.member_mask = member_mask
        self.cache_named_members = cache_named_members
        self.virtual_parameter_only = virtual_parameter_only
        self._cached_np = None
        self._cached_nb = None

        self.genotypes = [g[1] for g in rollout.genotype_list()]

    @contextlib.contextmanager
    def begin_virtual(self):
        """
        On entering, store the current states (parameters/buffers) of the network
        On exiting, restore the stored states.
        Needed for surrogate steps of each candidate network,
        as different SubCandidateNet share the same set of SuperNet weights.
        """

        w_clone = {k: v.clone() for k, v in self.named_parameters()}
        if not self.virtual_parameter_only:
            buffer_clone = {k: v.clone() for k, v in self.named_buffers()}

        yield

        for n, v in self.named_parameters():
            v.data.copy_(w_clone[n])
        del w_clone

        if not self.virtual_parameter_only:
            for n, v in self.named_buffers():
                v.data.copy_(buffer_clone[n])
            del buffer_clone

    def get_device(self):
        return self._device

    def forward(self, inputs): #pylint: disable=arguments-differ
        return self.super_net.forward(inputs, self.genotypes)

    def plot_arch(self):
        return self.super_net.search_space.plot_arch(self.genotypes)

    def named_parameters(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                # use cached members
                if self._cached_np is None:
                    self._cached_np = []
                    for n, v in self.active_named_parameters(prefix=""):
                        self._cached_np.append((n, v))
                prefix = prefix + ("/" if prefix else "")
                for n, v in self._cached_np:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_parameters(prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_parameters(prefix=prefix):
                yield n, v

    def named_buffers(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                if self._cached_nb is None:
                    self._cached_nb = []
                    for n, v in self.active_named_buffers(prefix=""):
                        self._cached_nb.append((n, v))
                prefix = prefix + ("/" if prefix else "")
                for n, v in self._cached_nb:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_buffers(prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_buffers(prefix=prefix):
                yield n, v

    def active_named_parameters(self, prefix="", recurse=True):
        """
        Get the generator of name-parameter pairs active
        in this candidate network. Always recursive.
        """
        for n, v in self.super_net.sub_named_members(self.genotypes,
                                                     prefix=prefix,
                                                     member="parameters"):
            yield n, v

    def active_named_buffers(self, prefix="", recurse=True):
        """
        Get the generator of name-buffer pairs active
        in this candidate network.
        """
        for n, v in self.super_net.sub_named_members(self.genotypes,
                                                     prefix=prefix,
                                                     member="buffers"):
            yield n, v


class SuperNet(SharedNet):
    """
    A cell-based super network
    """
    NAME = "supernet"

    def __init__(self, search_space, device,
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 candidate_member_mask=True, candidate_cache_named_members=False,
                 candidate_virtual_parameter_only=False):
        """
        Args:
            candidate_member_mask (bool): If true, the candidate network's `named_parameters`
                or `named_buffers` method will only return parameters/buffers that is active,
                `begin_virtual` just need to store/restore these active variables.
                This should be more efficient.
            candidate_cache_named_members (bool): If true, the candidate network's
                named parameters/buffers will be cached on the first calculation.
                It should not cause any logical, however, due to my benchmark, this bring no
                performance increase. So default disable it.
            candidate_virtual_parameter_only (bool): If true, the candidate network's
                `begin_virtual` will only store/restore parameters, not buffers (e.g. running
                mean/running std in BN layer).
        """
        super(SuperNet, self).__init__(search_space, device,
                                       cell_cls=DiscreteSharedCell, op_cls=DiscreteSharedOp,
                                       num_classes=10, init_channels=16, stem_multiplier=3,
                                       max_grad_norm=5.0, dropout_rate=0.1)

        # candidate net with/without parameter mask
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only

    def forward(self, inputs, genotypes): #pylint: disable=arguments-differ
        states = [self.stem(inputs) for _ in range(self._num_init)]
        for cg_idx, cell in zip(self._cell_layout, self.cells):
            genotype = genotypes[cg_idx]
            states.append(cell(states, genotype))
            states = states[1:]

        out = self.global_pooling(states[-1])
        out = self.dropout(out)
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

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return SubCandidateNet(self, rollout,
                               member_mask=self.candidate_member_mask,
                               cache_named_members=self.candidate_cache_named_members,
                               virtual_parameter_only=self.candidate_virtual_parameter_only)

    def rollout_type(self):
        return "discerete"

class DiscreteSharedCell(SharedCell):
    def num_out_channel(self):
        return self.num_channels * self.search_space.num_steps

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

class DiscreteSharedOp(SharedOp):
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
