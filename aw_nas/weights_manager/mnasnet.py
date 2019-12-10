# -*- coding: utf-8 -*-
"""
Shared weights super net.
"""

from __future__ import print_function

import itertools
from collections import OrderedDict
import contextlib
import six

import torch
from torch import nn

from aw_nas.common import assert_rollout_type, group_and_sort_by_to_node
from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp
from aw_nas.utils import data_parallel, use_params
from aw_nas.ops import *

__all__ = ["MnasnetCandidateNet", "MnasNetSupernet"]

class MobileCell(MobileNetBlock):
    def __init__(self, expansion, C, C_out, stride, affine, kernel_size=3):
        super(MobileCell, self).__init__(expansion, C, C_out, stride, affine, kernel_size)
        self.expansion = expansion

    def forward_rollout(self, inputs, rollout, ind_i, ind_j):
        self.conv1.weight = self.conv1.state_dict()['weight']
        self.conv1.bias = self.conv1.state_dict()['bias']
        self.conv2.weight = self.conv2.state_dict()['weight']
        self.conv2.bias = self.conv2.state_dict()['bias']
        self.conv3.weight = self.conv3.state_dict()['weight']
        self.bn1.weight = self.bn1.state_dict()['weight']
        self.bn1.bias = self.bn1.state_dict()['bias']
        self.bn2.weight = self.bn2.state_dict()['weight']
        self.bn2.bias = self.bn2.state_dict()['bias']
        if self.conv1.weight.shape[1] * rollout.width[ind_i][ind_j]\
            < self.conv2.weight.shape[0]:
            weights = self.conv2.weight
            norms = {}
            for i in range(weights.shape[0]):
                norm = 0
                for c in range(weights.shape[1]):
                    for h in range(weights.shape[2]):
                        for w in range(weights.shape[3]):
                            norm += abs(weights[i, c, h, w])
                norms[i] = norm
            norms = sorted(norms.items(), key=lambda x:x[1])
            cut_channels = [norms[i][0] for i in\
                range(weights.shape[0] - self.conv1.weight.shape[1] *\
                rollout.width[ind_i][ind_j])]
            weights[cut_channels,:,:,:] = 0
            bias = self.conv2.bias
            bias[cut_channels] = 0
            object.__setattr__(self.conv2, "weight", weights)
            object.__setattr__(self.conv2, "bias", bias)
            weights = self.bn2.weight
            weights[cut_channels] = 0
            bias = self.bn2.bias
            bias[cut_channels] = 0
            object.__setattr__(self.bn2, "weight", weights)
            object.__setattr__(self.bn2, "bias", bias)
            weights = self.conv1.weight
            weights[cut_channels,:,:,:] = 0
            bias = self.conv1.bias
            bias[cut_channels] = 0
            object.__setattr__(self.conv1, "weight", weights)
            object.__setattr__(self.conv1, "bias", bias)
            weights = self.bn1.weight
            weights[cut_channels] = 0
            bias = self.bn1.bias
            bias[cut_channels] = 0
            object.__setattr__(self.bn1, "weight", weights)
            object.__setattr__(self.bn1, "bias", bias)
            weights = self.conv3.weight
            weights[:,cut_channels,:,:] = 0
            object.__setattr__(self.conv3, "weight", weights)
        out = super(MobileCell, self).forward(inputs)
        return out

class MNasNetSupernet(nn.Module):
    """
    A Mnasnet super network
    """
    NAME = "mnasnet_supernet"

    def __init__(self, search_space, device, rollout_type='mnasnet',
                 blocks=[3, 3, 3, 2, 4, 1], stride=[2, 2, 2, 1, 2, 1],
                 expansion=[3, 3, 6, 6, 6, 6], channels=[16, 24, 40, 80, 96, 192, 320],
                 num_classes=1000, gpus=tuple(),
                 candidate_member_mask=True, candidate_cache_named_members=False,
                 candidate_virtual_parameter_only=False, candidate_eval_no_grad=True):
        super(MNasNetSupernet, self).__init__()
        self.search_space = search_space
        self.device = device
        self.rollout_type = rollout_type
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only
        self.candidate_eval_no_grad = candidate_eval_no_grad
        self.gpus = gpus
        self.set_hook()
        self._flops_calculated = False
        self.total_flops = 0
        self.blocks = blocks
        self.stride = stride
        self.expansion = expansion
        self.channels = channels
        self.cells = []
        self.sep_stem = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(32, channels[0], kernel_size=1, stride=1, padding=0),
                              nn.BatchNorm2d(32))
        for i in range(len(self.blocks)):
            self.cells.append(self.make_block(self.blocks[i], self.stride[i], self.expansion[i],
                  self.channels[i])
        self.conv_head = nn.Conv2d(channeld[-1], 1280, kernel_size=1, stride=1, padding=0)
        self.bn_head = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)

    def make_block(C_in, C_out, block_num, stride, expansion, channels):
        cell = []
        for i in range(block_num):
            if i == 0:
                cell.append(MobileCell(expansion, C_in, C_out, stride, true))
            else:
                cell.append(MobileCell(expansion, C_in, C_out, 1, true))
        return cell                   
                                    
    def forward(self, inputs, rollout):
        out = self.sep_stem(inputs)
        for i in range(len(self.blocks)):
            for j in range(len(rollout.depth[i])):
                out = self.cells[i][j].forward_rollout(out, rollout, i, j)
        out = self.bn_head(self.conv_head(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.flatten(1)
        return self.classifier(out)

    def forward_all(self, inputs):
        out = self.sep_stem(inputs)
        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                out = self.cells[i][j].forward(out)
        out = self.bn_head(self.conv_head(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.flatten(1)
        return self.classifier(out)

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def set_hook(self):
        for name, module in self.named_modules():
            if "auxiliary" in name:
                continue
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    inputs[0].size(2) * inputs[0].size(3) / \
                                    (module.stride[0] * module.stride[1] * module.groups)
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def sub_named_members(self, genotypes,
                          prefix="", member="parameters", check_visited=False):
        prefix = prefix + ("." if prefix else "")

        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "cells":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix+mod_name):
                yield n, v

        if check_visited:
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
                depend_cell_idxes = [cell_idx - self._num_init + node_idx
                                     for node_idx in depend_nodes]
                depend_cell_idxes = [i for i in depend_cell_idxes if i >= 0 and i not in visited]
                cell_idxes += depend_cell_idxes
        else:
            visited = list(range(self._num_layers))

        for cell_idx in sorted(visited):
            cell = self.cells[cell_idx]
            genotype = genotypes[self._cell_layout[cell_idx]]
            for n, v in cell.sub_named_members(genotype,
                                               prefix=prefix + "cells.{}".format(cell_idx),
                                               member=member,
                                               check_visited=check_visited):
                yield n, v

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return MnasnetCandidateNet(self, rollout,
                               gpus=self.gpus,
                               member_mask=self.candidate_member_mask,
                               cache_named_members=self.candidate_cache_named_members,
                               virtual_parameter_only=self.candidate_virtual_parameter_only,
                               eval_no_grad=self.candidate_eval_no_grad)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("mnasnet")]


class MnasnetCandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """

    def __init__(self, super_net, rollout, member_mask, gpus=tuple(), cache_named_members=False,
                 virtual_parameter_only=True, eval_no_grad=True):
        super(MnasnetCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.search_space = super_net.search_space
        self.member_mask = member_mask
        self.cache_named_members = cache_named_members
        self.virtual_parameter_only = virtual_parameter_only
        self._cached_np = None
        self._cached_nb = None

        self._flops_calculated = False
        self.total_flops = 0

        self.genotypes = [g[1] for g in rollout.genotype_list()]
        self.genotypes_grouped = [group_and_sort_by_to_node(g[1]) for g in rollout.genotype_list() \
                                  if "concat" not in g[0]]

    @contextlib.contextmanager
    def begin_virtual(self):
        """
        On entering, store the current states (parameters/buffers) of the network
        On exiting, restore the stored states.
        Needed for surrogate steps of each candidate network,
        as different SubCandidateNet share the same set of SuperNet weights.
        """

        w_clone = {k: v.clone() for k, v in self.super_net.named_parameters()}
        if not self.virtual_parameter_only:
            buffer_clone = {k: v.clone() for k, v in self.super_net.named_buffers()}

        yield

        for n, v in self.super_net.named_parameters():
            v.data.copy_(w_clone[n])
        del w_clone

        if not self.virtual_parameter_only:
            for n, v in self.super_net.named_buffers():
                v.data.copy_(buffer_clone[n])
            del buffer_clone

    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net.forward(inputs, self.rollout)

    def forward(self, inputs, single=False): #pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs={"single": True})

    def named_parameters(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                # use cached members
                if self._cached_np is None:
                    self._cached_np = []
                    for n, v in self.active_named_members(member="parameters", prefix=""):
                        self._cached_np.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_np:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(member="parameters", prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_parameters(prefix=prefix):
                yield n, v

    def named_buffers(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                if self._cached_nb is None:
                    self._cached_nb = []
                    for n, v in self.active_named_members(member="buffers", prefix=""):
                        self._cached_nb.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_nb:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(member="buffers", prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_buffers(prefix=prefix):
                yield n, v

    def active_named_members(self, member, prefix="", recurse=True, check_visited=False):
        """
        Get the generator of name-member pairs active
        in this candidate network. Always recursive.
        """
        # memo, there are potential weight sharing, e.g. when `tie_weight` is True in rnn_super_net,
        # encoder/decoder share weights. If there is no memo, `sub_named_members` will return
        # 'decoder.weight' and 'encoder.weight', both refering to the same parameter, whereasooo
        # `named_parameters` (with memo) will only return 'encoder.weight'. For possible future
        # weight sharing, use memo to keep the consistency with the builtin `named_parameters`.
        memo = set()
        for n, v in self.super_net.sub_named_members(self.genotypes,
                                                     prefix=prefix,
                                                     member=member,
                                                     check_visited=check_visited):
            if v in memo:
                continue
            memo.add(v)
            yield n, v

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        member_lst = []
        for n, v in itertools.chain(self.active_named_members(member="parameters", prefix=""),
                                    self.active_named_members(member="buffers", prefix="")):
            member_lst.append((n, v))
        state_dict = OrderedDict(member_lst)
        return state_dict

