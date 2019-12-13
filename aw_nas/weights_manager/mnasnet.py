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
from aw_nas.weights_manager.base import CandidateNet, BaseWeightsManager
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp
from aw_nas.utils import data_parallel, use_params
from aw_nas.ops import *

__all__ = ["MNasNetCandidateNet", "MNasNetSupernet"]

class MobileCell(MobileNetBlock):
    def __init__(self, expansion, C, C_out, stride, affine, kernel_size=3):
        super(MobileCell, self).__init__(expansion, C, C_out, stride, affine, kernel_size)
        self.expansion = expansion

    def forward_rollout(self, inputs, rollout, ind_i, ind_j):
        self.conv1.weight.data = self.conv1.state_dict()['weight']
        if "bias" in self.conv1.state_dict().keys():
            self.conv1.bias.data = self.conv1.state_dict()['bias']
        self.conv2.weight.data = self.conv2.state_dict()['weight']
        if "bias" in self.conv2.state_dict().keys():
            self.conv2.bias.data = self.conv2.state_dict()['bias']
        self.conv3.weight.data = self.conv3.state_dict()['weight']
        if "bias" in self.conv3.state_dict().keys():
            self.conv3.bias.data = self.conv3.state_dict()['bias']
        self.bn1.weight.data = self.bn1.state_dict()['weight']
        self.bn1.bias.data = self.bn1.state_dict()['bias']
        self.bn2.weight.data = self.bn2.state_dict()['weight']
        self.bn2.bias.data = self.bn2.state_dict()['bias']
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
            mask = torch.ones([weights.shape[0]]).to(weights.device)
            mask[cut_channels] = 0
            weights = weights * mask.view([-1, 1, 1, 1])
            object.__setattr__(self.conv2, "weight", weights)
            if "bias" in self.conv2.state_dict().keys():
                bias = self.conv2.bias * mask
                object.__setattr__(self.conv2, "bias", bias)
            weights = self.bn2.weight * mask
            bias = self.bn2.bias * mask
            object.__setattr__(self.bn2, "weight", weights)
            object.__setattr__(self.bn2, "bias", bias)
            weights = self.conv1.weight * mask.view([-1, 1, 1, 1])
            object.__setattr__(self.conv1, "weight", weights)
            if "bias" in self.conv1.state_dict().keys():
                bias = self.conv1.bias * mask
                object.__setattr__(self.conv1, "bias", bias)
            weights = self.bn1.weight * mask
            bias = self.bn1.bias * mask
            object.__setattr__(self.bn1, "weight", weights)
            object.__setattr__(self.bn1, "bias", bias)
            weights = self.conv3.weight * mask.view([1, -1, 1, 1])
            object.__setattr__(self.conv3, "weight", weights)
        out = super(MobileCell, self).forward(inputs)
        return out

class MNasNetSupernet(BaseWeightsManager, nn.Module):
    """
    A Mnasnet super network
    """
    NAME = "mnasnet_supernet"

    def __init__(self, search_space, device, rollout_type='mnasnet_ofa',
                 blocks=[6, 6, 6, 6, 6, 6], stride=[1, 2, 2, 1, 2, 1],
                 expansion=[6, 6, 6, 6, 6, 6], channels=[16, 24, 40, 80, 96, 192, 320],
                 num_classes=10, gpus=tuple()):
        super(MNasNetSupernet, self).__init__(search_space, device, rollout_type)
        nn.Module.__init__(self)
        self.search_space = search_space
        self.device = device
        self.rollout_type = rollout_type
        self.gpus = gpus
        self.set_hook()
        self._flops_calculated = False
        self.total_flops = 0
        self.blocks = blocks
        self.stride = stride
        self.expansion = expansion
        self.channels = channels
        self.cells = []
        self.sep_stem = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Conv2d(32, channels[0], kernel_size=1, stride=1, padding=0),
                              nn.BatchNorm2d(channels[0]))
        for i in range(len(self.blocks)):
            self.cells.append(
                nn.ModuleList(
                    self.make_block(self.channels[i], self.channels[i+1], self.blocks[i],
                                    self.stride[i], self.expansion[i])))
        self.cells = nn.ModuleList(self.cells)
        self.conv_head = nn.Conv2d(self.channels[-1], 1280, kernel_size=1, stride=1, padding=0)
        self.bn_head = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)

        self.to(self.device)

    def make_block(self, C_in, C_out, block_num, stride, expansion):
        cell = []
        for i in range(block_num):
            if i == 0:
                cell.append(MobileCell(expansion, C_in, C_out, stride, True))
            else:
                cell.append(MobileCell(expansion, C_out, C_out, 1, True))
        return cell                   
                                    
    def forward(self, inputs, rollout):
        out = self.sep_stem(inputs)
        for i in range(len(self.blocks)):
            for j in range(rollout.depth[i]):
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

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return MNasNetCandidateNet(self, rollout,
                               gpus=self.gpus)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("mnasnet_ofa")]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def save(self, path):
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # apply the gradients
        optimizer.step()

    def step_current_gradients(self, optimizer):
        optimizer.step()

    def set_device(self, device):
        self.device = device
        self.to(device)

class MNasNetCandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """

    def __init__(self, super_net, rollout, gpus=tuple()):
        super(MNasNetCandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.search_space = super_net.search_space

        self._flops_calculated = False
        self.total_flops = 0
        self.rollout = rollout

    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net.forward(inputs, self.rollout)

    def forward(self, inputs, single=False): #pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs={"single": True})

