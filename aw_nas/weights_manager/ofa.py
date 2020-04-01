# -*- coding: utf-8 -*-
"""
OFA super net.
"""

import contextlib
import itertools
from collections import OrderedDict

import numpy as np
import six
import timeit
import torch
import torch.nn.functional as F
from aw_nas.common import assert_rollout_type, group_and_sort_by_to_node
from aw_nas.ops import *
from aw_nas.utils import data_parallel, use_params
from aw_nas.utils.common_utils import make_divisible
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.weights_manager.ofa_backbone import BaseBackboneArch
from torch import nn
from torch.nn.parameter import Parameter
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.utils import DistributedDataParallel

__all__ = ["OFACandidateNet", "OFASupernet"]


class OFASupernet(BaseWeightsManager, nn.Module):
    NAME = "ofa_supernet"

    def __init__(self, search_space, device, 
                 rollout_type,
                 backbone_type='mbv2_backbone',
                 backbone_cfg={},
                 num_classes=10,
                 multiprocess=False, gpus=tuple(), schedule_cfg=None):
        super(OFASupernet, self).__init__(search_space, device, rollout_type, schedule_cfg)
        nn.Module.__init__(self)
        self.backbone = BaseBackboneArch.get_class_(backbone_type)(device, schedule_cfg=schedule_cfg, **backbone_cfg)

        self.multiprocess = multiprocess
        self.gpus = gpus
        object.__setattr__(self, 'parallel_model', self)

        self._parallelize()

    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self).to(self.device)
            object.__setattr__(self, 'parallel_model', DistributedDataParallel(net, self.gpus))

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def forward(self, inputs, rollout=None):
        return self.backbone.forward_rollout(inputs, rollout)

    def set_hook(self):
        for name, module in self.named_modules():
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
        model = OFACandidateNet(self, rollout, gpus=self.gpus)
        return model

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("mnasnet_ofa")]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def save(self, path):
        torch.save(
            {
                "epoch": self.epoch,
                "state_dict": self.state_dict(),
                # "norms": self.norms
            }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
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

class OFACandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """
    def __init__(self, super_net, rollout, gpus=tuple()):
        super(OFACandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.multiprocess = super_net.multiprocess
        self.search_space = super_net.search_space

        self._flops_calculated = False
        self.total_flops = 0
        self.rollout = rollout


    def get_device(self):
        return self._device

    def _forward(self, inputs):
        out = self.super_net.forward(inputs, self.rollout)
        return out

    def forward(self, inputs, single=False):  #pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        
        if self.multiprocess:
            out = self.super_net.parallel_model.forward(inputs, self.rollout)
        elif len(self.gpus) > 1:
            out = data_parallel(self, (inputs, ),
                             self.gpus,
                             module_kwargs={"single": True})

        return out


        