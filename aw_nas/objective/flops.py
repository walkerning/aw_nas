# -*- coding: utf-8 -*-

import torch
from torch import nn

from aw_nas.utils.torch_utils import accuracy
from aw_nas.objective.base import BaseObjective

class FlopsObjective(BaseObjective):
    NAME = "flops"

    def __init__(self, search_space):
        super(FlopsObjective, self).__init__(search_space)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return ["flops"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get flops.
        """
        cand_net.reset_flops()
        cand_net.forward(inputs)
        if isinstance(cand_net, nn.DataParallel):
            flops = cand_net.module.total_flops
        else:
            flops = cand_net.super_net.total_flops if hasattr(cand_net, "super_net") else \
                    cand_net.total_flops
        if hasattr(cand_net, "super_net"):
            cand_net.super_net._flops_calculated = True
        return [flops]

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        pass 

    def get_loss_item(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        return 0

