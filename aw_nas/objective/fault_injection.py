# -*- coding: utf-8 -*-
"""
Contributed by Wilson Li.
"""

import torch
import random
from torch import nn
import numpy as np

from aw_nas.utils.torch_utils import accuracy
from aw_nas.objective.base import BaseObjective

class FaultInjector(object):
    def __init__(self, mode="0-1flip", fault_bias=128):
        self.mode = mode
        self.fault_bias = fault_bias
        self.random_inject = 0.001
        self.gaussian_std = 1.
        self.mode = "gaussian"

    @staticmethod
    def inject2num(num_, min_, max_, step, bit_pos, mode):
        out = 0
        num = int(round(float(num_) / float(step)))
        if mode == "1-0flip":
            out = np.int8(num & bit_pos) * step
        elif mode == "0-1flip":
            out = np.int8(num | bit_pos) * step
        else:
            print("Error: no such mode!")
            exit(1)
        return out

    @staticmethod
    def get_index_by_shape(index, shape):
        index_list = []
        newshape = list(shape) + [1]
        for ind in range(len(newshape) - 1):
            index_s = (index % shape[ind]) // newshape[ind + 1]
            index_list.append(int(index_s))
        return tuple(index_list)

    def set_random_inject(self, value):
        self.random_inject = value

    def inject_gaussian(self, out):
        gaussian = torch.rand(out.shape) * self.gaussian_std
        out = out + gaussian
        return out

    def inject_fixed(self, out):
        random_tensor = out.clone()
        random_tensor.random_(0, int(1. / self.random_inject))
        scale = torch.ceil(torch.log(torch.max(torch.max(torch.abs(out)),
                                               torch.tensor(1e-5).float().to(out.device))) / np.log(2.))
        step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]).to(out.device),
                                                 requires_grad=False),
                         (scale.float() - 7.))
        fault_bias = step * 128.
        fault_ind = (random_tensor < 1)
        normal_ind = (random_tensor >= 1)
        random_tensor[fault_ind] = fault_bias
        random_tensor[normal_ind] = 0
        max_ = torch.max(torch.abs(out))
        out = out + random_tensor
        out[out > max_] = max_
        out[out < -max_] = -max_
        return out

    def inject(self, out):
        return eval("inject" + self.mode)(out)

class FaultInjectionObjective(BaseObjective):
    NAME = "fault_injection"

    def __init__(self, search_space, fault_reward_coeff=0.2, inject_prob=0.001, fault_mode="gaussian", gaussian_std=1.):
        super(FaultInjectionObjective, self).__init__(search_space)
        assert 0. <= fault_reward_coeff <= 1.
        self.injector = FaultInjector()
        self.injector.set_random_inject(inject_prob)
        self.injector.mode = fault_mode
        self.injector.gaussian_std = gaussian_std
        self.fault_reward_coeff = fault_reward_coeff

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @classmethod
    def perf_names(cls):
        return ["acc_clean", "acc_fault"]

    def get_reward(self, inputs, logits, targets, cand_net):
        perfs = self.get_perfs(inputs, logits, targets, cand_net)
        return perfs[0] * (1 - self.fault_reward_coeff) + perfs[1] * self.fault_reward_coeff

    def get_perfs(self, inputs, logits, targets, cand_net):
        """
        Get top-1 acc.
        """
        def inject(state, context):
            if context.is_end_of_cell or context.is_end_of_step:
                return
            context.last_state = self.injector.inject(state)
        # cand_net.train()
        logits_f = cand_net.forward_one_step_callback(inputs, callback=inject)
        return float(accuracy(logits, targets)[0]) / 100, float(accuracy(logits_f, targets)[0]) / 100

    def get_loss(self, inputs, logits, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            inputs: logits
            targets: labels
        """
        return nn.CrossEntropyLoss()(logits, targets)
