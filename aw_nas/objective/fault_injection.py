# -*- coding: utf-8 -*-
"""
Fault injection objective.
* Clean accuracy and fault-injected accuracy weighted for reward (for discrete controller search)
* Clean loss and fault-injected loss weighted for loss
  (for differentiable controller search or fault-injection training).

Copyright (c) 2019 Wenshuo Li
Copyright (c) 2019 Xuefei Ning
"""

import torch
import random
from torch import nn
import numpy as np

from aw_nas.utils.torch_utils import accuracy
from aw_nas.objective.base import BaseObjective
from aw_nas.utils.exception import expect, ConfigException

class FaultInjector(object):
    def __init__(self, mode="0-1flip", fault_bias=128):
        self.mode = mode
        self.fault_bias = fault_bias
        self.random_inject = 0.001

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

    def inject_gpu(self, out):
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
        # for masked bp
        normal_mask = torch.zeros_like(out)
        normal_mask[normal_ind] = 1
        masked = normal_mask * out
        out = (out - masked).detach() + masked
        return out

    def inject(self, out):
        out_origin = out
        device = out.device
        out = out.cpu()
        cumprod_shape = np.cumprod(list(reversed(out.shape)))[::-1]
        out_reshape = np.array(out.detach().reshape([cumprod_shape[0]]))
        min_, max_ = np.min(out_reshape), np.max(out_reshape)
        scale = torch.ceil(torch.log(torch.max(torch.max(torch.abs(out)),
                                               torch.tensor(1e-5).float().to(out.device))) / np.log(2.))
        step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]),
                                                 requires_grad=False), (scale.float() - 7.))
        inject_num = out_reshape.shape[0] * self.random_inject
        sample_set = random.sample(range(out_reshape.shape[0]), int(inject_num))
        for sample in sample_set:
            indexes = self.get_index_by_shape(sample, cumprod_shape)
            out[indexes] = \
                self.inject2num(out[indexes], min_, max_, step, self.fault_bias, self.mode)
        return out.to(device)


class FaultInjectionObjective(BaseObjective):
    NAME = "fault_injection"

    def __init__(self, search_space,
                 mode="0-1flip",
                 # loss
                 fault_loss_coeff=0.,
                 as_controller_regularization=False,
                 as_evaluator_regularization=False,
                 # reward
                 fault_reward_coeff=0.2, inject_prob=0.001):
        super(FaultInjectionObjective, self).__init__(search_space)
        assert 0. <= fault_reward_coeff <= 1.
        self.mode = mode
        self.injector = FaultInjector(mode)
        self.injector.set_random_inject(inject_prob)
        self.fault_loss_coeff = fault_loss_coeff
        self.as_controller_regularization = as_controller_regularization
        self.as_evaluator_regularization = as_evaluator_regularization
        if self.fault_loss_coeff > 0:
            expect(self.as_controller_regularization or self.as_evaluator_regularization,
                   "When `fault_loss_coeff` > 0, you should either use this fault-injected loss"
                   " as controller regularization or as evaluator regularization, or both. "
                   "By setting `as_controller_regularization` and `as_evaluator_regularization`.")
        self.fault_reward_coeff = fault_reward_coeff

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @classmethod
    def perf_names(cls):
        return ["acc_clean", "acc_fault"]

    def get_reward(self, inputs, outputs, targets, cand_net):
        perfs = self.get_perfs(inputs, outputs, targets, cand_net)
        return perfs[0] * (1 - self.fault_reward_coeff) + perfs[1] * self.fault_reward_coeff

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        # cand_net.train()
        outputs_f = cand_net.forward_one_step_callback(inputs, callback=self.inject)
        return float(accuracy(outputs, targets)[0]) / 100, \
            float(accuracy(outputs_f, targets)[0]) / 100

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            inputs: data inputs
            outputs: logits
            targets: labels
        """
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if self.fault_loss_coeff > 0 and \
           ((add_controller_regularization and self.as_controller_regularization) or \
            (add_evaluator_regularization and self.as_evaluator_regularization)):
            # only forward and random inject once, this might not be of high variance
            # for differentiable controller training?
            outputs_f = cand_net.forward_one_step_callback(inputs, callback=self.inject)
            ce_loss_f = nn.CrossEntropyLoss()(outputs_f, targets)
            loss += self.fault_loss_coeff * ce_loss_f
        return loss

    def inject(self, state, context):
        if context.is_end_of_cell or context.is_end_of_step:
            return
        context.last_state = self.injector.inject_gpu(state)
