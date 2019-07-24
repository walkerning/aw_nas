# -*- coding: utf-8 -*-
"""
Adversarial robustness objective, and corresponding weights_manager.

Copyright (c) 2019 Xuefei Ning
"""

import weakref
import functools
import contextlib
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable

from aw_nas import AwnasPlugin, utils
from aw_nas.objective.base import BaseObjective
from aw_nas.weights_manager.super_net import SuperNet, SubCandidateNet
from aw_nas.weights_manager.diff_super_net import DiffSuperNet, DiffSubCandidateNet
from aw_nas.utils.torch_utils import accuracy
from aw_nas.utils.exception import expect, ConfigException

class PgdAdvGenerator(object):
    def __init__(self, epsilon, n_step, step_size, rand_init):
        self.epsilon = epsilon
        self.n_step = n_step
        self.step_size = step_size
        self.rand_init = rand_init
        self.criterion = nn.CrossEntropyLoss()

    def generate_adv(self, inputs, outputs, targets, net):
        inputs_pgd = Variable(inputs.data.clone(), requires_grad=True)
        if self.rand_init:
            eta = inputs.new(inputs.size).uniform_(-self.epsilon, self.epsilon)
            inputs_pgd.data = inputs + eta
        for _ in range(self.n_step):
            out = net(inputs_pgd)
            loss = self.criterion(out, Variable(targets))
            loss.backward()
            eta = self.step_size * inputs_pgd.grad.data.sign()
            inputs_pgd = Variable(inputs_pgd.data + eta, requires_grad=True)

            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(inputs_pgd.data - inputs, -self.epsilon, self.epsilon)
            inputs_pgd.data = inputs + eta
        net.zero_grad()
        return inputs_pgd.data

class AdversarialRobustnessObjective(BaseObjective):
    NAME = "adversarial_robustness_objective"
    SCHEDULABLE_ATTRS = []

    def __init__(self, search_space,
                 # adversarial
                 epsilon=0.03, n_step=5, step_size=0.0078, rand_init=False,
                 # loss
                 adv_loss_coeff=0.,
                 as_controller_regularization=False,
                 as_evaluator_regularization=False,
                 # reward
                 adv_reward_coeff=0.,
                 schedule_cfg=None):
        super(AdversarialRobustnessObjective, self).__init__(search_space, schedule_cfg)

        # adversarial generator
        self.adv_generator = PgdAdvGenerator(epsilon, n_step, step_size, rand_init)

        self.adv_reward_coeff = adv_reward_coeff
        self.adv_loss_coeff = adv_loss_coeff
        self.as_controller_regularization = as_controller_regularization
        self.as_evaluator_regularization = as_evaluator_regularization
        self.cache_hit = 0
        self.cache_miss = 0
        if self.adv_loss_coeff > 0:
            expect(self.as_controller_regularization or self.as_evaluator_regularization,
                   "When `adv_loss_coeff` > 0, you should either use this adversarial loss"
                   " as controller regularization or as evaluator regularization, or both. "
                   "By setting `as_controller_regularization` and `as_evaluator_regularization`.",
                   ConfigException)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(cls):
        return ["acc_clean", "acc_adv"]

    def get_reward(self, inputs, outputs, targets, cand_net):
        perfs = self.get_perfs(inputs, outputs, targets, cand_net)
        return perfs[0] * (1 - self.adv_reward_coeff) + perfs[1] * self.adv_reward_coeff

    def get_perfs(self, inputs, outputs, targets, cand_net):
        inputs_adv = self._gen_adv(inputs, outputs, targets, cand_net)
        outputs_adv = cand_net(inputs_adv)
        return float(accuracy(outputs, targets)[0]) / 100, \
            float(accuracy(outputs_adv, targets)[0]) / 100

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
        if self.adv_loss_coeff > 0 and \
           ((add_controller_regularization and self.as_controller_regularization) or \
            (add_evaluator_regularization and self.as_evaluator_regularization)):
            inputs_adv = self._gen_adv(inputs, outputs, targets, cand_net)
            outputs_adv = cand_net(inputs_adv)
            ce_loss_adv = nn.CrossEntropyLoss()(outputs_adv, targets)
            loss = (1 - self.adv_loss_coeff) * loss + self.adv_loss_coeff * ce_loss_adv
        return loss

    def on_epoch_end(self, epoch):
        super(AdversarialRobustnessObjective, self).on_epoch_end(epoch)
        self.logger.info("Adversarial cache hit/miss : %d/%d", self.cache_hit, self.cache_miss)
        self.cache_miss = 0
        self.cache_hit = 0

    def _gen_adv(self, inputs, outputs, targets, cand_net):
        # NOTE: tightly-coupled with CacheAdvCandidateNet
        if hasattr(cand_net, "cached_advs") and inputs in cand_net.cached_advs:
            self.cache_hit += 1
            return cand_net.cached_advs[inputs]
        self.cache_miss += 1
        inputs_adv = self.adv_generator.generate_adv(inputs, outputs, targets, cand_net)
        if hasattr(cand_net, "cached_advs"):
            cand_net.cached_advs[inputs] = inputs_adv
        return inputs_adv

    @property
    def n_step(self):
        return self.adv_generator.n_step

    @n_step.setter
    def n_step(self, value):
        self.adv_generator.n_step = value

    @property
    def epsilon(self):
        return self.adv_generator.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.adv_generator.epsilon = value

    @property
    def step_size(self):
        return self.adv_generator.step_size

    @step_size.setter
    def step_size(self, value):
        self.adv_generator.step_size = value

class _Cache(OrderedDict):
    def __init__(self, *args, **kwargs):
        self.buffer_size = kwargs.pop("buffer_size", 3)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size()

    def _check_size(self):
        if self.buffer_size is not None:
            while len(self) > self.buffer_size:
                self.popitem(last=False)

class CacheAdvCandidateNet(SubCandidateNet):
    def __init__(self, *args, **kwargs):
        super(CacheAdvCandidateNet, self).__init__(*args, **kwargs)
        # 820s -> 400s
        self.cached_advs = _Cache([], buffer_size=3)

    def clear_cache(self):
        """
        There are model updates. Clear the cache.
        """
        self.cached_advs.clear()

    @contextlib.contextmanager
    def begin_virtual(self):
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

        self.clear_cache()

    def train_queue(self, queue, optimizer, criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                    eval_criterions=None, steps=1, **kwargs):
        assert steps > 0
        self._set_mode("train")

        average_ans = None
        for _ in range(steps):
            data = next(queue)
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            _, targets = data
            outputs = self.forward_data(*data, **kwargs)
            loss = criterion(data[0], outputs, targets)
            if eval_criterions:
                ans = utils.flatten_list([c(data[0], outputs, targets) for c in eval_criterions])
                if average_ans is None:
                    average_ans = ans
                else:
                    average_ans = [s + a for s, a in zip(average_ans, ans)]
            self.zero_grad()
            loss.backward()
            optimizer.step()
            self.clear_cache()

        if eval_criterions:
            return [s / steps for s in average_ans]
        return []

class CacheAdvSuperNet(SuperNet):
    NAME = "adv_supernet"

    @functools.wraps(SuperNet.__init__)
    def __init__(self, *args, **kwargs):
        super(CacheAdvSuperNet, self).__init__(*args, **kwargs)
        if self.candidate_eval_no_grad:
            self.logger.warning(
                "candidate_eval_no_grad for CacheAdvSuperNet should be set to `false` (not {}), "
                "automatically changed to `false`".format(self.candidate_eval_no_grad))
        self.candidate_eval_no_grad = False
        self.assembled = 0
        self.candidate_map = weakref.WeakValueDictionary()

    def assemble_candidate(self, rollout):
        cand_net = CacheAdvCandidateNet(
            self, rollout, gpus=self.gpus,
            member_mask=self.candidate_member_mask,
            cache_named_members=self.candidate_cache_named_members,
            virtual_parameter_only=self.candidate_virtual_parameter_only,
            eval_no_grad=self.candidate_eval_no_grad)
        self.candidate_map[self.assembled] = cand_net
        self.assembled += 1
        return cand_net

    def step_current_gradients(self, optimizer):
        assert 0, "step_current_gradient should not be called!"

    def step(self, gradients, optimizer):
        super(CacheAdvSuperNet, self).step(gradients, optimizer)
        for cand_net in self.candidate_map.values():
            cand_net.clear_cache()

    def load(self, path):
        super(CacheAdvSuperNet, self).load(path)
        for cand_net in self.candidate_map.values():
            cand_net.clear_cache()

    def __setstate__(self):
        super(CacheAdvSuperNet, self).__setstate__(state)
        self.candidate_map = weakref.WeakValueDictionary()

    def __getstate__(self):
        state = super(CacheAdvSuperNet, self).__getstate__()
        del state["candidate_map"]
        return state

class CacheAdvDiffCandidateNet(DiffSubCandidateNet):
    def __init__(self, *args, **kwargs):
        super(CacheAdvDiffCandidateNet, self).__init__(*args, **kwargs)
        # 820s -> 400s
        self.cached_advs = _Cache([], buffer_size=3)

    def clear_cache(self):
        """
        There are model updates. Clear the cache.
        """
        self.cached_advs.clear()

    @contextlib.contextmanager
    def begin_virtual(self):
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

        self.clear_cache()

    def train_queue(self, queue, optimizer, criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                    eval_criterions=None, steps=1, **kwargs):
        assert steps > 0
        self._set_mode("train")

        average_ans = None
        for _ in range(steps):
            data = next(queue)
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            _, targets = data
            outputs = self.forward_data(*data, **kwargs)
            loss = criterion(data[0], outputs, targets)
            if eval_criterions:
                ans = utils.flatten_list([c(data[0], outputs, targets) for c in eval_criterions])
                if average_ans is None:
                    average_ans = ans
                else:
                    average_ans = [s + a for s, a in zip(average_ans, ans)]
            self.zero_grad()
            loss.backward()
            optimizer.step()
            self.clear_cache()

        if eval_criterions:
            return [s / steps for s in average_ans]
        return []

class CacheAdvDiffSuperNet(DiffSuperNet):
    NAME = "adv_diff_supernet"

    @functools.wraps(DiffSuperNet.__init__)
    def __init__(self, *args, **kwargs):
        super(CacheAdvDiffSuperNet, self).__init__(*args, **kwargs)
        if self.candidate_eval_no_grad:
            self.logger.warning(
                "candidate_eval_no_grad for CacheAdvSuperNet should be set to `false` (not {}), "
                "automatically changed to `false`".format(self.candidate_eval_no_grad))
        self.candidate_eval_no_grad = False
        self.assembled = 0
        self.candidate_map = weakref.WeakValueDictionary()

    def assemble_candidate(self, rollout):
        cand_net = CacheAdvDiffCandidateNet(
            self, rollout, gpus=self.gpus,
            virtual_parameter_only=self.candidate_virtual_parameter_only,
            eval_no_grad=self.candidate_eval_no_grad)
        self.candidate_map[self.assembled] = cand_net
        self.assembled += 1
        return cand_net

    def step_current_gradients(self, optimizer):
        assert 0, "step_current_gradient should not be called!"

    def step(self, gradients, optimizer):
        super(CacheAdvDiffSuperNet, self).step(gradients, optimizer)
        for cand_net in self.candidate_map.values():
            cand_net.clear_cache()

    def load(self, path):
        super(CacheAdvDiffSuperNet, self).load(path)
        for cand_net in self.candidate_map.values():
            cand_net.clear_cache()

    def __setstate__(self):
        super(CacheAdvDiffSuperNet, self).__setstate__(state)
        self.candidate_map = weakref.WeakValueDictionary()

    def __getstate__(self):
        state = super(CacheAdvDiffSuperNet, self).__getstate__()
        del state["candidate_map"]
        return state

class AdversarialRobustnessPlugin(AwnasPlugin):
    NAME = "adversarial_robustness"
    objective_list = [AdversarialRobustnessObjective]
    weights_manager_list = [CacheAdvSuperNet, CacheAdvDiffSuperNet]
