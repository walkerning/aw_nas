# -*- coding: utf-8 -*-
"""
Adversarial robustness objective, and corresponding weights_manager.
Copyright (c) 2019 Xuefei Ning, Junbo Zhao
"""
#pylint: disable=missing-docstring,invalid-name,no-self-use

import sys
import weakref
import functools
import contextlib
from collections import OrderedDict
import os

import yaml
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from aw_nas import AwnasPlugin, utils
from aw_nas.objective.base import BaseObjective
from aw_nas.weights_manager.super_net import SuperNet, SubCandidateNet
from aw_nas.weights_manager.diff_super_net import DiffSuperNet, DiffSubCandidateNet
from aw_nas.utils.torch_utils import accuracy, _to_device
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.common import get_search_space
from aw_nas.final.base import FinalModel
from aw_nas.utils import DistributedDataParallel

# ---- different types of Adversaries ----
class PgdAdvGenerator(object):
    def __init__(self, epsilon, n_step, step_size, rand_init, mean, std):
        self.epsilon = epsilon
        self.n_step = n_step
        self.rand_init = rand_init
        self.criterion = nn.CrossEntropyLoss()
        self.step_size = step_size
        self.mean = torch.reshape(torch.tensor(mean), (3, 1, 1))
        self.std = torch.reshape(torch.tensor(std), (3, 1, 1))
        self.step_size = (step_size / self.std)

    def generate_adv(self, inputs, outputs, targets, net):
        self.mean = self.mean.to(inputs.device)
        self.std = self.std.to(inputs.device)
        self.step_size = self.step_size.to(inputs.device)
        inputs_pgd = Variable(inputs.data.clone(), requires_grad=True)

        if self.rand_init:
            # Re-adjust pixel values to [0,1]
            inputs_pgd.data = inputs_pgd * self.std + self.mean
            eta = inputs.new(inputs.size()).uniform_(-self.epsilon, self.epsilon)
            inputs_pgd.data = inputs_pgd + eta
            # Re-re-adjust pixel values
            inputs_pgd.data = (inputs_pgd - self.mean) / self.std

        # Re-adjust pixel values to [0,1]
        inputs_clone = inputs.data.clone() * self.std + self.mean
        # inputs.data = inputs * self.std + self.mean

        for _ in range(self.n_step):
            out = net(inputs_pgd)
            loss = self.criterion(out, Variable(targets))
            loss.backward()
            eta = self.step_size * inputs_pgd.grad.data.sign()
            inputs_pgd = Variable(inputs_pgd.data + eta, requires_grad=True)
            # adjust to be within [-epsilon, epsilon]
            # Re-adjust pixel values to [0,1]
            inputs_pgd.data = inputs_pgd.data * self.std + self.mean
            eta = torch.clamp(inputs_pgd.data - inputs_clone, -self.epsilon, self.epsilon)
            inputs_pgd.data = inputs_clone + eta
            inputs_pgd.data = torch.clamp(inputs_pgd.data, 0.0, 1.0)
            # Re-re-adjust pixel values
            inputs_pgd.data = (inputs_pgd.data - self.mean) / self.std

        # Re-re-adjust pixel values
        net.zero_grad()
        return inputs_pgd.data


Adversary = {
    "FGSM": \
    (lambda epsilon, n_step, step_size, rand_init, mean, std:
     PgdAdvGenerator(epsilon, 1, epsilon, rand_init, mean, std)),
    "PGD": PgdAdvGenerator,
}

# ---- Different types of objectives ----
class AdversarialRobustnessObjective(BaseObjective):
    NAME = "adversarial_robustness_objective"
    SCHEDULABLE_ATTRS = []

    def __init__(self, search_space,
                 # adversarial
                 epsilon=0.031, n_step=7, step_size=0.0078, rand_init=False,
                 mean=None, std=None, adversary_type="PGD",
                 # loss & reward
                 adv_loss_coeff=0., adv_reward_coeff=0.,
                 as_controller_regularization=False, as_evaluator_regularization=False,
                 schedule_cfg=None):
        super(AdversarialRobustnessObjective, self).__init__(search_space, schedule_cfg)

        # adversarial generator
        expect(mean is not None and std is not None,
               "Must explicitly specify mean and std used in the data augmentation",
               ConfigException)
        self.adv_generator = Adversary[adversary_type](
            epsilon, n_step, step_size, rand_init, mean, std)
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

    def perf_names(self):
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


class ARFlopsObjective(AdversarialRobustnessObjective):
    NAME = "adversarial_robustness_flops"

    def perf_names(self):
        return ["acc_clean", "acc_adv", "flops"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        inputs_adv = self._gen_adv(inputs, outputs, targets, cand_net)
        if hasattr(cand_net, "super_net"):
            # clear the flops statistics
            cand_net.super_net.reset_flops()
        # the forward hooks will calculate the flops statistics
        outputs_adv = cand_net(inputs_adv)
        if isinstance(cand_net, nn.DataParallel):
            flops = cand_net.module.total_flops
        else:
            flops = cand_net.super_net.total_flops if hasattr(cand_net, "super_net") else \
                (cand_net.module.total_flops if isinstance(cand_net, DistributedDataParallel) \
                 else cand_net.total_flops)

        return float(accuracy(outputs, targets)[0]) / 100, \
               float(accuracy(outputs_adv, targets)[0]) / 100, flops


class BlackAdversarialRobustnessObjective(AdversarialRobustnessObjective):
    NAME = "black_adversarial_robustness_objective"
    SCHEDULABLE_ATTRS = []

    def __init__(self, search_space, source_model_path, source_model_device,
                 source_model_cfg=None, epsilon=0.031, n_step=100,
                 step_size=0.0078, rand_init=True,
                 mean=None, std=None, adversary_type="PGD",
                 adv_loss_coeff=0., as_controller_regularization=False,
                 as_evaluator_regularization=False,
                 adv_reward_coeff=0., schedule_cfg=None):
        super(BlackAdversarialRobustnessObjective, self).__init__(
            search_space, epsilon, n_step, step_size,
            rand_init, mean, std, adversary_type,
            adv_loss_coeff, as_controller_regularization, as_evaluator_regularization,
            adv_reward_coeff, schedule_cfg)

        # load the substitute model
        self.load_model(source_model_path, source_model_device, source_model_cfg)
        # temp counter
        self._adv_total, self._adv_correct = 0, 0

    def load_model(self, path, device, source_model_cfg=None):
        model_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        if os.path.exists(model_path):
            self.source_model = torch.load(model_path, map_location=torch.device("cpu")).to(device)
        else:
            model_path = os.path.join(path, "model_state.pt")
            with open(source_model_cfg, 'r') as f:
                cfg = yaml.load(f)
            ss = get_search_space(cfg["search_space_type"], **cfg["search_space_cfg"])
            self.source_model = FinalModel.get_class_(cfg["final_model_type"])(
                ss, device, **cfg["final_model_cfg"])
            self.source_model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu")))
            self.source_model = self.source_model.to(device)
        self.source_model.eval()

    def get_perfs(self, inputs, outputs, targets, cand_net):
        inputs_adv = self._gen_adv(inputs, outputs, targets, cand_net)
        outputs_adv = cand_net(inputs_adv)
        self._adv_total += len(inputs_adv)
        self._adv_correct += float(accuracy(outputs_adv, targets)[0]) / 100 * len(inputs_adv)
        print("Acc: {}".format(self._adv_correct / self._adv_total))
        sys.stdout.flush()
        return float(accuracy(outputs, targets)[0]) / 100, \
               float(accuracy(outputs_adv, targets)[0]) / 100

    def _gen_adv(self, inputs, outputs, targets, cand_net=None):
        inputs_adv = self.adv_generator.generate_adv(inputs, outputs, targets, self.source_model)
        return inputs_adv, targets

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

    def __setstate__(self, state):
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

    def train_queue(self,
                    queue,
                    optimizer,
                    criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                    eval_criterions=None,
                    steps=1,
                    aggregate_fns=None,
                    **kwargs):
        assert steps > 0

        self._set_mode("train")

        aggr_ans = []
        for _ in range(steps):
            data = next(queue)
            data = _to_device(data, self.get_device())
            _, targets = data
            outputs = self.forward_data(*data, **kwargs)
            loss = criterion(data[0], outputs, targets)
            if eval_criterions:
                ans = utils.flatten_list(
                    [c(data[0], outputs, targets) for c in eval_criterions])
                aggr_ans.append(ans)
            self.zero_grad()
            loss.backward()
            optimizer.step()
            self.clear_cache()

        if eval_criterions:
            aggr_ans = np.asarray(aggr_ans).transpose()
            if aggregate_fns is None:
                # by default, aggregate batch rewards with MEAN
                aggregate_fns = [lambda perfs: np.mean(perfs) if len(perfs) > 0 else 0.]\
                                * len(aggr_ans)
            return [
                aggr_fn(ans) for aggr_fn, ans in zip(aggregate_fns, aggr_ans)
            ]
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

    def __setstate__(self, state):
        super(CacheAdvDiffSuperNet, self).__setstate__(state)
        self.candidate_map = weakref.WeakValueDictionary()

    def __getstate__(self):
        state = super(CacheAdvDiffSuperNet, self).__getstate__()
        del state["candidate_map"]
        return state


class AdversarialRobustnessPlugin(AwnasPlugin):
    NAME = "adversarial_robustness"
    objective_list = [AdversarialRobustnessObjective, ARFlopsObjective,
                      BlackAdversarialRobustnessObjective]
    weights_manager_list = [CacheAdvSuperNet, CacheAdvDiffSuperNet]
