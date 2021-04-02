# -*- coding: utf-8 -*-
#pylint: disable=arguments-differ
"""
Decision, search space, rollout.
And searchable block primitives.
"""

import re
import abc
import copy
from collections import OrderedDict

import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas.base import Component
from aw_nas.common import BaseRollout, SearchSpace
from aw_nas.utils import abstractclassmethod, expect, ConfigException


class GermSearchSpace(SearchSpace):
    NAME = "germ"

    def __init__(self, search_space_cfg_file=None):
        super().__init__()
        expect(search_space_cfg_file is not None, "Must specify search_space_cfg_file")
        with open(search_space_cfg_file, "r") as r_f:
            self.ss_cfg = yaml.load(r_f)
        self.decisions = OrderedDict()
        for decision_id, value in self.ss_cfg["decisions"].items():
            decision = BaseDecision.get_class_(value[0]).from_string(value[1])
            self.decisions[decision_id] = decision
        self.blocks = self.ss_cfg["blocks"]

        self.decision_ids = list(self.decisions.keys())
        self.num_decisions = len(self.decisions)
        self.num_blocks = len(self.blocks)

    def genotype(self, arch):
        return str(arch)

    def genotype_from_str(self, string):
        return string

    def random_sample(self):
        # generate a random sample for each decision
        arch = OrderedDict([
            (decision_id, decision.random_sample())
            for decision_id, decision in self.decisions.items()])
        return GermRollout(arch, search_space=self)

    def rollout_from_genotype(self, genotype):
        arch = eval(genotype) #pylint: disable=eval-used
        return GermRollout(arch, self, candidate_net=None)

    def mutate(self, rollout, mutate_num=None, mutate_proportion=None, mutate_prob=None):
        expect(
            sum([value is not None for value in [mutate_num, mutate_proportion, mutate_prob]]) == 1,
            "One and only one of `mutate_num, mutate_proportion, mutate_prob` should be specified.",
            ConfigException)

        if mutate_proportion is not None:
            mutate_num = int(self.num_decisions * mutate_proportion)
        if mutate_num is not None:
            idxes = np.random.choice(
                self.decision_ids, size=mutate_num, replace=False)
        elif mutate_prob is not None:
            idxes = [
                self.decision_ids[ind]
                for ind in np.where(np.random.rand(self.num_decisions) < mutate_prob)[0]]
        new_arch = copy.deepcopy(rollout.arch)
        for idx in idxes:
            new_arch[idx] = self.decisions[idx].mutate(new_arch[idx])
        return GermRollout(new_arch, search_space=self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        with open(filename, "w") as w_f:
            w_f.write(genotypes)
        return [filename]

    @classmethod
    def supported_rollout_types(cls):
        return ["germ"]

    def distance(self, arch1, arch2):
        return NotImplementedError()

    def on_epoch_start(self, epoch):
        # call on_epoch_start of all decisions
        [decision.on_epoch_start(epoch) for decision in self.decisions.values()]

class GermRollout(BaseRollout):
    NAME = "germ"
    supported_components = [
        ("trainer", "simple"), ("evaluator", "mepa"),
        ("evaluator", "discrete_shared_weights"),
        ("evaluator", "differentiable_shared_weights")]

    def __init__(self, decision_dict, search_space, candidate_net=None):
        super().__init__()
        self.arch = decision_dict
        self.search_space = search_space
        self.candidate_net = candidate_net
        self._perf = OrderedDict()
        self._genotype = None

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = str(self.arch)
        return self._genotype

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(self.genotype, filename=filename, label=label)

# ---- Searchable Blocks ----
class SearchableBlock(nn.Module, Component):
    REGISTRY = "searchable_block"

    def __init__(self, ctx):
        super(SearchableBlock, self).__init__()
        Component.__init__(self, schedule_cfg=None)
        self._decisions = OrderedDict()
        self.ctx = ctx

    def __setattr__(self, name, value):
        if isinstance(value, BaseDecision):
            decisions = self.__dict__.get("_decisions")
            if decisions is None:
                raise Exception("cannot assign decision before SearchableBlock.__init__() call")
            decisions[name] = value
        return super().__setattr__(name, value)

    def __getattr__(self, name):
        if "_decisions" in self.__dict__:
            if name in self.__dict__["_decisions"]:
                return self.__dict__["_decisions"][name]
        return super().__getattr__(name)

    def named_decisions(self, prefix="", recurse=False):
        # By default, not recursive
        _get_named_decisions = lambda mod: mod._decisions.items()
        memo = set()
        mod_list = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mod_name, mod in mod_list:
            if isinstance(mod, SearchableBlock):
                for d_id, d_obj in _get_named_decisions(mod):
                    if d_obj is None or d_obj in memo:
                        continue
                    memo.add(d_obj)
                    d_name = mod_name + ("." if mod_name else "") + d_id
                    yield d_name, d_obj

    def _get_decision(self, decision_obj, rollout):
        if isinstance(decision_obj, BaseDecision):
            return rollout.arch[decision_obj.decision_id]
        return decision_obj

    def forward(self, *args, **kwargs):
        return self.forward_rollout(self.ctx.rollout, *args, **kwargs)

    @abc.abstractmethod
    def forward_rollout(self, rollout, *args, **kwargs):
        pass

    @abc.abstractmethod
    def finalize_rollout(self, rollout):
        pass

    @abstractclassmethod
    def searchable_dimensions(cls):
        pass


class SearchableConvBNBlock(SearchableBlock):
    NAME = "conv_bn"

    def __init__(self, ctx, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__(ctx)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(out_channels, BaseDecision):
            self.max_out_channels = out_channels.range()[1]
            assert isinstance(out_channels, Choices)
            # only support Choice decision for now, use discretized channel/kernel mask
            # init feature mask for channel cohices
            self.register_buffer(
                "channel_masks", torch.zeros(out_channels.num_choices, self.max_out_channels))
            for i in range(out_channels.num_choices):
                self.channel_masks[i][:out_channels.choices[i]] = 1.0
            self._out_channels_choice_to_ind = {c: i for i, c in enumerate(out_channels.choices)}
        else:
            self.max_out_channels = out_channels
            self.channel_masks = None

        if isinstance(kernel_size, BaseDecision):
            # use centered mask
            self.max_kernel_size = kernel_size.range()[1]
            assert isinstance(kernel_size, Choices)
            expect(all(choice % 2 == 1 for choice in kernel_size.choices),
                   "Only support odd kernel_size choices.")

            # init weight mask for kernel size choices
            mask = np.abs(np.arange(-(self.max_kernel_size-1)//2, (self.max_kernel_size+1)//2)) \
                   <= np.array([(c-1)//2 for c in kernel_size.choices])[:, None]
            mask = mask.astype(np.float32)
            masks = torch.tensor(np.expand_dims(mask, -2) * np.expand_dims(mask, -1))
            self.register_buffer("kernel_masks", masks)
            self._kernel_size_choice_to_ind = {c: i for i, c in enumerate(kernel_size.choices)}
        else:
            self.max_kernel_size = kernel_size
            self.kernel_masks = None

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.max_out_channels,
            kernel_size=self.max_kernel_size,
            stride=self.stride,
            **kwargs
        )
        self.bn = nn.BatchNorm2d(self.max_out_channels)
        
    def forward_rollout(self, rollout, inputs):
        # stride
        r_s = self._get_decision(self.stride, rollout)
        # kernel size
        r_k_s = self._get_decision(self.kernel_size, rollout)
        padding = ((r_k_s - 1) // 2, (r_k_s - 1) // 2)

        if self.kernel_masks is not None:
            w_mask = self.kernel_masks[self._kernel_size_choice_to_ind[r_k_s]]
            object.__setattr__(self.conv, "weight", w_mask * self.conv._parameters["weight"])

        # forward conv and bn
        out = F.conv2d(inputs, self.conv.weight, self.conv.bias, (r_s, r_s),
                       padding, self.conv.dilation, self.conv.groups)
        out = self.bn(out)

        if self.kernel_masks is not None:
            # set back conv.weights to the full kernel
            object.__setattr__(self.conv, "weight", self.conv._parameters["weight"])

        # mask output channels
        if self.channel_masks is not None:
            r_o_c = self._get_decision(self.out_channels, rollout)
            f_mask = self.channel_masks[self._out_channels_choice_to_ind[r_o_c]].reshape(1, -1, 1, 1)
            out = out * f_mask
        # Should we mask before bn?
        # Currently not important, since this decision is only relevant
        # when bn statistics need to be used without calibration/retraining
        return out

    def finalize_rollout(self, rollout):
        # TODO: return a finalized convbn nn.module @tcc
        pass

    @classmethod
    def searchable_dimensions(cls):
        return ["out_channels", "kernel_size", "stride"]
# ---- End Searchable Blocks ----


# ---- Decisions ----
class BaseDecision(Component):
    REGISTRY = "decision"

    def __repr__(self):
        return self.to_string()

    @abc.abstractmethod
    def random_sample(self):
        pass

    @abc.abstractmethod
    def mutate(self, old):
        pass

    @abc.abstractmethod
    def range(self):
        pass

    @abc.abstractmethod
    def to_string(self):
        pass

    @abstractclassmethod
    def from_string(cls, string):
        pass


class Choices(BaseDecision):
    NAME = "choices"

    def __init__(self, choices, schedule_cfg=None):
        super().__init__(schedule_cfg=schedule_cfg)
        self.choices = choices
        self.num_choices = len(choices)

    def random_sample(self):
        return np.random.choice(self.choices, size=1)[0]

    def mutate(self, old):
        old_ind = self.choices.index(old)
        bias = np.random.randint(1, self.num_choices)
        new_ind = (old_ind + bias) % self.num_choices
        return self.choices[new_ind]

    def range(self):
        return (min(self.choices), max(self.choices))

    def to_string(self):
        return "Choices({})".format(self.choices)

    @classmethod
    def from_string(cls, string):
        return cls(eval(re.search(r"Choices\((.+)\)", string).group(1)))
# ---- End Decisions ----
