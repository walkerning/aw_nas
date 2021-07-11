# -*- coding: utf-8 -*-
"""
Rejection sampling controller.
"""

import abc
import copy

import torch

from aw_nas.controller.base import BaseController
from aw_nas.base import Component
from aw_nas.common import BaseRollout


__all__ = ["BaseRejector", "CanonicalTableRejector", "RejectionSampleController"]

class BaseRejector(Component):
    REGISTRY = "rejector"

    def __init__(self, search_space, schedule_cfg=None):
        super(BaseRejector, self).__init__(schedule_cfg)
        self.search_space = search_space

    @abc.abstractmethod
    def accept(self, rollout):
        """
        Returns: bool
        if `rollout` is accepted
        """

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass


class CanonicalTableRejector(BaseRejector):
    NAME = "canonical_table"

    def __init__(self, search_space):
        super(CanonicalTableRejector, self).__init__(search_space)
        assert hasattr(self.search_space, "canonicalize")
        self.cano_table = {}

    def accept(self, rollout):
        canoed = self.search_space.canonicalize(rollout)
        reject = canoed in self.cano_table and not rollout == self.cano_table[canoed]
        if not reject:
            # the canonicalized representation is already in the table,
            # and the rollout is not the representative rollout
            self.cano_table[canoed] = rollout #copy.deepcopy(rollout)
        return not reject

    def load(self, path):
        self.cano_table = torch.load(path)

    def save(self, path):
        torch.save(self.cano_table, path)


class RejectionSampleController(BaseController):
    NAME = "rejection_sample"

    def __init__(self, search_space, device, rollout_type=None, mode="eval",
                 base_sampler_type="random_sample", base_sampler_cfg=None,
                 rejector_type="canonical_table", rejector_cfg=None,
                 maximum_sample_threshold=10, accept_when_reaching_threshold=False,
                 schedule_cfg=None):
        super(RejectionSampleController, self).__init__(search_space, rollout_type, mode)
        self.device = device
        base_sampler_cfg = base_sampler_cfg or {}
        rejector_cfg = rejector_cfg or {}
        self.base_sampler = BaseController.get_class_(base_sampler_type)(
            self.search_space, device=device, rollout_type=rollout_type, mode=mode, **base_sampler_cfg
        )
        self.rejector = BaseRejector.get_class_(rejector_type)(search_space, **rejector_cfg)
        self.maximum_sample_threshold = maximum_sample_threshold
        self.accept_when_reaching_threshold = accept_when_reaching_threshold
        self.logger.info("Maximum sampling trials: %d", maximum_sample_threshold)

    def set_device(self, device):
        self.device = device
        self.base_sampler.set_device(device)

    def sample(self, n, batch_size):
        rollouts = []
        for i_sample in range(n):
            print("\r{}/{}".format(i_sample, n), end="")
            for num_trials in range(1, self.maximum_sample_threshold + 1):
                rollout = self.base_sampler.sample(1, batch_size)[0]
                if self.rejector.accept(rollout):
                    self.logger.debug("Accept rollout after %d trials", num_trials)
                    rollouts.append(rollout)
                    break
            else:
                self.logger.debug(
                    "Does not find a rollout to accept after %d trials",
                    self.maximum_sample_threshold)
                if self.accept_when_reaching_threshold:
                    rollouts.append(rollout)
        return rollouts

    def step(self, rollouts, optimizer, perf_name):
        self.base_sampler.step(rollouts, optimizer, perf_name)

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        self.base_sampler.summary(rollouts, log=log, log_prefix=log_prefix, step=step)
        # TODO: return rejection rate stats

    def save(self, path):
        self.rejector.save("{}_rejector".format(path))
        self.base_sampler.save("{}_base_sampler".format(path))

    def load(self, path):
        self.rejector.load("{}_rejector".format(path))
        self.base_sampler.load("{}_base_sampler".format(path))

    @classmethod
    def supported_rollout_types(cls):
        return list(BaseRollout.all_classes_().keys())
