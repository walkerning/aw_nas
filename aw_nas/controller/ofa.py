# -*- coding: utf-8 -*-
"""
ofa controllers
"""

from collections import OrderedDict

import numpy as np
import torch

from aw_nas import assert_rollout_type
from aw_nas.controller.base import BaseController
from aw_nas.rollout.ofa import MNasNetOFARollout, genotype_from_str


class OFAController(BaseController):
    NAME = "ofa"


    def __init__(self, search_space, device, rollout_type="ofa",
                 schedule_cfg=None):
        super(OFAController, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)

        self.device = device
        self.search_space = search_space

    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def forward(self, n=1): #pylint: disable=arguments-differ
        return self.sample(n=n)

    def sample(self, n=1, batch_size=1):
        rollouts = []
        for _ in range(n):
            rollouts.append(self.search_space.random_sample())
        return rollouts

    def save(self, path):
        """Save the parameters to disk."""
        pass

    def load(self, path):
        """Load the parameters from disk."""
        pass

    def step(self, rollouts, optimizer, perf_name):
        pass

    def summary(self, *args, **kwargs):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("ofa")]


class HardwareOFAController(OFAController):
    NAME = "hw_ofa"

    def __init__(self, search_space, device, rollout_type="ofa", schedule_cfg=None):
        super().__init__(search_space, device, rollout_type, schedule_cfg)

        self.record = {}

    def set_mode(self, mode):
        assert mode in ["train", "eval"]
        self.mode = mode

    def sample(self, n=1, batch_size=1):
        if self.mode == "eval":  # in aw_nas derive
            rollouts = []
            for genotype, _ in sorted(self.record.items(), key=lambda k_and_v: k_and_v[1][0], reverse=True):
                rollouts.append(self.search_space.rollout_from_genotype(genotype))
                n -= 1
                if n == 0:
                    break

            return rollouts
        elif self.mode == "train":  # in aw_nas search
            return super().sample(n, batch_size)

    def step(self, rollouts, optimizer, perf_name):
        for r in rollouts:
            if r.genotype not in self.record:
                self.record[str(r.genotype)] = (r.get_perf("mAP"), r.get_perf("latency"))

        return 0.0

    def save(self, path):
        torch.save({"record": self.record}, path)
        self.logger.info("Saved %d controller record to %s", len(self.record), path)

    def load(self, path):
        self.record = torch.load(path, "cpu")
        self.logger.info("Loaded %d controller record from %s", len(self.record), path)
