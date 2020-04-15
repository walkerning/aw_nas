# -*- coding: utf-8 -*-
"""
ofa controllers
"""

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import utils, assert_rollout_type
from aw_nas.controller.base import BaseController

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

    def step(self):
        pass

    def summary(self, *args, **kargs):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("ofa")]
