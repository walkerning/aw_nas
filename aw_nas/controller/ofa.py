# -*- coding: utf-8 -*-
"""
Differentiable-relaxation based controllers
"""

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import utils, assert_rollout_type
from aw_nas.controller.base import BaseController

class OFAController(BaseController):
    """
    Using the gumbel softmax reparametrization of categorical distribution.
    The sampled actions (ops/nodes) will be hard/soft vectors rather than discrete indexes.
    """
    NAME = "ofa"

    SCHEDULABLE_ATTRS = [
        "force_uniform"
    ]

    def __init__(self, search_space, device, rollout_type="mnasnet_ofa", force_uniform=True,
                 schedule_cfg=None):
        """
        Args:
            use_prob (bool): If true, use the probability directly instead of relaxed sampling.
                If false, use gumbel sampling. Default: false.
            gumbel_hard (bool): If true, the soft relaxed vector calculated by gumbel softmax
                in the forward pass will be argmax to a one-hot vector. The gradients are straightly
                passed through argmax operation. This will cause discrepancy of the forward and
                backward pass, but allow the samples to be sparse. Also applied to `use_prob==True`.
            gumbel_temperature (float): The temperature of gumbel softmax. As the temperature gets
                smaller, when used with `gumbel_hard==True`, the discrepancy of the forward/backward
                pass gets smaller; When used with `gumbel_hard==False`, the samples become more
                sparse(smaller bias), but the variance of the gradient estimation using samples
                becoming larger. Also applied to `use_prob==True`
        """
        super(OFAController, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)

        self.device = device
        self.search_space = search_space

        # training
        self.force_uniform = force_uniform


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
        return [assert_rollout_type("mnasnet_ofa")]
