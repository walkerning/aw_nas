"""
Stub weights manager. Some evaluator do not require weights manager.
"""

import abc
import contextlib

import six
import torch
from torch import nn

from aw_nas import Component, utils
from aw_nas.weights_manager.base import *

class StubWeightsManager(BaseWeightsManager, nn.Module):

    NAME = "stub"

    def __init__(self, search_space, device, rollout_type='stub'):
        super(StubWeightsManager, self).__init__(search_space, device, rollout_type)
        nn.Module.__init__(self)
        self.search_space = search_space
        self.device = device
        self.rollout_type = rollout_type

    def forward(self, inputs, rollout):
        pass

    def assemble_candidate(self, rollout):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["stub", "nasbench-101", "nasbench-201"]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def save(self, path):
        pass

    def load(self, path):
        pass

    def step(self, gradients, optimizer):
        pass

    def set_device(self, device):
        pass
