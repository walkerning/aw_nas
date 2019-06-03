# -*- coding: utf-8 -*-

import abc

from torch import nn

from aw_nas.base import Component, utils

class FinalTrainer(Component):
    REGISTRY = "final_trainer"

    @utils.abstractclassmethod
    def supported_data_types(cls):
        """Return the supported data types"""

    @abc.abstractmethod
    def train(self):
        """Train the model"""

    @abc.abstractmethod
    def setup(self, load=None, save_every=None, train_dir=None, report_every=50):
        """Setup the scaffold: saving/loading settings."""

    @abc.abstractmethod
    def evaluate_split(self, split):
        """Evaluate on dataset split"""

class FinalModel(Component, nn.Module):
    REGISTRY = "final_model"

    def __init__(self, schedule_cfg):
        super(FinalModel, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

    @utils.abstractclassmethod
    def supported_data_types(cls):
        """Return the supported data types"""
