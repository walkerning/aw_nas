# -*- coding: utf-8 -*-

import abc

from torch import nn

from aw_nas.base import Component

class FinalTrainer(Component):
    REGISTRY = "final_trainer"

    @abc.abstractmethod
    def supported_data_types(self):
        """Return the supported data types"""

class FinalModel(Component, nn.Module):
    REGISTRY = "final_model"

    def __init__(self, schedule_cfg):
        super(FinalModel, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

    @abc.abstractmethod
    def supported_data_types(self):
        """Return the supported data types"""
