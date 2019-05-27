# -*- coding: utf-8 -*-

from torch import nn

from aw_nas.base import Component, utils

class FinalTrainer(Component):
    REGISTRY = "final_trainer"

    @utils.abstractclassmethod
    def supported_data_types(cls):
        """Return the supported data types"""

class FinalModel(Component, nn.Module):
    REGISTRY = "final_model"

    def __init__(self, schedule_cfg):
        super(FinalModel, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

    @utils.abstractclassmethod
    def supported_data_types(cls):
        """Return the supported data types"""
