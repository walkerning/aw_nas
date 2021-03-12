# -*- coding: utf-8 -*-

import abc

import numpy as np
from aw_nas import Component, utils

class BaseObjective(Component):
    REGISTRY = "objective"

    def __init__(self, search_space, schedule_cfg=None):
        super(BaseObjective, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.is_training = True

    @utils.abstractclassmethod
    def supported_data_types(cls):
        pass

    @abc.abstractmethod
    def perf_names(self):
        pass

    @abc.abstractmethod
    def get_perfs(self, inputs, outputs, targets, cand_net):
        pass

    @abc.abstractmethod
    def get_reward(self, inputs, outputs, targets, cand_net):
        pass

    @abc.abstractmethod
    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        pass

    def get_loss_item(self, inputs, outputs, targets, cand_net,
                      add_controller_regularization=True, add_evaluator_regularization=True):
        return self.get_loss(inputs, outputs, targets, cand_net,
                             add_controller_regularization, add_evaluator_regularization).item()

    def aggregate_fn(self, perf_name, is_training=True):
        return lambda perfs: np.mean(perfs) if len(perfs) > 0 else 0.

    def set_mode(self, mode):
        if mode == "train":
            self.is_training = True
        elif mode == "eval":
            self.is_training = False
