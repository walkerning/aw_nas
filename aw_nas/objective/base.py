# -*- coding: utf-8 -*-

import abc

from aw_nas import Component, utils

class BaseObjective(Component):
    REGISTRY = "objective"

    def __init__(self, search_space, schedule_cfg=None):
        super(BaseObjective, self).__init__(schedule_cfg)

        self.search_space = search_space

    @utils.abstractclassmethod
    def supported_data_types(cls):
        pass

    @utils.abstractclassmethod
    def perf_name(cls):
        pass

    @abc.abstractmethod
    def get_perf(self, inputs, targets, cand_net):
        pass

    @abc.abstractmethod
    def get_loss(self, inputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        pass

    def get_loss_item(self, inputs, targets, cand_net,
                      add_controller_regularization=True, add_evaluator_regularization=True):
        return self.get_loss(inputs, targets, cand_net,
                             add_controller_regularization, add_evaluator_regularization).item()

    def get_reward(self, inputs, targets, cand_net):
        return self.get_perf(inputs, targets, cand_net)
