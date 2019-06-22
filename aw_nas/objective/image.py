# -*- coding: utf-8 -*-

from torch import nn

from aw_nas.utils.torch_utils import accuracy
from aw_nas.objective.base import BaseObjective

class ClassificationObjective(BaseObjective):
    NAME = "classification"

    def __init__(self, search_space):
        super(ClassificationObjective, self).__init__(search_space)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return ["acc"]

    def get_perfs(self, inputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        return [float(accuracy(inputs, targets)[0]) / 100]

    def get_reward(self, inputs, targets, cand_net):
        return self.get_perfs(inputs, targets, cand_net)[0]

    def get_loss(self, inputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            inputs: logits
            targets: labels
        """
        return nn.CrossEntropyLoss()(inputs, targets)
