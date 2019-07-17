# -*- coding: utf-8 -*-

import numpy as np
from torch import nn

from aw_nas.objective.base import BaseObjective

class LanguageObjective(BaseObjective):
    NAME = "language"

    def __init__(self, search_space, act_reg=0., slowness_reg=0., reward_c=80.):
        super(LanguageObjective, self).__init__(search_space)
        self.act_reg = act_reg
        self.slowness_reg = slowness_reg
        self.reward_c = reward_c

    @classmethod
    def supported_data_types(cls):
        return ["sequence"]

    def perf_names(self):
        return ["perp"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get perplexity.
        """
        return [np.exp(self.get_loss_item(inputs, outputs, targets, cand_net,
                                          add_evaluator_regularization=False))]

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.reward_c / self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs(Tuple): (predict_logits, raw_outs, dropped_outs, hidden)
            targets: target tokens
        """
        logits, raw_outs, outs, _ = outputs
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        if not add_evaluator_regularization:
            return loss

        if self.act_reg > 0: # activation L2 reguarlization on dropped outputs
            loss += self.act_reg * outs.pow(2).mean()
        if self.slowness_reg > 0: # slowness regularization
            loss += self.slowness_reg * (raw_outs[1:] - raw_outs[:-1]).pow(2).mean()
        return loss
