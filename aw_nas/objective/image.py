# -*- coding: utf-8 -*-

import torch
from torch import nn

from aw_nas.utils.torch_utils import accuracy, count_parameters
from aw_nas.objective.base import BaseObjective

class ClassificationObjective(BaseObjective):
    NAME = "classification"

    def __init__(self, search_space, label_smooth=None, aggregate_as_list=False,
                 return_param=False, schedule_cfg=None):
        super(ClassificationObjective, self).__init__(search_space, schedule_cfg=schedule_cfg)
        self.label_smooth = label_smooth
        self._criterion = nn.CrossEntropyLoss() if not self.label_smooth \
                          else CrossEntropyLabelSmooth(self.label_smooth)
        self.aggregate_as_list = aggregate_as_list
        self.return_param = return_param # return parameter number

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        if self.return_param:
            return ["acc", "param_size"]
        else:
            return ["acc"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        perfs = [float(accuracy(outputs, targets)[0]) / 100]
        if self.return_param:
            perfs.append(count_parameters(cand_net)[0] / 1.e6) # in million (M)
        return perfs

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        return self._criterion(outputs, targets)

    def aggregate_fn(self, perf_name, is_training=True):
        if self.aggregate_as_list and perf_name == "acc":
            return list
        else:
            return super().aggregate_fn(perf_name, is_training)


class FLOPsRegClassificationObjective(ClassificationObjective):
    NAME = "classification-with-flops-reg"

    SCHEDULABLE_ATTRS = ["reg_lambda"]

    def __init__(self, search_space, label_smooth=None, reg_lambda=0., \
                 flops_budget = 5.e8, flops_penalty_cfg={"a": 0, "b": 1},
                 schedule_cfg=None):
        super(FLOPsRegClassificationObjective, self).__init__(search_space, schedule_cfg=schedule_cfg)
        self.reg_lambda = reg_lambda
        self.flops_budget = float(flops_budget) # FIXME: yaml load 1.e-5 as string
        self.flops_penalty_cfg = flops_penalty_cfg
        self.flops = 0.
        self.flops_reg = 1.0

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        # add flops regularization here
        if add_controller_regularization:
            self.flops = cand_net.total_flops
            if cand_net.total_flops < self.flops_budget and self.flops_penalty_cfg["a"] == 0:
                self.flops_reg = -1.
                return self._criterion(outputs, targets)
            else:
                self.flops_reg = (cand_net.total_flops / self.flops_budget) \
                            **(self.flops_penalty_cfg["a"] if cand_net.total_flops < self.flops_budget\
                               else self.flops_penalty_cfg["b"])
                # self.logger.info("Cur Supernet :{:.9e}, Budget: {:.3e}, FLOPS-reg-rate {:.2}".format(self.flops, self.flops_budget, self.flops_reg))
                return self._criterion(outputs, targets)*self.flops_reg
        else:
            return self._criterion(outputs, targets)

    def on_epoch_start(self, epoch):
        super(FLOPsRegClassificationObjective, self).on_epoch_start(epoch)
        self.logger.info("On Epoch {}, Cur Supernet :{:.3e}, Budget: {:.3e}, FLOPS-reg-rate {:.6}".format(epoch, self.flops, self.flops_budget, self.flops_reg))


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        num_classes = int(inputs.shape[-1])
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
