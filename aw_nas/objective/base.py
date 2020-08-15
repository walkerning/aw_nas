# -*- coding: utf-8 -*-

import abc

from torch import nn

from aw_nas import Component, utils

class BaseObjective(Component):
    REGISTRY = "objective"

    def __init__(self, search_space, schedule_cfg=None):
        super(BaseObjective, self).__init__(schedule_cfg)

        self.search_space = search_space

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


class Losses(Component, nn.Module):
    REGISTRY = "loss"

    def __init__(self, schedule_cfg=None):
        super(Losses, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(self, predictions, targets):
        pass


class AnchorsGenerator(Component):
    REGISTRY = "anchors_generator"

    def __init__(self, schedule_cfg=None):
        super(AnchorsGenerator, self).__init__(schedule_cfg)

    @abc.abstractmethod
    def __call__(self, image_shape):
        pass


class Matcher(Component):
    REGISTRY = "matcher"

    def __init__(self, schedule_cfg=None):
        super(Matcher, self).__init__(schedule_cfg)

    @abc.abstractmethod
    def __call__(self, boxes, labels, anchors):
        pass

class PostProcessing(Component):
    REGISTRY = "post_processing"

    def __init__(self, schedule_cfg=None):
        super(PostProcessing, self).__init__(schedule_cfg)

    @abc.abstractmethod
    def __call__(self, confidences, locations, img_shape):
        pass