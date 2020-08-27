import abc

from torch import nn

from aw_nas import Component

class Losses(Component, nn.Module):
    REGISTRY = "loss"

    def __init__(self, schedule_cfg=None):
        super(Losses, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(self, predictions, targets, indices, normalizer):
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

class Metrics(Component):
    REGISTRY = "det_metrics"

    def __init__(self, schedule_cfg=None):
        super(Metrics, self).__init__(schedule_cfg)

    @abc.abstractmethod
    def __call__(self, boxes):
        pass
