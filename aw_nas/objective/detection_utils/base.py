import abc
import os

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

    def __init__(self, eval_dir=None, schedule_cfg=None):
        super(Metrics, self).__init__(schedule_cfg)
        if eval_dir is None:
            eval_dir = os.environ["HOME"]
            pid = os.getpid()
            eval_dir = os.path.join(eval_dir, ".det_exp", str(pid))
            os.makedirs(eval_dir, exist_ok=True)
        self.eval_dir = eval_dir

    @abc.abstractmethod
    def __call__(self, boxes):
        pass
