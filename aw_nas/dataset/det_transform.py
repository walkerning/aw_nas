import numpy as np

from aw_nas.dataset.det_augmentation import Preproc
from aw_nas.utils import getLogger

_LOGGER = getLogger("det_transform")

try:
    from mmdet.datasets.pipelines.compose import Compose
except ImportError as e:
    _LOGGER.warn("Cannot import mmdet, detection NAS might not work: {}".format(e))


class TrainAugmentation(object):
    def __init__(self, pipeline, size, mean, std, norm_factor=255.0, bias=0.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.std = std
        self.norm_factor = norm_factor
        self.bias = bias
        self.preproc = Preproc(size, 0.6)
        self.size = size

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        if self.pipeline is not None:
            return {
                k: v
                for k, v in self.pipeline(
                    {
                        "img": img,
                        "gt_bboxes": boxes,
                        "gt_labels": labels,
                        "bbox_fields": ["gt_bboxes"],
                    }
                ).items()
            }

        img, boxes, labels = self.preproc(img, boxes, labels)
        img /= self.norm_factor
        img += self.bias
        img -= np.array([self.mean]).reshape(-1, 1, 1)
        img /= np.array([self.std]).reshape(-1, 1, 1)
        return img, boxes, labels


class TestTransform(object):
    def __init__(self, pipeline, size, mean=0.0, std=1.0, norm_factor=255.0, bias=0.0):
        self.mean = mean
        self.std = std
        self.norm_factor = norm_factor
        self.bias = bias
        self.preproc = Preproc(size, -1)
        self.size = size

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)

    def __call__(self, img, boxes, labels):
        if self.pipeline is not None:
            return {
                k: v[0]
                for k, v in self.pipeline(
                    {
                        "img": img,
                        "gt_bboxes": boxes,
                        "gt_labels": labels,
                        "bbox_fields": ["gt_bboxes"],
                    }
                ).items()
            }

        img, boxes, labels = self.preproc(img, boxes, labels)
        img /= self.norm_factor
        img += self.bias
        img -= np.array([self.mean]).reshape(-1, 1, 1)
        img /= np.array([self.std]).reshape(-1, 1, 1)
        return img, boxes, labels
