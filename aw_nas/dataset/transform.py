import numpy as np
import torch

from aw_nas.dataset.data_augmentation import Preproc


class TrainAugmentation(object):
    def __init__(self, size, mean, std, norm_factor=255., bias=0.):
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

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        img, boxes, labels = self.preproc(img, boxes, labels)
        img /= self.norm_factor
        img += self.bias
        img -= np.array([self.mean]).reshape(-1, 1, 1)
        img /= np.array([self.std]).reshape(-1, 1, 1)
        return img, boxes, labels


class TestTransform(object):
    def __init__(self, size, mean=0.0, std=1.0, norm_factor=255., bias=0.):
        self.mean = mean
        self.std = std
        self.norm_factor = norm_factor
        self.bias = bias
        self.preproc = Preproc(size, -1)

    def __call__(self, image, boxes, labels):
        img, boxes, labels = self.preproc(image, boxes, labels)
        img /= self.norm_factor
        img += self.bias
        img -= np.array([self.mean]).reshape(-1, 1, 1)
        img /= np.array([self.std]).reshape(-1, 1, 1)
        return img, boxes, labels
