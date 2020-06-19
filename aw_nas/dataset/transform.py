import numpy as np
import torch

from aw_nas.dataset.data_augmentation import Preproc

class TrainAugmentation:
    def __init__(self, size, mean, std):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.std = std
        self.preproc = Preproc(size, mean, std, 0.6)

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.preproc(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.preproc = Preproc(size, mean, std, -1)

    def __call__(self, image, boxes, labels):
        return self.preproc(image, boxes, labels)
