from math import sqrt
from itertools import product
import numpy as np
import torch

from aw_nas.objective.detection_utils.base import AnchorsGenerator

__all__ = ["SSDAnchorsGenerator"]


class SSDAnchorsGenerator(AnchorsGenerator):
    NAME = "ssd_anchors_generator"

    def __init__(self,
                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                 feature_maps=[4, 5, 6, 7, 8, 9],
                 scales=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                 clip=True,
                 schedule_cfg=None):

        super(SSDAnchorsGenerator, self).__init__(schedule_cfg)
        self.feature_maps = feature_maps  # [(height, width), ...]
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(aspect_ratios)
        self.clip = clip
        self.scales = scales

        self.anchors_boxes = {}

    def __call__(self, image_shape):
        if image_shape in self.anchors_boxes:
            return self.anchors_boxes[image_shape]

        mean = []
        feature_maps = [image_shape]
        fz = image_shape
        for _ in range(max(self.feature_maps) + 1):
            fz = np.ceil(fz / 2)
            feature_maps += [int(fz)]
        feature_maps = [feature_maps[i] for i in self.feature_maps]
        steps = [1 / f for f in feature_maps]
        offset = [step * 0.5 for step in steps]
        for k, f in enumerate(feature_maps): 
            for i, j in product(range(f), range(f)):
                cx = j * steps[k] + offset[k]
                cy = i * steps[k] + offset[k]
                s_k = self.scales[k]
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * self.scales[k + 1])
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    if isinstance(ar, int):
                        ar_sqrt = sqrt(ar)
                        mean += [cx, cy, s_k * ar_sqrt, s_k / ar_sqrt]
                        mean += [cx, cy, s_k / ar_sqrt, s_k * ar_sqrt]
                    elif isinstance(ar, list):
                        mean += [cx, cy, s_k * ar[0], s_k * ar[1]]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        self.anchors_boxes[image_shape] = output
        return output
