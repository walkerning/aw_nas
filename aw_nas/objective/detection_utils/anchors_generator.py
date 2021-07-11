from math import sqrt
from itertools import product
import numpy as np
import torch

from aw_nas.objective.detection_utils.base import AnchorsGenerator
from aw_nas.utils.box_utils import point_form


__all__ = ["SSDAnchorsGenerator"]


class SSDAnchorsGenerator(AnchorsGenerator):
    NAME = "ssd_anchors_generator"

    def __init__(
        self,
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
        clip=True,
        schedule_cfg=None,
    ):

        super(SSDAnchorsGenerator, self).__init__(schedule_cfg)
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(aspect_ratios)
        self.clip = clip
        self.scales = scales

        self.anchors_boxes = {}

    def __call__(self, feature_maps):
        feature_maps = tuple(feature_maps)
        if feature_maps in self.anchors_boxes:
            return self.anchors_boxes[feature_maps]

        means = []
        steps = [(float(1 / f[0]), float(1 / f[1])) for f in feature_maps]
        offset = [(step[0] * 0.5, step[1] * 0.5) for step in steps]
        for k, f in enumerate(feature_maps):
            mean = []
            for i, j in product(range(int(f[0])), range(int(f[1]))):
                cx = j * steps[k][1] + offset[k][1]
                cy = i * steps[k][0] + offset[k][0]
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
            mean = torch.Tensor(mean).view(-1, 4)
            if self.clip:
                mean.clamp_(max=1, min=0)
            mean = point_form(mean)
            means += [mean]
        self.anchors_boxes[feature_maps] = means
        return means
