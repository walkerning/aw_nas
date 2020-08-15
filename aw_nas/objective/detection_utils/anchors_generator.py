from math import sqrt
from itertools import product

import torch

from aw_nas.objective.detection_utils.base import AnchorsGenerator

__all__ = ["SSDAnchorsGenerator"]

class SSDAnchorsGenerator(AnchorsGenerator):
    NAME = "ssd_anchors_generator"

    def __init__(self,
                 min_dim=300,
                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                 feature_maps=[19, 10, 5, 3, 2, 1],
                 scales=[45, 90, 135, 180, 225, 270, 315],
                 steps=[16, 32, 64, 100, 150, 300],
                 clip=True,
                 schedule_cfg=None):

        super(SSDAnchorsGenerator, self).__init__(schedule_cfg)
        self.min_dim = min_dim  #[height, width]
        self.feature_maps = feature_maps  #[(height, width), ...]
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(aspect_ratios)
        self.clip = clip
        self.scales = [s / min_dim for s in scales]

        if steps:
            self.steps = [step / min_dim for step in steps]
        else:
            self.steps = [(1 / f_h, 1 / f_w) for f_h, f_w in feature_maps]

        self.offset = [step * 0.5 for step in self.steps]

        self.anchors_boxes = {}

    def __call__(self, image_shape):
        if image_shape in self.anchors_boxes:
            return self.anchors_boxes[image_shape]

        mean = []
        # l = 0
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), range(f)):
                cx = j * self.steps[k] + self.offset[k]
                cy = i * self.steps[k] + self.offset[k]
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
