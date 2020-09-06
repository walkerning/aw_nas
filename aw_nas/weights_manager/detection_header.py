
import copy

import torch
from torch import nn

from aw_nas.weights_manager import FlexibleBlock, FlexibleMobileNetV3Block
from aw_nas.final.ssd_model import Classifier, Extras
from aw_nas import Component


class DetectionHeader(Component, nn.Module):
    REGISTRY = "detection_header"

    def __init__(self, schedule_cfg=None):
        super().__init__(schedule_cfg)
        nn.Module.__init__(self)
        self.extras = None
        self.classification_headers = None
        self.regression_headers = None

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        features = self.extras.forward_rollout(features, rollout)
        confidences = self.classification_headers(features)
        locations = self.regression_headers(features)
        return confidences, locations

    def finalize(self, rollout):
        raise NotImplementedError()


class SSDExtras(FlexibleBlock):
    NAME = "ssd_extras"

    def __init__(self, expansions, channels, kernel_sizes, schedule_cfg=None):
        super().__init__(schedule_cfg)
        self.expansions = expansions
        self.channels = channels
        self.kernel_sizes = sorted(kernel_sizes)
        self.max_kernel_size = max(self.kernel_sizes)
        self.blocks = nn.ModuleList([
            FlexibleMobileNetV3Block(
                exp,
                C,
                C_out,
                stride=2,
                kernel_sizes=self.kernel_sizes,
                do_kernel_transform=True,
                affine=True,
                activation="relu",
                use_se=False
            ) for exp, C, C_out in zip(expansions, channels[:-1], channels[1:])
        ])
        self.reset_mask()

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        out = features[-1]
        for i, block in enumerate(self.blocks):
            if rollout:
                out =block.forward_rollout(
                    out, rollout.head_width[i], rollout.head_kernel[i])
            else:
                out = block(out)
            features.append(out)
        return features

    def set_mask(self, expansions, kernel_sizes):
        assert len(expansions) == len(kernel_sizes) == len(self.blocks)
        for block, exp, kernel in zip(self.blocks, expansions, kernel_sizes):
            block.set_mask(exp, kernel)

    def finalize(self):
        extras = Extras(self.expansions, self.channels)
        finalized_blocks = nn.ModuleList([m.finalize() for m in self.blocks])
        extras.blocks = finalized_blocks
        return extras


class SSD(DetectionHeader):
    NAME = "ssd_header"

    def __init__(self, device, num_classes, feature_channels, expansions, channels,
                 kernel_sizes, aspect_ratios, pretrained_path=None, schedule_cfg=None):
        super().__init__(schedule_cfg)
        head_channels = feature_channels + channels
        self.num_classes = num_classes
        self.extras = SSDExtras(expansions, head_channels[1:], kernel_sizes)
        multi_ratios = [len(r) * 2 + 2 for r in aspect_ratios]
        self.regression_headers = Classifier(4, head_channels, multi_ratios)
        self.classification_headers = Classifier(num_classes + 1, head_channels,
                                                 multi_ratios)
        self.device = device
        self.pretrained_path = pretrained_path

    def finalize(self, rollout):
        finalized_model = copy.deepcopy(self)
        finalized_model.extras.set_mask(
            rollout.head_width, rollout.head_kernel)
        finalized_model.extras = finalized_model.extras.finalize()
        return finalized_model
