import numpy as np
import torch
from torch import nn

from aw_nas.ops import Scale
from aw_nas.weights_manager.wrapper import BaseHead
from aw_nas.weights_manager.headers.classifiers import SharedClassifier


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probability."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


__all__ = ["AtssHead"]


class AtssHead(BaseHead):
    NAME = "atss"

    def __init__(
        self,
        device,
        feature_channel_nums,
        num_classes,
        in_channels=64,
        pyramid_layers=5,
        num_anchors=4,
        stacked_convs=3,
        activation="relu",
        use_separable_conv=True,
        has_background=True,
        schedule_cfg=None,
    ):
        super(AtssHead, self).__init__(device, feature_channel_nums, schedule_cfg)

        self.stacked_convs = stacked_convs
        self.in_channels = in_channels
        self.has_background = has_background

        assert len(set(feature_channel_nums)) == 1, (
            "Except each layer has the" "same channel number."
        )

        feat_channels = feature_channel_nums[-1]

        self.regression = SharedClassifier(
            in_channels,
            feat_channels,
            stacked_convs,
            pyramid_layers,
            activation=activation,
            use_separable_conv=use_separable_conv,
        )
        self.classification = SharedClassifier(
            in_channels,
            feat_channels,
            stacked_convs,
            pyramid_layers,
            activation=activation,
            use_separable_conv=use_separable_conv,
        )

        self.atss_cls = nn.Conv2d(
            feat_channels,
            num_anchors * (num_classes + int(has_background)),
            3,
            padding=1,
        )
        self.atss_reg = nn.Conv2d(feat_channels, num_anchors * 4, 3, padding=1)
        self.atss_centerness = nn.Conv2d(feat_channels, num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(pyramid_layers)])

        self.init_weights()

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            matched = self.load_state_dict(torch.load(pretrained, "cpu"), strict=False)
            self.logger.info(matched)
            return

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                normal_init(mod, std=0.1)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)

    def forward(self, features, rollout=None):
        cls_pred = [self.atss_cls(f) for f in self.classification(features)]
        reg_feats = self.regression(features)
        bbox_pred = [
            scale(self.atss_reg(f)) for scale, f in zip(self.scales, reg_feats)
        ]

        centerness = [self.atss_centerness(f) for f in reg_feats]
        return cls_pred, bbox_pred, centerness
