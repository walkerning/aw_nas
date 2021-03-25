import torch
from torch import nn

from aw_nas.utils import getLogger as _getLogger
from aw_nas.ops import SeparableConv, ConvModule, Scale, get_op
from aw_nas.weights_manager.wrapper import BaseHead

from .classifiers import SharedClassifier

try:
    from mmcv.cnn import normal_init, bias_init_with_prob
except ImportError as e:
    _getLogger("det_header").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e)
    )

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.1)

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
