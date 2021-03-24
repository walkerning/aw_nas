import torch
from torch import nn

from aw_nas.utils import getLogger as _getLogger
from aw_nas.ops import FlexibleBatchNorm2d, SeparableConv, ConvModule, Scale, get_op
from aw_nas import Component

try:
    from mmcv.cnn import normal_init, bias_init_with_prob
except ImportError as e:
    _getLogger("det_header").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e)
    )

__all__ = ["DetectionHeader", "AtssHead"]


class DetectionHeader(Component, nn.Module):
    REGISTRY = "detection_header"

    def __init__(self, schedule_cfg=None):
        super().__init__(schedule_cfg)
        nn.Module.__init__(self)

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        raise NotImplementedError()

    def finalize(self, rollout):
        raise NotImplementedError()


class SharedClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        feat_channels,
        stack_layers,
        pyramid_layers,
        share_bn=True,
        activation="swish",
        use_separable_conv=True,
    ):
        super(SharedClassifier, self).__init__()
        self.stack_layers = stack_layers
        self.pyramid_layers = pyramid_layers
        self.share_bn = share_bn

        if use_separable_conv:
            self.conv_list = nn.ModuleList(
                [
                    SeparableConv(
                        in_channels if i == 0 else feat_channels,
                        feat_channels,
                        activation=activation,
                        norm=share_bn,
                        final_activation="relu",
                    )
                    for i in range(stack_layers)
                ]
            )
        else:
            self.conv_list = nn.ModuleList(
                [
                    ConvModule(
                        in_channels if i == 0 else feat_channels,
                        feat_channels,
                        activation=activation,
                        norm=share_bn,
                        final_activation="relu",
                    )
                    for i in range(stack_layers)
                ]
            )

        if not self.share_bn:
            self.bn_list = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.BatchNorm2d(feat_channels, momentum=0.01, eps=1e-3)
                            for i in range(stack_layers)
                        ]
                    )
                    for j in range(pyramid_layers)
                ]
            )
        # self.head = SeparableConv(in_channels, num_anchors * num_classes, norm=False)
        self.act = get_op(activation)()

    def forward(self, inputs):
        feats = []
        for i, feat in enumerate(inputs):
            for j, conv in enumerate(self.conv_list):
                feat = conv(feat)
                if not self.share_bn:
                    feat = self.bn_list[i][j](feat)
                    feat = self.act(feat)
            # feat = self.head(feat)
            feats.append(feat)
        return feats


class Classifier(nn.Module):
    def __init__(self, num_classes, channels, ratios):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.convs = nn.ModuleList(
            [
                SeparableConv2d(
                    in_channels,
                    out_channels=ratio * num_classes,
                    kernel_size=3,
                    padding=1,
                )
                for in_channels, ratio in zip(channels, ratios)
            ]
        )

    def forward(self, features):
        return [conv(ft) for ft, conv in zip(features, self.convs)]


class AnchorHead(DetectionHeader):
    NAME = "anchor_based"

    def __init__(
        self,
        device,
        num_classes,
        head_channels,
        aspect_ratios,
        num_layers=1,
        has_background=True,
        use_separable_conv=True,
        pretrained_path=None,
        schedule_cfg=None,
    ):
        super().__init__(schedule_cfg)
        self.num_classes = num_classes
        multi_ratios = [len(r) * 2 + 2 for r in aspect_ratios]
        self.regression = Classifier(4, head_channels, multi_ratios)
        self.classification = Classifier(
            num_classes + int(has_background), head_channels, multi_ratios
        )
        self.device = device
        self.pretrained_path = pretrained_path

    def forward_rollout(self, features, rollout=None):
        return self.classification(features), self.regression(features)

    def finalize(self, rollout):
        return self


class AtssHead(DetectionHeader):
    NAME = "atss"

    def __init__(
        self,
        device,
        num_classes,
        in_channels,
        feat_channels,
        num_anchors,
        stacked_convs,
        activation="relu",
        use_separable_conv=True,
        has_background=True,
        schedule_cfg=None,
    ):
        super().__init__(schedule_cfg)

        self.stacked_convs = stacked_convs
        self.in_channels = in_channels
        self.has_background = has_background

        assert isinstance(in_channels, (list, tuple)) and len(set(in_channels)) == 1

        pyramid_layers = len(in_channels)
        in_channels = in_channels[0]

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

    def forward_rollout(self, features, rollout=None):
        cls_pred = [self.atss_cls(f) for f in self.classification(features)]
        reg_feats = self.regression(features)
        bbox_pred = [
            scale(self.atss_reg(f)) for scale, f in zip(self.scales, reg_feats)
        ]

        centerness = [self.atss_centerness(f) for f in reg_feats]
        return cls_pred, bbox_pred, centerness

    def finalize(self, rollout):
        return self
