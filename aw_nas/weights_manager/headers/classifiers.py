from torch import nn

from aw_nas.weights_manager.necks.bifpn import BiFPNSepConv
from aw_nas.ops import SeparableConv, get_op


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
                SeparableConv(
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


class BiFPNClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        num_layers,
        activation="swish",
        onnx_export=False,
    ):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [
                BiFPNSepConv(in_channels, in_channels, norm=False)
                for i in range(num_layers)
            ]
        )

        self.bn_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FlexibleBatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
                        for i in range(num_layers)
                    ]
                )
                for j in range(5)
            ]
        )
        self.header = BiFPNSepConv(in_channels, num_anchors * num_classes, norm=False)
        self.act = ops.get_op(activation)()

    def forward(self, inputs, post_process=False):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.act(feat)
            feat = self.header(feat)
            feats.append(feat)
        return feats
