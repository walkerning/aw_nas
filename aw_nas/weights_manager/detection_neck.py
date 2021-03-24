import copy

import torch
from torch import nn
from torch.nn import functional as F


from aw_nas import ops
from aw_nas.ops import (
    FlexibleSepConv,
    FlexiblePointLinear,
    FlexibleBatchNorm2d,
    ConvModule,
)
from aw_nas.weights_manager import FlexibleBlock, FlexibleMobileNetV3Block
from aw_nas import Component
from aw_nas.utils import getLogger as _getLogger

try:
    from mmcv.cnn import xavier_init
except ImportError as e:
    _getLogger("det_neck").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e)
    )


class DetectionNeck(Component, nn.Module):
    REGISTRY = "detection_neck"

    def __init__(self, search_space, device, schedule_cfg=None):
        super(DetectionNeck, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

        self.search_space = search_space
        self.device = device

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        raise NotImplementedError()

    def finalize(self, rollout):
        raise NotImplementedError()

    def get_feature_channel_num(self):
        raise NotImplementedError()


class SSD(DetectionNeck, FlexibleBlock):
    NAME = "ssd"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        backbone_stage_channels,
        expansions,
        channels,
        kernel_sizes,
        activation="relu",
        schedule_cfg=None,
    ):
        super(SSD, self).__init__(search_space, device, rollout_type)
        FlexibleBlock.__init__(self)

        self.expansions = expansions
        self.channels = backbone_stage_channels + channels
        channels = backbone_stage_channels[-1:] + channels
        self.kernel_sizes = sorted(kernel_sizes)
        self.max_kernel_size = max(self.kernel_sizes)
        self.blocks = nn.ModuleList(
            [
                FlexibleMobileNetV3Block(
                    exp,
                    C,
                    C_out,
                    stride=2,
                    kernel_sizes=self.kernel_sizes,
                    do_kernel_transform=True,
                    affine=True,
                    activation=activation,
                    use_se=False,
                )
                for exp, C, C_out in zip(expansions, channels[:-1], channels[1:])
            ]
        )
        self.reset_mask()

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        out = features[-1]
        for i, block in enumerate(self.blocks):
            if rollout:
                out = block.forward_rollout(
                    out, rollout.head_width[i], rollout.head_kernel[i]
                )
            else:
                out = block(out)
            features.append(out)
        return features

    def set_mask(self, expansions, kernel_sizes):
        assert len(expansions) == len(kernel_sizes) == len(self.blocks)
        for block, exp, kernel in zip(self.blocks, expansions, kernel_sizes):
            block.set_mask(exp, kernel)

    def finalize(self, rollout):
        neck = copy.deepcopy(self)
        finalized_blocks = nn.ModuleList([m.finalize() for m in neck.blocks])
        neck.blocks = finalized_blocks
        return neck

    def get_feature_channel_num(self):
        return self.channels


class FPN(DetectionNeck, FlexibleBlock):
    NAME = "fpn"

    def __init__(
        self,
        search_space,
        device,
        in_channels,
        out_channels,
        pyramid_layers,
        kernel_sizes=[3],
        upsample_cfg={"mode": "nearest"},
        activation=None,
        use_separable_conv=True,
        schedule_cfg=None,
    ):
        super(FPN, self).__init__(search_space, device)
        FlexibleBlock.__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pyramid_layers = pyramid_layers
        self.kernel_sizes = kernel_sizes
        self.upsample_cfg = upsample_cfg
        self.activation = activation

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i, c in enumerate(in_channels):
            if use_separable_conv:
                l_conv = FlexibleSepConv(
                    c,
                    out_channels,
                    kernel_sizes=[1],
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
                # l_conv = nn.Sequential(
                #        FlexiblePointLinear(c, out_channels),
                #        FlexibleBatchNorm2d(out_channels),
                #        ops.get_op(activation)() if activation else nn.Sequential()
                #    )
                fpn_conv = FlexibleSepConv(
                    out_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
            else:
                l_conv = ConvModule(
                    c,
                    out_channels,
                    kernel_size=1,
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=max(kernel_sizes),
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = pyramid_layers - len(in_channels)

        assert extra_levels >= 0

        if extra_levels >= 1:
            for i in range(extra_levels):
                if use_separable_conv:
                    extra_fpn_conv = FlexibleSepConv(
                        out_channels,
                        out_channels,
                        kernel_sizes=kernel_sizes,
                        stride=2,
                        norm=True,
                        activation=activation,
                        final_activation=activation,
                    )
                else:
                    extra_fpn_conv = ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=max(kernel_sizes),
                        stride=2,
                        norm=True,
                        activation=activation,
                        final_activation=activation,
                    )
                self.fpn_convs.append(extra_fpn_conv)

        self.reset_mask()
        self.init_weights()

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            matched = self.load_state_dict(torch.load(pretrained, "cpu"))
            self.logger.info(matched)
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        assert len(features) == len(self.in_channels)
        laterals = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg
            )

        outs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]

        if self.pyramid_layers > len(outs):
            outs.append(self.fpn_convs[len(outs)](outs[-1]))
            for i in range(len(outs), self.pyramid_layers):
                outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    def set_mask(self, kernel_sizes):
        pass

    def finalize(self, rollout):
        return self

    def get_feature_channel_num(self):
        return [self.out_channels] * self.pyramid_layers
