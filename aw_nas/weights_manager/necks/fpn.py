import torch
from torch import nn
from torch.nn import functional as F

from aw_nas import ops
from aw_nas.common import assert_rollout_type
from aw_nas.weights_manager.ofa_backbone import FlexibleBlock
from aw_nas.utils import getLogger as _getLogger
from .base import BaseNeck

try:
    from mmcv.cnn import xavier_init
except ImportError as e:
    _getLogger("det_neck").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e)
    )

    def xavier_init(mod, distribution):
        getattr(torch.nn.init, "xavier_{}_".format(distribution))(mod.weight)


__all__ = ["FPN"]  # , "SearchableFPN"]


class FPN(BaseNeck, FlexibleBlock):
    NAME = "fpn"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        in_channels,
        out_channels,
        pyramid_layers,
        kernel_sizes=[3],
        upsample_cfg={"mode": "nearest"},
        activation=None,
        use_separable_conv=True,
        gpus=tuple(),
        schedule_cfg=None,
    ):
        super(FPN, self).__init__(
            search_space, device, rollout_type, gpus, schedule_cfg
        )
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
                l_conv = ops.FlexibleSepConv(
                    c,
                    out_channels,
                    kernel_sizes=[1],
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
                fpn_conv = ops.FlexibleSepConv(
                    out_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
            else:
                l_conv = ops.ConvModule(
                    c,
                    out_channels,
                    kernel_size=1,
                    norm=True,
                    activation=activation,
                    final_activation=activation,
                )
                fpn_conv = ops.ConvModule(
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
                    extra_fpn_conv = ops.FlexibleSepConv(
                        out_channels,
                        out_channels,
                        kernel_sizes=kernel_sizes,
                        stride=2,
                        norm=True,
                        activation=activation,
                        final_activation=activation,
                    )
                else:
                    extra_fpn_conv = ops.ConvModule(
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

    def forward(self, features, rollout=None):
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

    def forward_rollout(self, rollout, features):
        return self.forward(features, rollout)

    def set_mask(self, kernel_sizes):
        pass

    def finalize(self, rollout):
        return self

    def get_feature_channel_num(self):
        return [self.out_channels] * self.pyramid_layers

    @classmethod
    def supported_rollout_types(cls):
        return [None]
