import torch
from torch import nn
from torch.nn import functional as F

from aw_nas import utils
from aw_nas.germ.supernet import GermSuperNet
from aw_nas.germ.decisions import Choices
from aw_nas.germ.searchable_blocks import SearchableSepConv
from aw_nas.germ.germ import finalize_rollout

from aw_nas.weights_manager.necks.base import BaseNeck

try:
    from mmcv.cnn import xavier_init
except ImportError as e:
    utils.getLogger("det_neck").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e)
    )

    def xavier_init(mod, distribution):
        getattr(torch.nn.init, "xavier_{}_".format(distribution))(mod.weight)


class SearchableFPN(BaseNeck, GermSuperNet):
    NAME = "fpn_germ"

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
        gpus=tuple(),
        schedule_cfg=None,
    ):
        super().__init__(search_space, device, rollout_type, gpus, schedule_cfg)
        GermSuperNet.__init__(self, search_space)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pyramid_layers = pyramid_layers
        self.kernel_sizes = kernel_sizes
        self.upsample_cfg = upsample_cfg
        self.activation = activation

        with self.begin_searchable() as ctx:

            self.lateral_convs = nn.ModuleList()
            self.fpn_convs = nn.ModuleList()
            for c in in_channels:
                l_conv = SearchableSepConv(ctx, c, out_channels, 1, 1, activation)

                kernel = Choices(kernel_sizes)
                fpn_conv = SearchableSepConv(
                    ctx, out_channels, out_channels, kernel, 1, activation
                )
                self.lateral_convs.append(l_conv)
                self.fpn_convs.append(fpn_conv)

            extra_levels = pyramid_layers - len(in_channels)

            assert extra_levels >= 0

            if extra_levels >= 1:
                for i in range(extra_levels):
                    kernel = Choices(kernel_sizes)
                    extra_fpn_conv = SearchableSepConv(
                        ctx,
                        out_channels,
                        out_channels,
                        kernel,
                        stride=2,
                        activation=activation,
                    )
                    self.fpn_convs.append(extra_fpn_conv)
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

    def finalize(self, rollout):
        self.ctx.rollout = rollout
        return finalize_rollout(self, rollout)

    def get_feature_channel_num(self):
        return [self.out_channels] * self.pyramid_layers
