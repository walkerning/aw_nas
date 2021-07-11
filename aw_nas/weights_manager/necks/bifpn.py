import copy

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from aw_nas import ops
from aw_nas.ops import FlexibleDepthWiseConv, FlexiblePointLinear, FlexibleBatchNorm2d
from aw_nas.weights_manager.ofa_backbone import FlexibleBlock

from .base import BaseNeck

__all__ = ["BiFPNSepConv", "BiFPN"]


class BiFPNSepConv(FlexibleBlock):
    NAME = "sep_conv"

    """
    The implementation of BiFPN in Tensorflow is slightly different with
    Pytorch version. In order to reuse its weight, we implement
    Conv2dStaticSamePadding and MaxPool2dStaticSamePadding in comments.
    """

    def __init__(self, in_channels, out_channels, norm=True, kernel_sizes=[3]):
        super(BiFPNSepConv, self).__init__()

        self.depthwise_conv = FlexibleDepthWiseConv(
            in_channels, kernel_sizes, stride=1, bias=False
        )
        # self.depthwise_conv = ops.Conv2dStaticSamePadding(in_channels, in_channels, kernel_sizes[0], stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = FlexiblePointLinear(in_channels, out_channels, bias=True)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.reset_mask()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)
        return x

    def set_mask(self, mask):
        pass

    def finalize(self):
        return self


class FlexibleBiFPNBlock(FlexibleBlock):
    NAME = "bifpn_block"

    def __init__(
        self,
        num_channels,
        conv_channels,
        first_time=False,
        epsilon=1e-4,
        attention=True,
        activation="swish",
        schedule_cfg=None,
    ):
        super().__init__(schedule_cfg)
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.first_time = first_time
        self.epsilon = epsilon
        self.attention = attention

        self.conv_up = nn.ModuleDict(
            OrderedDict(
                {
                    str(k): BiFPNSepConv(num_channels, num_channels)
                    for k in range(6, 2, -1)
                }
            )
        )
        self.conv_down = nn.ModuleDict(
            OrderedDict(
                {str(k): BiFPNSepConv(num_channels, num_channels) for k in range(4, 8)}
            )
        )

        self.downsample = nn.ModuleDict(
            OrderedDict(
                {
                    # str(k): ops.MaxPool2dStaticSamePadding((3, 3), (2, 2)) for k in range(4, 8)
                    str(k): nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1))
                    for k in range(4, 8)
                }
            )
        )

        self.swish = ops.get_op(activation)()
        # self.relu_fn = ops.get_op("relu6")

        self.weights_1 = nn.ParameterDict(
            {
                str(k): nn.Parameter(
                    torch.ones(2, dtype=torch.float32), requires_grad=True
                )
                for k in range(3, 7)
            }
        )

        self.weights_2 = nn.ParameterDict(
            {
                str(k): nn.Parameter(
                    torch.ones(3, dtype=torch.float32), requires_grad=True
                )
                for k in range(4, 7)
            }
        )
        self.weights_2["7"] = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True
        )

        if self.first_time:
            self.down_channel = nn.ModuleDict(
                {
                    str(k): nn.Sequential(
                        FlexiblePointLinear(conv_channels[i], num_channels, bias=True),
                        FlexibleBatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                    )
                    for i, k in enumerate(range(3, 6))
                }
            )

            self.down_channel_2 = nn.ModuleDict(
                {
                    str(k): nn.Sequential(
                        FlexiblePointLinear(conv_channels[i], num_channels, bias=True),
                        FlexibleBatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                    )
                    for i, k in enumerate(range(4, 6), 1)
                }
            )

            self.p5_to_p6 = nn.Sequential(
                FlexiblePointLinear(conv_channels[2], num_channels, bias=True),
                FlexibleBatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                # ops.MaxPool2dStaticSamePadding((3, 3), (2, 2)),
                nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1)),
            )

            self.p6_to_p7 = nn.Sequential(
                # ops.MaxPool2dStaticSamePadding((3, 3), (2, 2)),
                nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1)),
            )

        self.reset_mask()

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        if self.attention:
            return self._forward_rollout(features, rollout, self._fast_attention)
        return self._forward_rollout(features, rollout, self._no_attention)

    def _fast_attention(self, weight, *args):
        weight = ops.get_op("relu6")(inplace=False)(weight)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        assert len(weight) == len(args)
        return sum([w * arg for w, arg in zip(weight, args)])

    def _no_attention(self, weight, *args):
        return sum(args)

    def _forward_rollout(self, features, rollout=None, attention_fn=None):
        if attention_fn is None:
            attention_fn = self._fast_attention

        if self.first_time:
            p_in = {}
            p_in[3], p_in[4], p_in[5] = features
            p4, p5 = p_in[4], p_in[5]

            p_in[6] = self.p5_to_p6(p_in[5])
            p_in[7] = self.p6_to_p7(p_in[6])

            for i, down in sorted(self.down_channel.items()):
                i = int(i)
                p_in[i] = down(p_in[i])
        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p_in = {k: f for k, f in enumerate(features, 3)}

        # Connections for P6_0 and P7_0 to P6_1 respectively
        prev = p_in[7]
        up = {}
        for i, conv in sorted(self.conv_up.items(), reverse=True):
            ii = int(i)
            up[ii] = conv(
                self.swish(
                    attention_fn(
                        self.weights_1[i],
                        p_in[ii],
                        F.interpolate(prev, p_in[ii].shape[-2:], mode="nearest"),
                    )
                )
            )
            prev = up[ii]

        if self.first_time:
            p_in[4] = self.down_channel_2["4"](p4)
            p_in[5] = self.down_channel_2["5"](p5)

        outs = {3: prev}
        for i, conv in sorted(self.conv_down.items()):
            ii = int(i)
            if ii in up:
                outs[ii] = conv(
                    self.swish(
                        attention_fn(
                            self.weights_2[i],
                            p_in[ii],
                            up[ii],
                            self.downsample[i](prev),
                        )
                    )
                )
            else:
                outs[ii] = conv(
                    self.swish(
                        attention_fn(
                            self.weights_2[i], p_in[ii], self.downsample[i](prev)
                        )
                    )
                )
            prev = outs[ii]

        return tuple([v for k, v in sorted(outs.items())])

    def set_mask(self, expansions, kernel_sizes):
        assert len(expansions) == len(kernel_sizes) == len(self.blocks)
        for block, exp, kernel in zip(self.blocks, expansions, kernel_sizes):
            block.set_mask(exp, kernel)

    def finalize(self, rollout):
        return self


class BiFPN(BaseNeck, FlexibleBlock):
    NAME = "bifpn"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        in_channels,
        out_channels,
        activation="swish",
        attention=True,
        repeat=3,
        gpus=tuple(),
        schedule_cfg=None,
    ):
        super(BiFPN, self).__init__(
            search_space, device, rollout_type, gpus, schedule_cfg
        )
        FlexibleBlock.__init__(self)

        self.blocks = nn.Sequential(
            *[
                FlexibleBiFPNBlock(
                    out_channels,
                    in_channels,
                    first_time=i == 0,
                    epsilon=1e-4,
                    activation=activation,
                    attention=attention,
                )
                for i in range(repeat)
            ]
        )

        # The implementation of BiFPN in this version has 6 layers.
        self.pyramid_layers = 6
        self.in_channels = in_channels
        self.out_channels

    def forward(self, features, rollout=None):
        for i, block in enumerate(self.blocks):
            sub_rollout = None if rollout is None else rollout.sub_rollouts[i]
            features = block(features, sub_rollout)

    def set_mask(self, *args, **kwargs):
        pass

    def finalize(self, rollout=None):
        if rollout is None:
            return self
        self.blocks = nn.Sequential(
            *[m.finalize(sub_r) for m, sub_r in zip(self.blocks, rollout.sub_rollouts)]
        )
        return self

    def get_feature_channel_num(self):
        return [self.out_channels] * self.pyramid_layers

    @classmethod
    def supported_rollout_types(cls):
        return [None]
