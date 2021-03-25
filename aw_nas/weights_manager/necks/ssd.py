import copy

from torch import nn

from aw_nas.weights_manager import (FlexibleBlock, FlexibleMobileNetV3Block)

from .base import BaseNeck

__all__ = ["SSD"]


class SSD(BaseNeck, FlexibleBlock):
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
        gpus=tuple(),
        schedule_cfg=None,
    ):
        super(SSD, self).__init__(search_space, device, rollout_type, gpus,
                schedule_cfg)
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

    def forward(self, features, rollout=None):
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


