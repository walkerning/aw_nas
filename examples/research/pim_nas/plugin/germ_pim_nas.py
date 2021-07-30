# pylint: disable=arguments-differ
from collections import abc as collection_abcs

from torch import nn

from aw_nas import germ
from aw_nas.utils.common_utils import _get_channel_mask

class SearchableNAS4RRAMBlock(germ.SearchableBlock):
    NAME = "nas4rram_block"

    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        conv_cfg={},
        norm_cfg={},
    ):
        super().__init__(ctx)
        self.conv = germ.SearchableConv(
            ctx,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=False,
            groups=groups,
            **conv_cfg,
        )
        self.bn = germ.SearchableBN(ctx, out_channels, **norm_cfg)
        # output channel choices
        self.output_choices = germ.ChoiceMax(in_channels, out_channels)
    def forward(self, inputs):
        # get max channel
        c1 = self._get_decision(self.conv.ci_handler.choices, self.ctx.rollout)
        c2 = self._get_decision(self.conv.co_handler.choices, self.ctx.rollout)
        c_output = self._get_decision(self.output_choices, self.ctx.rollout)
        # downsample input channel if stride not 1
        _stride = self._get_decision(self.conv.s_handler.choices, self.ctx.rollout)
        if not (_stride == 1):
            down_sample = nn.functional.interpolate(inputs, scale_factor = 0.5, mode = "area")
        else:
            down_sample = inputs
        # get intermediate output
        inter_outputs = self.bn(self.conv(inputs))
        # complete c1 or inter_outpus
        if c1 < c2:
            pcd = (0, 0, 0, 0, 0, c_output - c1)
            out = inter_outputs + nn.functional.pad(down_sample, pcd, "constant", 0)
        elif c1 == c2:
            out = inter_outputs + inputs
        else:
            pcd = (0, 0, 0, 0, 0, c_output - c2)
            out = nn.functional.pad(inter_outputs, pcd, "constant", 0) + down_sample
        return nn.functional.hardtanh(out)

class SearchableNAS4RRAMGroup(germ.SearchableBlock):
    NAME = "nas4rram_group"

    def __init__(
        self,
        ctx,
        depth,
        in_channels,
        out_channels_list,
        kernel_size_list,
        stride_list,
        groups_list,
        conv_cfg_list,
        norm_cfg_list,
    ):
        super().__init__(ctx)
        self.depth_choices = depth
        # init block list
        max_depth = self.depth_choices.range()[1]
        self.blocks = nn.ModuleList()
        output_choices_list = [in_channels]
        for i in range(max_depth):
            self.blocks.append(
                SearchableNAS4RRAMBlock(
                    ctx,
                    output_choices_list[i],
                    out_channels_list[i],
                    kernel_size_list[i],
                    stride_list[i],
                    groups_list[i],
                    conv_cfg_list[i],
                    norm_cfg_list[i],
                )
            )
            output_choices_list.append(self.blocks[-1].output_choices)
        self.output_choices = germ.SelectNonleafChoices(
            output_choices_list[1:],
            self.depth_choices
        )
    def forward(self, inputs):
        _depth = self._get_decision(self.depth_choices, self.ctx.rollout)
        out = inputs
        for i in range(_depth):
            out = self.blocks[i](out)
        return out
