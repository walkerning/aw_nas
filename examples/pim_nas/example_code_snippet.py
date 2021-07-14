from aw_nas import germ
from torch import nn
import torch.nn.functional as F
from aw_nas.germ import GermSuperNet
from aw_nas.utils.common_utils import _get_channel_mask, _get_feature_mask


class nas4rram_supernet(GermSuperNet):
    NAME = "nas4rram_snippet"

    def __init__(self, search_space):
        super().__init__(search_space)
        # choice for group1
        group_1_depth = germ.Choices([2, 4, 6, 8, 10])
        group_1_in_channels = 16
        group_1_out_channels_list = [germ.Choices([_ for _ in range(16, 64 + 4, 4)]) for _ in range(10)]
        group_1_kernel_size_list = [3] * 10
        group_1_stride_list = [1] * 10
        group_1_groups_list = [1] * 10
        group_1_conv_cfg_list = [{}] * 10
        group_1_norm_cfg_list = [{}] * 10
        # output fc
        output_place_holder = germ.Choices([_ for _ in range(16, 64 + 4, 4)])
        # construct
        with self.begin_searchable() as ctx:
            # group 0
            self.group_0 = nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            )
            self.bn_0 = nn.BatchNorm2d(16)
            # group 1
            self.group_1 = ctx.SearchableNAS4RRAMGroup(
                depth = group_1_depth,
                in_channels = group_1_in_channels,
                out_channels_list = group_1_out_channels_list,
                kernel_size_list = group_1_kernel_size_list,
                stride_list = group_1_stride_list,
                groups_list = group_1_groups_list,
                conv_cfg_list = group_1_conv_cfg_list,
                norm_cfg_list = group_1_norm_cfg_list,
            )
            # output
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = ctx.SearchableFC(
                in_features = output_place_holder,
                out_features = 10,
            )

    def forward(self, inputs):
        # group 0
        x = F.hardtanh(self.bn_0(self.group_0(inputs)))
        # group 1
        x = self.group_1(x)
        # output
        _group_1_depth = self.group_1._get_decision(
            self.group_1.d_choices,
            self.group_1.ctx.rollout,
        )
        mask_idx = _get_feature_mask(
            self.fc.weight.data,
            x.shape[1],
            1,
        )
        self.ctx.rollout.masks[self.fc.fi_handler.choices.decision_id] = mask_idx
        self.ctx.rollout.arch[self.fc.fi_choices.decision_id] = self.group_1._get_decision(
            self.group_1.blocks[_group_1_depth - 1].conv.ci_choices.decision_id,
            self.group_1.ctx.rollout,
        )
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
