from aw_nas import germ
from torch import nn
import torch.nn.functional as F
from aw_nas.germ import GermSuperNet


class _tmp_supernet(GermSuperNet):
    NAME = "tmp_code_snippet"

    def __init__(self, search_space):
        super().__init__(search_space)
        channel_choice_1 = germ.Choices([16, 32, 64])
        channel_choice_2 = germ.Choices([16, 32, 64])
        channel_choice_3 = germ.Choices([32, 64, 128])
        channel_choice_4 = germ.Choices([32, 64, 128])
        with self.begin_searchable() as ctx:
            self.block_0 = nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
            )
            self.bn_0 = nn.BatchNorm2d(32)
            self.block_1 = ctx.SearchableConvBNBlock(
                in_channels=32,
                out_channels=channel_choice_1,
                kernel_size=germ.Choices([3, 5, 7]),
            )
            self.block_2 = ctx.SearchableConvBNBlock(
                in_channels=channel_choice_1,
                out_channels=channel_choice_2,
                kernel_size=germ.Choices([3, 5]),
            )

            self.block_3 = germ.SearchableConvBNBlock(
                ctx,
                in_channels=channel_choice_2,
                out_channels=channel_choice_2,
                kernel_size=3,
                groups=channel_choice_2,
            )
            self.block_4 = ctx.SearchableConvBNBlock(
                in_channels=channel_choice_2,
                out_channels=channel_choice_4,
                kernel_size=3,
            )
            self.block_5 = ctx.SearchableConvBNBlock(
                in_channels=channel_choice_4, out_channels=128, kernel_size=3
            )
            self.fc_6 = nn.Linear(128, 10)

    def forward(self, inputs):
        # stage 1
        x = F.relu(self.bn_0(self.block_0(inputs)))
        x = F.relu(self.block_1(x))
        x = F.relu(self.block_2(x))
        # x = self.s_block_1(x)

        # stage 2
        x = F.relu(self.block_3(x))
        x = F.relu(self.block_4(x))
        x = F.relu(self.block_5(x))
        x = self.fc_6(F.avg_pool2d(x, x.shape[-1]).view((x.shape[0], -1)))
        return x
