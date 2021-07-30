import functools

import torch
import torch.nn.functional as F
from torch import nn

from aw_nas import germ
from aw_nas.ops import get_op, MobileNetV2Block
from aw_nas.utils import feature_level_to_stage_index

from .utils import divisor_fn


searchable_blocks = {
    "IBN": germ.SearchableMBV2Block,
    "Fused": germ.SearchableFusedConv,
    "Tucker": germ.SearchableTucker
}


class MobileDet(germ.GermSuperNet):
    NAME = "mobiledet"

    def __init__(self, search_space,
                 strides=[1, 2, 2, 2, 1, 2, 1],
                 depth=[1, 4, 4, 4, 4, 4, 1],
                 channels=[32, 32, 64, 96, 192, 192, 320, 384, 384],
                 mult_ratio_choices=[0.25, 0.3125,
                                     0.375, 0.5, 0.625, 0.75, 1.0],
                 kernel_sizes=[3, 5],
                 expansion_choices=[4, 8],
                 block_choices=["IBN", "Fused", "Tucker"],
                 compression_choices=[0.25, 0.75],
                 activation="relu",
                 stem_stride=2,
                 pretrained_path=None,
                 schedule_cfg={}):

        super().__init__(search_space)

        self.strides = strides

        self.kernel_sizes = kernel_sizes
        self.expansion_choices = expansion_choices
        self.channels = channels
        self.mult_ratio_choices = mult_ratio_choices

        for b in block_choices:
            assert b in searchable_blocks, f"expect blocks in [IBN, Fused, Tucker], got {b} instead."

        self.cells = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3,
                      stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            get_op(activation)()
        )

        with self.begin_searchable() as ctx:
            prev_channel = 32
            cur_channel = (germ.Choices(
                mult_ratio_choices) * self.channels[1]).apply(divisor_fn)
            for i, (stride, d) in enumerate(zip(strides, depth)):
                _blocks = nn.ModuleList()
                for j in range(d):
                    init_kwargs = {
                        "in_channels": prev_channel,
                        "out_channels": cur_channel,
                        "exp_ratio": germ.Choices(expansion_choices),
                        "sqz_ratio_1": germ.Choices(compression_choices),
                        "sqz_ratio_2": germ.Choices(compression_choices),
                        "stride": self.strides[i] if j == 0 else 1,
                        "kernel_size": germ.Choices(kernel_sizes),
                        "activation": activation
                    }
                    branch = [getattr(self, f"_{block}_initializer")(ctx, **init_kwargs) for block in
                              block_choices]
                    selector = germ.GermMixedOp(ctx, branch)
                    prev_channel = cur_channel
                    _blocks.append(selector)
                self.cells.append(_blocks)
                if i < len(strides) - 1 and (stride == 2 or self.channels[i + 1] != self.channels[i + 2]):
                    cur_channel = (germ.Choices(
                        mult_ratio_choices) * self.channels[i + 2]).apply(divisor_fn)

    def forward(self, inputs):
        out = self.stem(inputs)
        features = [inputs, out]
        for cell in self.cells:
            for block in cell:
                out = block(out)
            features.append(out)
        return features

    def extract_features(self, inputs):
        return self.forward(inputs)

    def extract_features_rollout(self, rollout, inputs):
        self.ctx.rollout = rollout
        return self.extract_features(inputs)

    def get_feature_channel_num(self, p_levels):
        level_indexes = feature_level_to_stage_index(self.strides, 1)
        return [self.cells[level_indexes[p]][-1].p_ops[0].out_channels for p in p_levels]

    def _IBN_initializer(self, ctx, **kwargs):
        kwargs.pop("sqz_ratio_1", None)
        kwargs.pop("sqz_ratio_2", None)
        return searchable_blocks["IBN"](ctx, **kwargs)

    def _Fused_initializer(self, ctx, **kwargs):
        kwargs.pop("sqz_ratio_1", None)
        kwargs.pop("sqz_ratio_2", None)
        return searchable_blocks["Fused"](ctx, **kwargs)

    def _Tucker_initializer(self, ctx, **kwargs):
        kwargs.pop("exp_ratio", None)
        return searchable_blocks["Tucker"](ctx, **kwargs)
