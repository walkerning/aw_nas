import functools
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from aw_nas import germ
from aw_nas.ops import get_op, MobileNetV2Block
from aw_nas.utils import make_divisible, feature_level_to_stage_index


def schedule_choice_callback(
    choices: germ.Choices, epoch: int, schedule: List[dict]
) -> None:
    """
    Args:
        choices: instances of Choices
        epoch: int
        schedule: list
            [
                {
                    "epoch": int,
                    "choices": list,
                },
                ...
            ]
    """
    if schedule is None:
        return
    for sch in schedule:
        assert "epoch" in sch and "choices" in sch
        if epoch >= sch["epoch"]:
            choices.choices = sch["choices"]
    print(
        "Epoch: {:>4d}, decision id: {}, choices: {}".format(
            epoch, choices.decision_id, choices.choices
        )
    )


class MobileNetV2(germ.SearchableBlock):
    NAME = "mbv2"

    def __init__(
        self,
        ctx,
        num_classes=10,
        depth_choices=[2, 3, 4],
        strides=[2, 2, 2, 1, 2, 1],
        channels=[32, 16, 24, 32, 64, 96, 160, 320, 1280],
        mult_ratio_choices=(1.0,),
        kernel_sizes=[3, 5, 7],
        expansion_choices=[2, 3, 4, 6],
        activation="relu",
        stem_stride=2,
        pretrained_path=None,
        schedule_cfg={},
    ):
        super().__init__(ctx)

        self.num_classes = num_classes
        self.stem_stride = stem_stride
        self.strides = strides

        self.depth_choices = depth_choices
        self.kernel_sizes = kernel_sizes
        self.expansion_choices = expansion_choices
        self.channels = channels
        self.mult_ratio_choices = mult_ratio_choices

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.channels[0],
                kernel_size=3,
                stride=self.stem_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels[0]),
            get_op(activation)(),
        )

        kernel_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("kernel_sizes")
        )
        exp_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("expansion_choices")
        )
        width_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("mult_ratio_choices")
        )
        depth_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("depth_choices")
        )

        prev_channels = make_divisible(
            self.channels[1] * max(self.mult_ratio_choices), 8
        )
        self.first_block = MobileNetV2Block(
            expansion=1,
            C=self.channels[0],
            C_out=prev_channels,
            stride=1,
            kernel_size=3,
            affine=True,
            activation=activation,
        )

        self.cells = nn.ModuleList([])
        self.depth_decs = germ.DecisionDict()
        self.stage_out_channel_decs = germ.DecisionDict()

        divisor_fn = functools.partial(make_divisible, divisor=8)

        for i, stride in enumerate(self.strides):
            stage = nn.ModuleList([])
            cur_channels = (
                germ.Choices(
                    mult_ratio_choices,
                    epoch_callback=width_choices_cb,
                    post_mul_fn=divisor_fn,
                )
                * self.channels[i + 2]
            )
            if stride == 1 and cur_channels.choices == prev_channels.choices:
                # accidently, we found the first block in some stage that has the same channels
                # coincidentlly, then add shortcut to it.
                cur_channels = prev_channels

            if i < len(self.strides) - 1:
                self.depth_decs[str(i)] = germ.Choices(
                    depth_choices, epoch_callback=depth_choices_cb
                )
            for j in range(max(depth_choices)):
                exp_ratio = germ.Choices(
                    expansion_choices,
                    epoch_callback=exp_choices_cb,
                    post_mul_fn=divisor_fn,
                )
                kernel_choice = germ.Choices(
                    self.kernel_sizes, epoch_callback=kernel_choices_cb
                )

                block = germ.SearchableMBV2Block(
                    ctx,
                    prev_channels,
                    cur_channels,
                    exp_ratio,
                    kernel_choice,
                    stride=stride if j == 0 else 1,
                )
                prev_channels = cur_channels
                stage.append(block)

                if i == len(self.strides) - 1:
                    break

            self.cells.append(stage)

        self.conv_final = germ.SearchableConvBNBlock(
            ctx, prev_channels, self.channels[-1], 1
        )

        self.classifier = nn.Conv2d(self.channels[-1], num_classes, 1, 1, 0)

        if pretrained_path:
            state_dict = torch.load(pretrained_path, "cpu")
            if (
                "classifier.weight" in state_dict
                and state_dict["classifier.weight"].shape[0] != self.num_classes
            ):
                del state_dict["classifier.weight"]
                del state_dict["classifier.bias"]
            self.logger.info(self.load_state_dict(state_dict, strict=False))

    def extract_features_rollout(self, rollout, inputs):
        self.ctx.rollout = rollout
        return self.extract_features(inputs)

    def extract_features(self, inputs):
        stemed = self.stem(inputs)
        out = self.first_block(stemed)
        features = [inputs, stemed, out]
        for i, cell in enumerate(self.cells):
            if self.ctx.rollout is not None and str(i) in self.depth_decs:
                depth = self._get_decision(self.depth_decs[str(i)], self.ctx.rollout)
            else:
                depth = len(cell)
            for j, block in enumerate(cell):
                if j >= depth:
                    break
                out = block(out)
            features.append(out)
        return features

    def forward(self, inputs):
        features = self.extract_features(inputs)
        out = features[-1]
        out = F.adaptive_avg_pool2d(out, 1)
        out = self.conv_final.forward(out)
        return self.classifier(out).flatten(1)

    def finalize_rollout(self, rollout):
        with self.finalize_context(rollout):
            cells = nn.ModuleList()
            for i, cell in enumerate(self.cells):
                if str(i) in self.depth_decs:
                    depth = self._get_decision(self.depth_decs[str(i)], rollout)
                else:
                    depth = len(cell)
                cells.append(
                    nn.ModuleList([c.finalize_rollout(rollout) for c in cell[:depth]])
                )
            self.cells = cells
            self.conv_final.finalize_rollout(rollout)
        return self


class MBV2SuperNet(germ.GermSuperNet):
    NAME = "mbv2"

    def __init__(self, search_space, *args, **kwargs):
        super().__init__(search_space)
        with self.begin_searchable() as ctx:
            self.backbone = MobileNetV2(ctx, *args, **kwargs)

    def forward(self, inputs):
        return self.backbone(inputs)

    def extract_features(self, inputs):
        return self.backbone.extract_features(inputs)

    def extract_features_rollout(self, rollout, inputs):
        self.ctx.rollout = rollout
        return self.extract_features(inputs)

    def get_feature_channel_num(self, p_levels):
        level_indexes = feature_level_to_stage_index(self.backbone.strides, 1)
        return [
            self.backbone.cells[level_indexes[p]][-1].out_channels for p in p_levels
        ]
