import functools

import torch
import torch.nn.functional as F
from torch import nn

from aw_nas import germ
from aw_nas.ops import get_op, MobileNetV2Block
from aw_nas.utils import make_divisible, feature_level_to_stage_index

from typing import List


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


class ShuffleNetV2(germ.SearchableBlock):
    NAME = "shufflev2"

    def __init__(
        self,
        ctx,
        num_classes=10,
        depth_choices=[2, 3, 4],
        strides=[2, 2, 1, 2],
        channels=[24, 116, 232, 232, 464, 1024],
        mult_ratio_choices=(1.0,),
        kernel_sizes=[3,],
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

        self.pooling = nn.MaxPool2d(3, stride=2, padding=1)

        kernel_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("kernel_sizes")
        )
        width_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("mult_ratio_choices")
        )
        depth_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("depth_choices")
        )

        prev_channels = self.channels[0]

        self.cells = nn.ModuleList([])
        self.depth_decs = germ.DecisionDict()
        self.stage_out_channel_decs = germ.DecisionDict()

        divisor_fn = functools.partial(make_divisible, divisor=2)

        for i, stride in enumerate(self.strides):
            stage = nn.ModuleList([])
            cur_channels = (
                germ.Choices(
                    mult_ratio_choices,
                    epoch_callback=width_choices_cb,
                    post_mul_fn=divisor_fn,
                )
                * self.channels[i + 1]
            )

            self.depth_decs[str(i)] = germ.Choices(
                depth_choices, epoch_callback=depth_choices_cb
            )
            for j in range(max(depth_choices)):
                kernel_choice = germ.Choices(
                    self.kernel_sizes, epoch_callback=kernel_choices_cb
                )
                block = germ.SearchableShuffleV2Block(
                    ctx,
                    prev_channels,
                    cur_channels,
                    kernel_choice,
                    stride=stride if j == 0 else 1,
                )
                prev_channels = cur_channels
                stage.append(block)

            self.cells.append(stage)

        self.conv_final = germ.SearchableConvBNBlock(
            ctx, prev_channels, self.channels[-1], 1
        )
        self.act = get_op(activation)()

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
        out = self.pooling(stemed)
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
        out = self.conv_final(out)
        out = self.act(out)
        features.append(out)
        return features

    def forward(self, inputs):
        features = self.extract_features(inputs)
        out = features[-1]
        out = self.conv_final.forward(out)
        out = self.act(out)
        out = F.adaptive_avg_pool2d(out, 1)
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


class ShuffleV2SuperNet(germ.GermSuperNet):
    NAME = "shufflev2"

    def __init__(self, search_space, *args, **kwargs):
        super().__init__(search_space)
        with self.begin_searchable() as ctx:
            self.backbone = ShuffleNetV2(ctx, *args, **kwargs)

    def forward(self, inputs):
        return self.backbone(inputs)

    def extract_features(self, inputs):
        return self.backbone.extract_features(inputs)

    def get_feature_channel_num(self, p_levels):
        level_indexes = feature_level_to_stage_index(self.backbone.strides + [1], 2)
        return [
            make_divisible(
                max(self.backbone.mult_ratio_choices)
                * self.backbone.channels[1 + level_indexes[p]],
                2,
            )
            for p in p_levels
        ]
