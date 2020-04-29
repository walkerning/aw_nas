# -*- coding: utf-8 -*-
"""Base class definition of OFA Backbone."""

import abc

import torch
from torch import nn

from aw_nas import Component
from aw_nas.ops import *
from aw_nas.ops.baseline_ops import MobileNetV2Block, MobileNetV3Block
from aw_nas.utils import make_divisible
from aw_nas.utils.common_utils import _get_channel_mask
from aw_nas.utils.exception import ConfigException, expect



class FlexibleBlock(Component, nn.Module):
    REGISTRY = "ofa_block"

    def __init__(self, schedule_cfg=None):
        super(FlexibleBlock, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

    def reset_mask(self):
        for m in self.modules():
            if isinstance(m, FlexibleLayer):
                m.reset_mask()


class FlexibleMobileNetV2Block(MobileNetV2Block, FlexibleBlock):
    NAME = "mbv2_block"

    def __init__(
        self,
        expansion,
        C,
        C_out,
        stride,
        kernel_sizes=(3, 5, 7),
        do_kernel_transform=True,
        affine=True,
        activation="relu",
        schedule_cfg=None,
    ):
        FlexibleBlock.__init__(self, schedule_cfg)
        self.activation = activation
        act_fn = get_op(activation)
        C_inner = C * expansion
        self.kernel_sizes = sorted(kernel_sizes)
        self.kernel_size = self.kernel_sizes[-1]
        self.do_kernel_transform = do_kernel_transform
        self.affine = affine

        inv_bottleneck = None
        if expansion != 1:
            inv_bottleneck = nn.Sequential(
                FlexiblePointLinear(C, C_inner, 1, 1, 0),
                FlexibleBatchNorm2d(C_inner, affine=affine),
                act_fn(inplace=True),
            )

        depth_wise = nn.Sequential(
            FlexibleDepthWiseConv(
                C_inner,
                self.kernel_sizes,
                stride,
                do_kernel_transform=do_kernel_transform,
            ),
            FlexibleBatchNorm2d(C_inner, affine=affine),
            act_fn(inplace=True),
        )

        point_linear = nn.Sequential(
            FlexiblePointLinear(C_inner, C_out, 1, 1, 0),
            FlexibleBatchNorm2d(C_out, affine=affine),
        )
        super(FlexibleMobileNetV2Block, self).__init__(
            expansion,
            C,
            C_out,
            stride,
            self.kernel_size,
            affine,
            activation,
            inv_bottleneck,
            depth_wise,
            point_linear,
        )

        self.reset_mask()

    def set_mask(self, expansion, kernel_size):
        if expansion != self.expansion:
            filters = self.point_linear[0].weight.data
            mask = _get_channel_mask(filters, self.C * expansion)
            if self.inv_bottleneck:
                self.inv_bottleneck[0].set_mask(None, mask)
                self.inv_bottleneck[1].set_mask(None, mask)

            self.depth_wise[0].set_mask(mask, kernel_size)
            self.depth_wise[1].set_mask(mask, kernel_size)
            self.point_linear[0].set_mask(mask, None)

    def forward_rollout(self, inputs, expansion, kernel_size):
        self.set_mask(expansion, kernel_size)
        out = self.forward(inputs)
        self.reset_mask()
        return out

    def finalize(self):
        inv_bottleneck = None
        if self.inv_bottleneck:
            inv_bottleneck = nn.Sequential(
                *[
                    m.finalize() if isinstance(m, FlexibleLayer) else m
                    for m in self.inv_bottleneck
                ]
            )

        depth_wise = nn.Sequential(
            *[
                m.finalize() if isinstance(m, FlexibleLayer) else m
                for m in self.depth_wise
            ]
        )
        point_linear = nn.Sequential(
            *[
                m.finalize() if isinstance(m, FlexibleLayer) else m
                for m in self.point_linear
            ]
        )
        return MobileNetV2Block(
            self.expansion,
            self.C,
            self.C_out,
            self.stride,
            self.kernel_size,
            self.affine,
            self.activation,
            inv_bottleneck,
            depth_wise,
            point_linear,
        )


class FlexibleMobileNetV3Block(MobileNetV3Block, FlexibleBlock):
    NAME = "mbv3_block"
    def __init__(self, 
        expansion, 
        C, 
        C_out, 
        stride, 
        kernel_sizes=(3, 5, 7), 
        do_kernel_transform=True, 
        affine=True,
        activation="relu",
        use_se=False,
        schedule_cfg=None
    ):
        FlexibleBlock.__init__(self, schedule_cfg)
        self.expansion = expansion
        self.activation = activation
        self.C = C
        self.C_out = C_out
        self.C_inner = C * expansion
        self.stride = stride
        self.kernel_sizes = sorted(kernel_sizes)
        self.kernel_size = self.kernel_sizes[-1]
        self.do_kernel_transform = do_kernel_transform
        self.use_se = use_se
        self.affine = affine

        self.act_fn = get_op(activation)

        inv_bottleneck = None
        if expansion != 1:
            inv_bottleneck = nn.Sequential(
                FlexiblePointLinear(C, self.C_inner, 1, 1, 0),
                FlexibleBatchNorm2d(self.C_inner, affine=affine),
                self.act_fn(inplace=True),
            )

        depth_wise = nn.Sequential(
            FlexibleDepthWiseConv(
                self.C_inner,
                self.kernel_sizes,
                stride,
                do_kernel_transform=do_kernel_transform,
            ),
            FlexibleBatchNorm2d(self.C_inner, affine=affine),
            self.act_fn(inplace=True),
        )

        point_linear = nn.Sequential(
            FlexiblePointLinear(self.C_inner, C_out, 1, 1, 0),
            FlexibleBatchNorm2d(C_out, affine=affine),
        )

        se = None
        if self.use_se:
            se = FlexibleSEModule(self.C_inner)

        super(FlexibleMobileNetV3Block, self).__init__(
            expansion,
            C,
            C_out,
            stride,
            self.kernel_size,
            activation,
            use_se,
            inv_bottleneck,
            depth_wise,
            point_linear,
            se,
        )
        self.reset_mask()

    def set_mask(self, expansion, kernel_size):
        if expansion != self.expansion:
            filters = self.point_linear[0].weight.data
            mask = _get_channel_mask(filters, self.C * expansion)
            if self.inv_bottleneck:
                self.inv_bottleneck[0].set_mask(None, mask)
                self.inv_bottleneck[1].set_mask(mask)
            self.depth_wise[0].set_mask(mask, kernel_size)
            self.depth_wise[1].set_mask(mask)
            self.point_linear[0].set_mask(mask, None)
            if self.se:
                self.se.set_mask(mask)

    def forward_rollout(self, inputs, expansion, kernel_size):
        self.set_mask(expansion, kernel_size)
        out = self.forward(inputs)
        self.reset_mask()
        return out

    def finalize(self):
        inv_bottleneck = None
        if self.inv_bottleneck:
            inv_bottleneck = nn.Sequential(
                *[
                    m.finalize() if isinstance(m, FlexibleLayer) else m
                    for m in self.inv_bottleneck
                ]
            )

        depth_wise = nn.Sequential(
            *[
                m.finalize() if isinstance(m, FlexibleLayer) else m
                for m in self.depth_wise
            ]
        )
        point_linear = nn.Sequential(
            *[
                m.finalize() if isinstance(m, FlexibleLayer) else m
                for m in self.point_linear
            ]
        )
        se = None
        if self.se:
            se = self.se.finalize()
        return MobileNetV3Block(
            self.expansion,
            self.C,
            self.C_out,
            self.stride,
            self.kernel_size,
            self.affine,
            self.activation,
            self.use_se,
            inv_bottleneck,
            depth_wise,
            point_linear,
            se,
        )


class BaseBackboneArch(Component, nn.Module):
    REGISTRY = "ofa_backbone"

    def __init__(
        self,
        device,
        blocks=[1, 4, 4, 4, 4, 4],
        strides=[1, 2, 2, 1, 2, 1],
        expansions=[1, 6, 6, 6, 6, 6],
        channels=[16, 24, 40, 80, 96, 192, 320],
        mult_ratio=1.0,
        kernel_sizes=[3, 5, 7],
        do_kernel_transform=True,
        num_classes=10,
        cell_type="mbv2_cell",
        schedule_cfg=None,
    ):
        super(BaseBackboneArch, self).__init__(schedule_cfg)
        nn.Module.__init__(self)
        self.device = device

        self.blocks = blocks
        self.strides = strides
        self.expansions = expansions
        self.channels = channels
        self.mult_ratio = mult_ratio
        self.kernel_sizes = kernel_sizes
        self.do_kernel_transform = do_kernel_transform
        self.num_classes = num_classes

    @abc.abstractmethod
    def make_stage(
        self, C_in, C_out, depth, stride, expansion, kernel_size, mult_ratio=1.0
    ):
        """
        make a serial of blocks as a stage
        """


class MobileNetV2Arch(BaseBackboneArch):
    NAME = "mbv2_backbone"

    def __init__(
        self,
        device,
        blocks=[1, 4, 4, 4, 4, 4, 1],
        strides=[1, 2, 2, 2, 1, 2, 1],
        expansions=[1, 6, 6, 6, 6, 6, 6],
        channels=[32, 16, 24, 32, 64, 96, 160, 320, 1280],
        mult_ratio=1.0,
        kernel_sizes=[3, 5, 7],
        do_kernel_transform=True,
        num_classes=10,
        block_type="mbv2_block",
        schedule_cfg=None,
    ):
        super(MobileNetV2Arch, self).__init__(
            device,
            blocks,
            strides,
            expansions,
            channels,
            mult_ratio,
            kernel_sizes,
            do_kernel_transform,
            num_classes,
            schedule_cfg,
        )
        self.block_initializer = FlexibleBlock.get_class_(block_type)

        self.channels = [make_divisible(c * mult_ratio, 8) for c in channels]
        self.stem = nn.Sequential(
            nn.Conv2d(
                3, self.channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            get_op("relu")(inplace=True),
        )
        expect(
            blocks[0] == expansions[0] == 1,
            "The first conv layer should have single block and no expansion.",
            ValueError,
        )
        self.mult_ratio = mult_ratio

        self.cells = [
            self.make_stage(
                self.channels[0],
                self.channels[1],
                self.blocks[0],
                self.strides[0],
                self.expansions[0],
                [3],
            )
        ]
        for i, depth in enumerate(self.blocks[1:], 1):
            self.cells.append(
                self.make_stage(
                    self.channels[i],
                    self.channels[i + 1],
                    depth,
                    self.strides[i],
                    self.expansions[i],
                    self.kernel_sizes,
                )
            )
        self.cells = nn.ModuleList(self.cells)
        self.conv_head = nn.Sequential(
            FlexiblePointLinear(self.channels[-2], self.channels[-1], 1, 1, 0),
            nn.BatchNorm2d(self.channels[-1]),
        )
        self.classifier = nn.Conv2d(self.channels[-1], num_classes, 1, 1, 0)

        self.to(self.device)

    def make_stage(self, C_in, C_out, block_num, stride, expansion, kernel_sizes):
        cell = []
        for i in range(block_num):
            if i == 0:
                s = stride
            else:
                s = 1
                C_in = C_out
            cell.append(
                self.block_initializer(
                    expansion,
                    C_in,
                    C_out,
                    s,
                    kernel_sizes,
                    self.do_kernel_transform,
                    "relu",
                    affine=True,
                )
            )
        return nn.ModuleList(cell)

    def forward(self, inputs):
        return self.forward_rollout(inputs)

    def forward_rollout(self, inputs, rollout=None):
        out = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            for j, block in enumerate(cell):
                if rollout is None:
                    out = block(out)
                else:
                    if j >= rollout.depth[i]:
                        break
                    out = block.forward_rollout(
                        out, rollout.width[i][j], rollout.kernel[i][j]
                    )

        out = self.conv_head(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return self.classifier(out).flatten(1)

    def finalize(self, blocks, expansions, kernel_sizes):
        cells = []
        for i, cell in enumerate(self.cells):
            cells.append([])
            for j, block in enumerate(cell):
                if j >= blocks[i]:
                    break
                block.set_mask(expansions[i][j], kernel_sizes[i][j])
                cells[-1].append(block.finalize())
            cells[-1] = nn.ModuleList(cells[-1])
        self.cells = nn.ModuleList(cells)
        return self


class MobileNetV3Arch(BaseBackboneArch):
    NAME = "mbv3_backbone"

    def __init__(
        self,
        device,
        blocks=[1, 4, 4, 4, 4, 4],
        strides=[1, 2, 2, 2, 1, 2],
        expansions=[1, 6, 6, 6, 6, 6],
        channels=[16, 16, 24, 40, 80, 112, 160, 960, 1280],
        mult_ratio=1.0,
        kernel_sizes=[3, 5, 7],
        do_kernel_transform=True,
        use_ses=[False, False, True, False, True, True],
        acts=["relu", "relu", "relu", "h_swish", "h_swish", "h_swish"],
        num_classes=10,
        block_type="mbv3_block",
        schedule_cfg=None,
    ):
        super(MobileNetV3Arch, self).__init__(
            device,
            blocks,
            strides,
            expansions,
            channels,
            mult_ratio,
            kernel_sizes,
            do_kernel_transform,
            num_classes,
            schedule_cfg,
        )
        self.block_initializer = FlexibleBlock.get_class_(block_type)
        self.channels = [make_divisible(c * mult_ratio, 8) for c in channels]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, self.channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(channels[0]),
            get_op("h_swish")(inplace=True),
        )
        expect(
            blocks[0] == expansions[0] == 1,
            "The first conv layer should have single block and no expansion.",
            ValueError,
        )

        self.mult_ratio = mult_ratio
        self.use_ses = use_ses
        self.acts = acts

        self.cells = [
            self.make_stage(
                self.channels[0],
                self.channels[1],
                self.blocks[0],
                self.strides[0],
                self.expansions[0],
                [3],
                self.use_ses[0],
                self.acts[0],
            )
        ]
        for i, depth in enumerate(self.blocks[1:], 1):
            self.cells.append(
                self.make_stage(
                    self.channels[i],
                    self.channels[i + 1],
                    depth,
                    self.strides[i],
                    self.expansions[i],
                    self.kernel_sizes,
                    self.use_ses[i],
                    self.acts[i],
                )
            )
        self.cells = nn.ModuleList(self.cells)
        self.conv_head = nn.Sequential(
            nn.Conv2d(self.channels[-3], self.channels[-2], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels[-2]),
            get_op("h_swish")(inplace=True),
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(self.channels[-2], self.channels[-1], 1, 1, 0, bias=False),
            get_op("h_swish")(inplace=True),
        )
        self.classifier = nn.Linear(self.channels[-1], num_classes)

        self.to(self.device)

    def make_stage(
        self, C_in, C_out, block_num, stride, expansion, kernel_sizes, use_se, act
    ):
        cell = []
        for i in range(block_num):
            if i == 0:
                s = stride
            else:
                s = 1
                C_in = C_out
            cell.append(
                self.block_initializer(
                    expansion,
                    C_in,
                    C_out,
                    s,
                    kernel_sizes,
                    self.do_kernel_transform,
                    act,
                    affine=True,
                    use_se=use_se,
                )
            )
        return nn.ModuleList(cell)

    def flatten(self):
        flattened = [self.stem]
        for i, cell in enumerate(self.cells):
            for j, block in enumerate(cell):
                flattened.append(block)
        flattened.append(self.conv_head)
        flattened.append(self.conv_final)
        return nn.ModuleList(flattened)

    def forward(self, inputs):
        return self.forward_rollout(inputs)

    def forward_rollout(self, inputs, rollout=None):
        out = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            for j, block in enumerate(cell):
                if rollout is None:
                    out = block(out)
                else:
                    if j >= rollout.depth[i]:
                        break
                    out = block.forward_rollout(
                        out, rollout.width[i][j], rollout.kernel[i][j]
                    )
        out = self.conv_head(out)
        out = out.mean(3, keepdim=True).mean(2, keepdim=True)
        out = self.conv_final(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

    def finalize(self, blocks, expansions, kernel_sizes):
        cells = []
        for i, cell in enumerate(self.cells):
            cells.append([])
            for j, block in enumerate(cell):
                if j >= blocks[i]:
                    break
                block.set_mask(expansions[i][j], kernel_sizes[i][j])
                cells[-1].append(block.finalize())
            cells[-1] = nn.ModuleList(cells[-1])
        self.cells = nn.ModuleList(cells)
        return self
