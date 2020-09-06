# -*- coding: utf-8 -*-
from itertools import product
from functools import reduce

import numpy as np

from aw_nas.hardware.base import BaseHardwareObjectiveModel, MixinProfilingSearchSpace
from aw_nas.hardware.utils import Prim

from aw_nas.utils import logger as _logger
from aw_nas.utils import make_divisible
from aw_nas.rollout.ofa import MNasNetOFASearchSpace

logger = _logger.getChild("ofa_obj")


class OFAMixinProfilingSearchSpace(MNasNetOFASearchSpace,
                                   MixinProfilingSearchSpace):
    NAME = "ofa_mixin"

    def __init__(
        self,
        width_choice,
        depth_choice,
        kernel_choice,
        image_size_choice,
        num_cell_groups,
        expansions,
        fixed_primitives=None,
        schedule_cfg=None,
    ):
        super(OFAMixinProfilingSearchSpace, self).__init__(
            width_choice,
            depth_choice,
            kernel_choice,
            image_size_choice,
            num_cell_groups,
            expansions,
            schedule_cfg=schedule_cfg,
        )
        MixinProfilingSearchSpace.__init__(self, schedule_cfg=schedule_cfg)

        self.fixed_primitives = fixed_primitives

    def _traverse_search_space(self, sample=None):
        depths = self.num_cell_groups
        widths = self.expansions

        width_choice = self.width_choice
        kernel_choice = self.kernel_choice

        # the first stage is excluded since it is fixed as d = 1, w = 1, k = 3
        producted = product(width_choice, kernel_choice, range(1, len(depths)),
                            (0, 1))
        if sample is not None:
            producted = list(producted)
            np.random.shuffle(producted)
            producted = producted[:sample]
        for p in producted:
            yield p

    def generate_profiling_primitives(
        self,
        base_channels,
        mult_ratio,
        strides,
        acts=None,
        use_ses=None,
        primitive_type="mobilenet_v2_block",
        spatial_size=224,
        stem_stride=2,
        stem_type="conv_3x3",
        sample=None,
        as_dict=True,
    ):
        channels = [make_divisible(c * mult_ratio, 8) for c in base_channels]
        primitives = []
        acts = acts or [
            None,
        ] * len(strides)
        use_ses = use_ses or [
            None,
        ] * len(strides)
        sizes = [
            round(spatial_size / stem_stride /
                  reduce(lambda p, q: p * q, strides[:i]))
            for i in range(1,
                           len(strides) + 1)
        ]

        stem_prim = Prim(stem_type, spatial_size, 3, channels[0], stem_stride,
                         True)
        first_fixed_prim = Prim(primitive_type,
                                spatial_size / stem_stride,
                                channels[0],
                                channels[1],
                                strides[0],
                                True,
                                kernel_size=3,
                                activation=acts[0],
                                use_se=use_ses[0],
                                expansion=1)
        primitives += [stem_prim, first_fixed_prim]
        for w, k, stage, i in self._traverse_search_space(sample):
            primitives.append(
                Prim(
                    primitive_type,
                    sizes[stage - 1],
                    channels[stage + i],
                    channels[stage + 1],
                    1 if i else strides[stage],
                    True,
                    kernel_size=k,
                    activation=acts[stage],
                    use_se=use_ses[stage],
                    expansion=w,
                ))
        if self.fixed_primitives is not None:
            primitives += self.fixed_primitives
        primitives = list(set(primitives))
        if as_dict:
            primitives = [dict(p._asdict()) for p in primitives]
        return primitives

    def parse_profiling_primitives(self, prof_prims_cfg, hwobjmodel_type,
                                   hwobjmodel_cfg):
        return BaseHardwareObjectiveModel.get_class_(hwobjmodel_type)(self, prof_prims_cfg,
                                         **hwobjmodel_cfg)

    @classmethod
    def rollout_to_primitives(cls,
                              rollout,
                              primitive_type,
                              spatial_size,
                              strides,
                              base_channels,
                              mult_ratio=1.,
                              stem_type="conv_3x3",
                              stem_stride=2,
                              **kwargs):
        acts = kwargs.get("acts", [None] * len(rollout.depth))
        use_ses = kwargs.get("use_ses", [None] * len(rollout.depth))
        stem_stride = kwargs.get("stem_stride", 2)
        sizes = [
            round(spatial_size / stem_stride /
                  reduce(lambda p, q: p * q, strides[:i]))
            for i in range(1,
                           len(strides) + 1)
        ]
        channels = [make_divisible(mult_ratio * c, 8) for c in base_channels]
        primitives = [
            Prim(stem_type, spatial_size, 3, channels[0], stem_stride, True)
        ]
        for i, (depth, size, s, c_in, c_out, act, se) in enumerate(
                zip(
                    rollout.depth,
                    sizes,
                    strides,
                    channels[:-1],
                    channels[1:],
                    acts,
                    use_ses,
                )):
            for j, width, kernel in zip(range(depth), rollout.width[i],
                                        rollout.kernel[i]):
                if j > 0:
                    c_in = c_out
                    s = 1
                primitives.append(
                    Prim(
                        primitive_type,
                        size,
                        c_in,
                        c_out,
                        s,
                        True,
                        kernel_size=kernel,
                        activation=act,
                        use_se=se,
                        expansion=width,
                    ))
        return primitives
