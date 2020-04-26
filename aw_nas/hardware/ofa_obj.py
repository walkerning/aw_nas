import copy
import pickle
from collections import namedtuple
from itertools import count, product
from functools import reduce

import numpy as np
import pandas as pd

from aw_nas.hardware.base import BaseHardwareObjectiveModel, MixinProfilingSearchSpace
from aw_nas.hardware.utils import Prim

from aw_nas.utils.exception import expect
from aw_nas.utils import logger as _logger
from aw_nas.rollout.ofa import MNasNetOFASearchSpace

logger = _logger.getChild("ofa_obj")


def ofa_rollout_to_primitive(
    rollout,
    spatial_size,
    primitive_type,
    strides,
    channels,
    activation=None,
    use_se=None,
):
    primitives = []
    if activation is None:
        activation = [None] * len(rollout.depth)
    if use_se is None:
        use_se = [None] * len(rollout.depth)
    sizes = [
        round(spatial_size / reduce(lambda p, q: p * q, strides[:i]))
        for i in range(1, len(strides) + 1)
    ]
    for i, (depth, size, s, c_in, c_out, act, se) in enumerate(
        zip(
            rollout.depth,
            sizes,
            strides,
            channels[:-1],
            channels[1:],
            activation,
            use_se,
        )
    ):
        for j, width, kernel in zip(range(depth), rollout.width[i], rollout.kernel[i]):
            if i > 0:
                c_in = c_out
                s = 1
            primitives.append(
                Prim(
                    primitive_type,
                    size,
                    c_in,
                    c_out,
                    s,
                    kernel,
                    activation=act,
                    use_se=se,
                    expansion=width,
                )
            )
    return primitives


class OFAHardwareObjectiveModel(BaseHardwareObjectiveModel):
    NAME = "ofa"

    def __init__(
        self,
        prof_prims,
        prof_prims_cfg,
        base_channels,
        mult_ratio,
        strides,
        activation=None,
        use_se=None,
        spatial_size=112,
        schedule_cfg=None,
    ):
        super(OFAHardwareObjectiveModel, self).__init__(schedule_cfg)
        self.default_prim_type = prof_prims_cfg.get(
            "default_prim_type", "mobilenet_v2_block"
        )
        self.performances = prof_prims_cfg.get("performances", ["latency"])

        self.prof_prims = prof_prims
        self.prof_prims_cfg = prof_prims_cfg
        self.channels = [
            c * prof_prims_cfg["mult_ratio"] for c in prof_prims["base_channels"]
        ]
        self.strides = prof_prims_cfg["strides"]
        self.activation = prof_prims_cfg.get("activation")
        self.use_se = prof_prims_cfg.get("use_se")
        self.spatial_size = prof_prims_cfg.get("spatial_size", 112)

        self.Perf = namedtuple("Performances", self.performances)

        self._orgnize_table()

    def _orgnize_table(self):
        # key: a namedtuple. Prim
        # value: a namedtuple. self.Perf
        self._table = {}
        for prim in self.prof_prims:
            perf = self.Perf(
                *[prim.pop(f) if f in prim else 0.0 for f in self.performances]
            )
            prim = Prim(**prim)
            self._table[prim] = perf

    def predict(self, rollout):
        primtives = ofa_rollout_to_primitive(
            rollout,
            self.default_prim_type,
            self.spatial_size,
            self.strides,
            self.channels,
            self.activation,
            self.use_se,
        )
        perfs = []
        for prim in primtives:
            perf = self._table[prim.prim_type].get(prim)
            if perf is None:
                logger.warn(
                    f"primitive {prim} is not found in the table, return default value 0."
                )
                perf = self.Perf(*[0.0 for f in self.performances])
            perfs.append(perf)
        # assert that each of performances is float
        perfs = np.array(perfs)
        return perfs.sum(axis=0)

    def save(self, path):
        pickle.dump({"table": self._table}, path)

    def load(self, path):
        m = pickle.load(path)
        self._table = m["table"]


class OFAMixinProfilingSearchSpace(MNasNetOFASearchSpace, MixinProfilingSearchSpace):
    NAME = "ofa_mixin"

    def __init__(
        self,
        width_choice,
        depth_choice,
        kernel_choice,
        num_cell_groups,
        expansions,
        fixed_primitives=None,
        schedule_cfg=None,
    ):
        super(OFAMixinProfilingSearchSpace, self).__init__(
            width_choice,
            depth_choice,
            kernel_choice,
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
        producted = product(width_choice, kernel_choice, range(1, len(depths)), (0, 1))
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
        activation=None,
        use_se=None,
        primitive_type="mobilenet_v2_block",
        spatial_size=112,
        sample=None,
        as_dict=True,
    ):
        channels = [c * mult_ratio for c in base_channels]
        primitives = []
        activation = activation or [None,] * len(strides)
        use_se = use_se or [None,] * len(strides)
        sizes = [
            round(spatial_size / reduce(lambda p, q: p * q, strides[:i]))
            for i in range(1, len(strides) + 1)
        ]
        for w, k, stage, i in self._traverse_search_space(sample):
            primitives.append(
                Prim(
                    primitive_type,
                    sizes[stage - 1],
                    channels[stage + i],
                    channels[stage + 1],
                    1 if i else strides[stage],
                    k,
                    activation=activation[stage],
                    use_se=use_se[stage],
                    expansion=w,
                )
            )
        if self.fixed_primitives is not None:
            primitives += self.fixed_primitives
        primitives = list(set(primitives))
        if as_dict:
            primitives = [dict(p._asdict()) for p in primitives]
        return primitives

    def parse_profiling_primitives(self, prof_prims, prof_prims_cfg, hwobjmodel_cfg):
        return OFAHardwareObjectiveModel(prof_prims, prof_prims_cfg, **hwobjmodel_cfg)
