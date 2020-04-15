
import copy
import pickle
from collections import namedtuple
from itertools import count, product

import numpy as np
import pandas as pd

from aw_nas.hardware.base import (BaseHardwareObjectiveModel,
                                  MixinProfilingSearchSpace)

OFAPrim = namedtuple("OFAPrim", ["prim_type", "input_ch", "output_ch", "stride", "spatial_size", "kernel", "stage_idx"])

def ofa_rollout_to_primitive(rollout, strides, channels, primitive_type):
    primitives = []
    for i, (depth, s, c_in, c_out) in enumerate(
        zip(
            rollout.depth, 
            strides, 
            channels[:-1], 
            channels[1:])
        ):
        for j, width, kernel in zip(
            range(d), 
            rollout.width[i], 
            rollout.kernel[i]
        ):
            if i > 0:
                c_in = c_out
                s = 1
            primitives.append(OFAPrim(primitive_type, c_in, c_out, s, width[j], kernel[j], i))
    return primitives

class OFAHardwareObjectiveModel(BaseHardwareObjectiveModel):
    def __init__(self, prof_prims, prof_prims_cfg, schedule_cfg):
        super(OFAHardwareObjectiveModel, self).__init__(schedule_cfg)
        assert "base_channels" in prof_prims_cfg
        assert "mult_ratio" in prof_prims_cfg
        assert "strides" in prof_prims_cfg
        self.default_prim_type = prof_prims_cfg.get("default_prim_type", "default")
        self.performances = prof_prims_cfg.get("performances", ["latency"])

        self.prof_prims = prof_prims
        self.prof_prims_cfg = prof_prims_cfg
        self.channels = [c * prof_prims_cfg["mult_ratio"] for c in prof_prims["base_channels"]]
        self.strides = prof_prims_cfg["strides"]

        self.Perf = namedtuple("Performances", self.performances)

        self._orgnize_table()

        # TODO: @z_c
        self._perf_model = None

    def _orgnize_table(self):
        self._table = {self.default_prim_type: {}}
        for prim in self.prof_prims:
            prim_type = prim["prim_type"]
            if prim_type not in self._table:
                self._table[prim_type] = {}
            sub_table = self._table[prim_type]

            perf = self.Perf(*[prim[f] for f in self.performances])
            if prim_type == self.default_prim_type:
                prim = OFAPrim(*[prim[f] for f in OFAPrim._fields])
            else:
                prim = tuple([prim[k] for k in prim if k not in self.performances])
            sub_table[prim] = perf


    def predict(self, rollout):
        primtives = ofa_rollout_to_primitive(rollout, self.strides, self.channels, self.default_prim_type)
        perfs = []
        for prim in primtives:
            perf = self._table[prim.prim_type].get(prim)
            if perf is None:
                perf = self._perf_model(prim) # TODO: latency model
            perfs.append(perf)
        # assert that each of performances is float
        perfs = np.array(perfs)
        return perfs.sum(axis=0)

    def save(self, path):
        pickle.dump({"table": self._table, "model": self._perf_model.dumps()}, path)

    def load(self, path):
        m = pickle.load(path)
        self._table = m["table"]
        self._perf_model.loads(m["model"])

class OFAMixinProfilingSearchSpace(MixinProfilingSearchSpace):
    NAME = 'ofa_mixin'

    def __init__(self, search_space, base_channels, mult_ratio, strides, primitive_type, fixed_primitives=None, schedule_cfg=None):
        super(OFAMixinProfilingSearchSpace, self).__init__(schedule_cfg=schedule_cfg)
        self.search_space = search_space
        self.base_channels = base_channels
        self.mult_ratio = mult_ratio
        self.strides = strides
        self.primitive_type = primitive_type
        self.fixed_primitives = fixed_primitives

    def traverse_search_space(self, sample=None):
        depths = self.search_space.num_cell_groups
        widths = self.search_space.expansions

        width_choice = self.search_space.width_choice
        kernel_choice = self.search_space.kernel_choice

        # the first stage is excluded since it is fixed as d = 1, w = 1, k = 3
        producted = product(width_choice, kernel_choice, range(1, len(depths)), (0, 1))
        if sample is not None:
            producted = list(producted)
            np.random.shuffle(producted)
            producted = producted[:sample]
        for p in producted:
            yield p

    def generate_profiling_primitives(self, sample=None, as_dict=True):
        channels = [c * self.mult_ratio for c in self.base_channels]
        primitives = []
        for w, k, stage, i in self.traverse_search_space(sample):
            primitives.append(OFAPrim(self.primitive_type, channels[stage + i], channels[stage + 1], 1 if i else self.strides[stage], w, k, stage))
        if self.fixed_primitives is not None:
            primitives += self.fixed_primitives
        primitives = list(set(primitives))
        if as_dict:
            primitives = [p._asdict() for p in primitives]
        return primitives

    def parse_profiling_primitives(self, prof_prims, prof_prims_cfg, hwobjmodel_cfg):
        return OFAHardwareObjectiveModel(prof_prims, prof_prims_cfg, **hwobjmodel_cfg)
    
    def primitives_to_genotypes(self, prof_prims, sample=None):
        """
        prof_prims: list of dict
        """
        prims_df = pd.DataFrame(prof_prims)
        if sample is None:
            sample = np.inf
        ith_arch = 0
        idx = set(range(len(prims_df)))
        while len(idx) > 0 and ith_arch < sample:
            ith_arch += 1
            arch = {"depth": self.search_space.num_cell_groups, "width": [[1],], "kernel": [[3,]]}
            cur_stage = 0
            for stage, depth in enumerate(arch["depth"][1:], 1):
                w = []
                k = []
                stride_layers = prims_df.query(f"stage_idx=={stage} and input_ch != output_ch")
                norm_layers = prims_df.query(f"stage_idx=={stage} and input_ch == output_ch")
                if len(stride_layers) == 0 or len(norm_layers) == 0:
                    raise StopIteration()
                stride_l = stride_layers.sample(1)
                w.append(stride_l["spatial_size"].values[0])
                k.append(stride_l["kernel"].values[0])
                idx.remove(stride_l.index.values[0])
                
                norm_l = norm_layers.sample(depth - 1)
                w.extend(norm_l["spatial_size"].values.tolist())
                k.extend(norm_l["kernel"].values.tolist())
                idx = idx.difference(norm_l.index.values)

                prims_df = prims_df.loc[idx, :]
                arch["width"].append(w)
                arch["kernel"].append(k)
            yield self.search_space.genotype(arch)








