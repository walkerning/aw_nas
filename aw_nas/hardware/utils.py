# -*- coding: utf-8 -*-

import copy
import json
import yaml
import numpy as np
import pandas as pd

from collections import namedtuple

Prim_ = namedtuple("OFAPrim", ["prim_type", "spatial_size", "C", "C_out", "stride", "kernel_size", "kwargs"])

class Prim(Prim_):
    def __new__(_cls, prim_type, spatial_size, C, C_out, stride=1, kernel_size=3, **kwargs):
        kwargs = tuple(sorted([(k, v) for k, v in kwargs.items() if v is not None]))
        return super(Prim, _cls).__new__(_cls, prim_type, int(spatial_size), int(C), int(C_out), int(stride), int(kernel_size), kwargs)

    def _asdict(self):
        origin_dict = dict(super(Prim, self)._asdict())
        kwargs = origin_dict.pop("kwargs")
        origin_dict.update(dict(kwargs))
        return origin_dict

def assemble_profiling_nets_from_file(fname, base_cfg_fname, image_size=224, sample=None, max_layers=20):
    with open(fname, "r") as f:
        prof_prims = yaml.load(f)
    with open(base_cfg_fname, "r") as f:
        base_cfg = yaml.load(f)
    return assemble_profiling_nets(prof_prims, base_cfg, image_size, sample, max_layers)


def assemble_profiling_nets(profiling_primitives, base_cfg_template, image_size=224, sample=None, max_layers=20):
    """
    Args:
        profiling_primitives: (list of dict)
            possible keys: prim_type, spatial_size, C, C_out, stride, primitive_kwargs
            (Don't use dict and list that is unhashable. Use tuple instead: (key, value) )
        base_cfg_template: (dict) final configuration template

    Returns:
        list of yaml configs
    """

    if sample is None:
        sample = np.inf
    
    # genotypes: "[prim_type, *params], [] ..."
    geno_df = pd.DataFrame(sorted(profiling_primitives, key=lambda x: x["C"]))
    ith_arch = 0
    max_channal = int(geno_df.iloc[-1, :]["C_out"])

    glue_layer = lambda spatial_size, C, C_out, stride=1: {"prim_type": "conv_1x1", "spatial_size": spatial_size, "C": C, "C_out": C_out, "stride": stride, "affine": True}

    # use C as the index of df to accelerate query
    geno_df["idx"] = geno_df.index
    geno_df["c_idx"] = geno_df["C"]
    geno_df = geno_df.set_index(["c_idx", "idx"])
    
    while len(geno_df) > 0 and ith_arch < sample:
        ith_arch += 1
        geno = []

        sampled_prim = geno_df.iloc[0, :]
        cur_channel = int(sampled_prim["C"])
        first_cov_op = {"prim_type": "conv_3x3", "spatial_size": image_size, "C": 3, "C_out": cur_channel, "stride": 2, "affine": True}
        cur_size = round(image_size / 2)
        geno.append(first_cov_op)

        for _ in range(max_layers):
            if len(geno_df) == 0:
                break

            # find layer which has consist channel number with the previous
            try:
                sampled_prim = geno_df.loc[(cur_channel, slice(None)), :].iloc[0]
            except KeyError as e:
                available_layer = geno_df.query(f"spatial_size <= {cur_size}")
                if len(available_layer) == 0:
                    break
                
                available_layer = available_layer[available_layer["spatial_size"] == available_layer["spatial_size"].max()].iloc[0]
                out_channel = int(available_layer["C"])
                spatial_size = int(available_layer["spatial_size"])
                stride = int(round(cur_size / spatial_size))
                assert isinstance(stride, int) and stride > 0, f"stride: {stride}" 
                # assert stride in [1, 2], f"spatial size: {spatial_size}, cur_size: {cur_size}"
                geno.append(glue_layer(cur_size, cur_channel, out_channel, stride))
                sampled_prim = available_layer
            
            cur_channel = int(sampled_prim["C_out"])
            cur_size = round(sampled_prim["spatial_size"] / sampled_prim["stride"])
            # sampled_prim = sampled_prim.drop("spatial_size")
            geno_df = geno_df.drop(sampled_prim.name)
            
            geno.append(json.loads(sampled_prim.to_json()))

        base_cfg_template["final_model_cfg"]["genotypes"] = str(geno).replace("'", '"')
        yield copy.deepcopy(base_cfg_template)