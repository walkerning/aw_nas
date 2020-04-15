# -*- coding: utf-8 -*-

import yaml

def assemble_profiling_nets_from_file(mixin_search_space, fname, base_cfg_fname):
    with open(fname, "r") as f:
        prof_prims = yaml.load(f)
    with open(base_cfg_fname, "r") as f:
        base_cfg = yaml.load(f)
    return assemble_profiling_nets(mixin_search_space, prof_prims, base_cfg)


def assemble_profiling_nets(mixin_search_space, profiling_primitives, base_cfg_template):
    """
    Args:
        profiling_primitives: (list of dict)
            possible keys: spatial_size, input_ch, output_ch, stride, prim_type, primitive_kwargs
        base_cfg_template: (dict) final configuration template

    Returns:
        list of yaml configs
    """
    genotypes_iter = mixin_search_space.primitives_to_genotypes(profiling_primitives)
    for geno in genotypes_iter:
        base_cfg_template["final_model_cfg"]["genotypes"] = geno
        yield base_cfg_template
