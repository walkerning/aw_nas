# -*- coding: utf-8 -*-

import yaml

def assemble_profiling_nets_from_file(fname, base_cfg_fname):
    with open(fname, "r") as f:
        prof_prims = yaml.load(f)
    with open(base_cfg_fname, "r") as f:
        base_cfg = yaml.load(f)
    return assemble_profiling_nets(prof_prims, base_cfg)


def assemble_profiling_net(profiling_primitives, base_cfg_template):
    """
    Args:
        profiling_primitives: (list of dict)
            possible keys: spatial_size, input_ch, output_ch, stride, primitive, primitive_kwargs
        base_cfg_template: (dict) final configuration template

    Returns:
        yaml config
    """
    

def assemble_profiling_nets(profiling_primitives, base_cfg_template):
    """
    Args:
        profiling_primitives: (list of dict)
            possible keys: spatial_size, input_ch, output_ch, stride, primitive, primitive_kwargs
        base_cfg_template: (dict) final configuration template

    Returns:
        list of yaml configs
    """
    pass
