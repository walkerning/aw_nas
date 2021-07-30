# -*- coding: utf-8 -*-
"""
The main entrypoint of awnas-hw.
"""

from __future__ import print_function

import os
import shutil
import functools

import yaml
import click

import aw_nas
from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.utils.exception import expect
from aw_nas.utils.common_utils import _OrderedCommandGroup
from aw_nas.hardware.utils import assemble_profiling_nets, iterate, sample_networks
from aw_nas.hardware.base import BaseHardwareCompiler, MixinProfilingSearchSpace

# patch click.option to show the default values
click.option = functools.partial(click.option, show_default=True)

LOGGER = utils.getLogger("main_hw")


@click.group(
    cls=_OrderedCommandGroup,
    help="The awnas-hw command line interface. "
    "Use `AWNAS_LOG_LEVEL` environment variable to modify the log level.")
@click.version_option(version=aw_nas.__version__)
def main():
    pass


# ---- generate profiling networks ----
@main.command(
    help="Generate profiling networks for search space. "
    "hwobj_cfg_fil sections: profiling_primitive_cfg, profiling_net_cfg, "
    "[optional] hardware_compilers")
@click.argument("cfg_file", required=True, type=str)
@click.argument("hwobj_cfg_file", required=True, type=str)
@click.option("--result-dir",
              required=True,
              help="Save the results files to RESULT_DIR")
@click.option(
    "-c",
    "--compile-hardware",
    multiple=True,
    type=click.Choice(BaseHardwareCompiler.all_classes_().keys()),
    help=
    ("Call hardware compilers, if you want to specify configs for HardwareCompiler"
     "Use the `hardware_compiler` section in `hwobj_cfg_file`."))
@click.option(
    "-n",
    "--num_sample",
    type=int,
    help=(
        "Random sample networks from search space if this value is specified, "
        "otherwise traverse the whole search space to find all non-duplicated "
        "primitives and then assemble them."))
def genprof(cfg_file, hwobj_cfg_file, result_dir, compile_hardware,
            num_sample):
    with open(cfg_file, "r") as ss_cfg_f:
        ss_cfg = yaml.load(ss_cfg_f)
    with open(hwobj_cfg_file, "r") as hw_cfg_f:
        hw_cfg = yaml.load(hw_cfg_f)

    ss = get_search_space(hw_cfg["mixin_search_space_type"],
                          **ss_cfg["search_space_cfg"],
                          **hw_cfg["mixin_search_space_cfg"])
    expect(isinstance(ss, MixinProfilingSearchSpace),
           "search space must be a subclass of MixinProfilingsearchspace")

    result_dir = utils.makedir(result_dir)
    # copy cfg files
    shutil.copyfile(cfg_file, os.path.join(result_dir, "config.yaml"))
    shutil.copyfile(hwobj_cfg_file,
                    os.path.join(result_dir, "hwobj_config.yaml"))

    # generate profiling primitive list
    assert 'prof_prims_cfg' in hw_cfg, "key prof_prims_cfg must be specified in hardware configuration file."
    hw_obj_cfg = hw_cfg['prof_prims_cfg']
    prof_prims = list(
        ss.generate_profiling_primitives(**hw_obj_cfg))
    prof_prim_fname = os.path.join(result_dir, "prof_prims.yaml")
    with open(prof_prim_fname, "w") as prof_prim_f:
        yaml.dump(prof_prims, prof_prim_f)
    LOGGER.info("Save the list of profiling primitives to %s", prof_prim_fname)

    if num_sample:
        prof_net_cfgs = sample_networks(
            ss,
            base_cfg_template=hw_cfg["profiling_net_cfg"]
            ["base_cfg_template"],
            num_sample=num_sample,
            **hw_obj_cfg)
    else:
        # assemble profiling nets
        # the primitives can actually be mapped to layers in model during the assembling process
        prof_net_cfgs = assemble_profiling_nets(prof_prims,
                                                **hw_cfg["profiling_net_cfg"])
    prof_net_cfgs = list(prof_net_cfgs)
    prof_net_dir = utils.makedir(os.path.join(result_dir, "prof_nets"),
                                 remove=True)
    prof_fnames = []
    for i_net, prof_net_cfg in enumerate(prof_net_cfgs):
        prof_fname = os.path.join(prof_net_dir, "{}.yaml".format(i_net))
        prof_fnames.append(prof_fname)
        with open(prof_fname, "w") as prof_net_f:
            yaml.dump(prof_net_cfg, prof_net_f)
    LOGGER.info("Save the profiling net configs to directory %s", prof_net_dir)

    # optional (hardware specific): call hardware-specific compiling process
    hw_cfgs = hw_cfg.get("hardware_compilers", [])
    if compile_hardware:
        hw_cfgs.extend([{
            "hardware_compiler_type": hw_name,
            'hardware_compiler_cfg': {}
        } for hw_name in compile_hardware])
    if hw_cfgs:
        hw_compile_dir = utils.makedir(os.path.join(result_dir, "hardwares"),
                                       remove=True)
        LOGGER.info("Call hardware compilers: total %d", len(hw_cfgs))
        for i_hw, hw_cfg in enumerate(hw_cfgs):
            hw_name = hw_cfg["hardware_compiler_type"]
            hw_kwargs = hw_cfg.get("hardware_compiler_cfg", {})
            hw_compiler = BaseHardwareCompiler.get_class_(hw_name)(**hw_kwargs)
            LOGGER.info("{}: Constructed hardware compiler {}{}".format(
                i_hw, hw_name, ":{}".format(hw_kwargs) if hw_kwargs else ""))
            hw_res_dir = utils.makedir(
                os.path.join(hw_compile_dir, "{}-{}".format(i_hw, hw_name)))
            for i_net, prof_cfg in enumerate(prof_net_cfgs):
                res_dir = utils.makedir(os.path.join(hw_res_dir, str(i_net)))
                hw_compiler.compile("{}-{}-{}".format(i_hw, hw_name, i_net),
                                    prof_cfg, res_dir)


@main.command(help="Parse raw net hwobj results to primitive latencies")
@click.argument("hwobj_cfg_file", required=True, type=str)
@click.argument("prof_result_dir", required=True, type=str)
@click.argument("prof_prim_file", required=True, type=str)
@click.argument("prim_to_ops_file", required=True, type=str)
@click.option("--hwobj-type", default="latency")
@click.option("--result-dir",
              required=True,
              help="Save the raw hwobj of profiling primitives to RESULT_DIR.")
def parse(hwobj_cfg_file, prof_result_dir, prof_prim_file, prim_to_ops_file,
          hwobj_type, result_dir):
    """
    Hardware specific conversion, parse the latencies of the profiling networks
    into the uniform format.

    -- prof_result_dir:
        -- net1
            -- latency_file_1.txt
            -- latency_file_2.txt
            -- ...
        -- net2
        -- ...

    -- result-dir:
        -- net1
            -- latency_file_1.yaml
            -- latency_file_2.yaml
            -- ...

    The format of latency_file.yaml: 
        overall_latency: FLOAT
        block_sum_latency: FLOAT
        primitives: 
            - performances: 
                latency: FLOAT 
              C: INT
              C_out: INT
              *OTHER_PRIM_ARGS: ...
            - ...
    """

    with open(hwobj_cfg_file, "r") as lat_cfg_f:
        lat_cfg = yaml.load(lat_cfg_f)

    hw_name = lat_cfg["hardware_compiler_type"]
    hw_kwargs = lat_cfg.get("hardware_compiler_cfg", {})
    hw_compiler = BaseHardwareCompiler.get_class_(hw_name)(**hw_kwargs)
    LOGGER.info("Constructed hardware compiler {}{}".format(
        hw_name, ":{}".format(hw_kwargs) if hw_kwargs else ""))

    hw_compiler.parse_file(prof_result_dir, prof_prim_file, prim_to_ops_file, result_dir)
    

@main.command(help="Parse primitive statistics to generate the hwobj model"
              "hwobj_cfg_file sections: profiling_primitive_cfg, hwobj_cfg")
@click.argument("cfg_file", required=True, type=str)
@click.argument("hwobj_cfg_file", required=True, type=str)
@click.argument("prof_prim_dir", required=True, type=str)
@click.option("--result-file",
              required=True,
              help="Save the hwobj model to RESULT_DIR")
def genmodel(cfg_file, hwobj_cfg_file, prof_prim_dir, result_file):
    with open(cfg_file, "r") as ss_cfg_f:
        ss_cfg = yaml.load(ss_cfg_f)
    with open(hwobj_cfg_file, "r") as hw_cfg_f:
        hw_cfg = yaml.load(hw_cfg_f)
    ss = get_search_space(hw_cfg["mixin_search_space_type"],
                          **ss_cfg["search_space_cfg"],
                          **hw_cfg["mixin_search_space_cfg"])
    expect(isinstance(ss, MixinProfilingSearchSpace),
           "search space must be a subclass of MixinProfilingsearchspace")


    hwobj_model = ss.parse_profiling_primitives(
       hw_cfg['hwperfmodel_type'], hw_cfg['hwperfmodel_cfg']
    )

    prof_nets = iterate(prof_prim_dir)
    hwobj_model.train(prof_nets)
    hwobj_model.save(result_file)
    LOGGER.info("Saved the hardware obj model to %s", result_file)


if __name__ == "__main__":
    main()
