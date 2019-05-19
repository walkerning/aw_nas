# -*- coding: utf-8 -*-
"""
The main entrypoint of aw_nas.
"""

import os
import sys
import random
import shutil

import click
import numpy as np
import setproctitle
import torch
from torch.backends import cudnn
import yaml

from aw_nas import utils
from aw_nas.utils.vis_utils import WrapWriter
from aw_nas.utils import RegistryMeta
from aw_nas.utils import logger as _logger

LOGGER = _logger.getChild("main")

def _onlycopy_py(src, names):
    return [name for name in names if not \
            (name == "VERSION" or name.endswith(".py") or os.path.isdir(os.path.join(src, name)))]

def _init_component(cfg, registry_name, **addi_args):
    type_ = cfg[registry_name + "_type"]
    cfg = cfg.get(registry_name + "_cfg", None)
    if not cfg:
        cfg = {}
    addi_args.update(cfg)
    LOGGER.info("Component [%s] type： %s", registry_name, type_)
    return RegistryMeta.get_class(registry_name, type_)(**addi_args)


def _set_gpu(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        LOGGER.info('GPU device = %d' % gpu)
    else:
        LOGGER.warn('No GPU available, use CPU!!')


@click.group(help="The awnas NAS framework command line interface. "
             "Use `AWNAS_LOG_LEVEL` environment variable to modify the log level.")
def main():
    pass


@main.command(help="Searching for architecture.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--gpu", default=0, type=int,
              help="the gpu to run training on")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--load", default=None, type=str,
              help="the directory to load checkpoint")
@click.option("--save-every", default=None, type=int,
              help="the number of epochs to save checkpoint every")
@click.option("--train-dir", default=None, type=str,
              help="the directory to save checkpoints")
@click.option("--vis-dir", default=None, type=str,
              help="the directory to save tensorboard events. "
              "need `tensorboard` extra, `pip install aw_nas[tensorboard]`")
@click.option("--develop", default=False, type=bool, is_flag=True,
              help="in develop mode, will copy the `aw_nas` source files into train_dir for backup")
def search(cfg_file, gpu, seed, load, save_every, train_dir, vis_dir, develop):
    # check dependency and initialize visualization writer
    if vis_dir:
        try:
            import tensorboardX
        except ImportError:
            LOGGER.error("Error importing module tensorboardX. Will IGNORE the `--vis-dir` option! "
                         "Try installing the dependency manually, or `pip install aw_nas[vis]`")
            _writer = None
        else:
            _writer = tensorboardX.SummaryWriter(log_dir=vis_dir)
    else:
        _writer = None
    writer = WrapWriter(_writer)

    if train_dir:
        # backup config file, and if in `develop` mode, also backup the aw_nas source code
        train_dir = utils.makedir(train_dir)
        shutil.copyfile(cfg_file, os.path.join(train_dir, "config.yaml"))

        if develop:
            import pkg_resources
            src_path = pkg_resources.resource_filename("aw_nas", "")
            backup_code_path = os.path.join(train_dir, "aw_nas")
            if os.path.exists(backup_code_path):
                shutil.rmtree(backup_code_path)
            LOGGER.info("Copy `aw_nas` source code to %s", backup_code_path)
            shutil.copytree(src_path, backup_code_path, ignore=_onlycopy_py)

        # add log file handler
        log_file = os.path.join(train_dir, "search.log")
        _logger.addFile(log_file)

    LOGGER.info("CWD: %s", os.getcwd())
    LOGGER.info("CMD: %s", " ".join(sys.argv))

    setproctitle.setproctitle("awnas-search config: {}; train_dir: {}; vis_dir: {}; cwd: {}"\
                              .format(cfg_file, train_dir, vis_dir, os.getcwd()))

    # set gpu
    _set_gpu(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    if seed is not None:
        LOGGER.info("Setting random seed: %d.", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    # load components config
    LOGGER.info("Loading configuration files.")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # initialize components
    LOGGER.info("Initializing components.")
    search_space = _init_component(cfg, "search_space")
    weights_manager = _init_component(cfg, "weights_manager",
                                      search_space=search_space, device=device)
    controller = _init_component(cfg, "controller",
                                 search_space=search_space, device=device)
    whole_dataset = _init_component(cfg, "dataset")

    # initialize, setup, run trainer
    LOGGER.info("Initializing trainer and starting the search.")
    trainer = _init_component(cfg, "trainer", weights_manager=weights_manager,
                              controller=controller, dataset=whole_dataset)
    trainer.setup(load, save_every, train_dir, writer=writer)
    trainer.train()


@main.command(help="Train an architecture.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--gpus", default="0", type=str,
              help="the gpus to run training on, split by single comma")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--load", default=None, type=str,
              help="the directory to load checkpoint")
@click.option("--save-every", default=None, type=int,
              help="the number of epochs to save checkpoint every")
@click.option("--train-dir", default=None, type=str,
              help="the directory to save checkpoints")
def train(gpus, seed, cfg_file, load, save_every, train_dir):
    import aw_nas.final #pylint: disable=unused-import
    if train_dir:
        # backup config file, and if in `develop` mode, also backup the aw_nas source code
        train_dir = utils.makedir(train_dir)
        shutil.copyfile(cfg_file, os.path.join(train_dir, "train_config.yaml"))

        # add log file handler
        log_file = os.path.join(train_dir, "train.log")
        _logger.addFile(log_file)

    LOGGER.info("CWD: %s", os.getcwd())
    LOGGER.info("CMD: %s", " ".join(sys.argv))

    setproctitle.setproctitle("awnas-train config: {}; train_dir: {}; cwd: {}"\
                              .format(cfg_file, train_dir, os.getcwd()))

    # set gpu
    gpu_list = [int(g) for g in gpus.split(",")]
    if not gpu_list:
        _set_gpu(None)
        device = "cpu"
    else:
        _set_gpu(gpu_list[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set seed
    if seed is not None:
        LOGGER.info("Setting random seed: %d.", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    # load components config
    LOGGER.info("Loading configuration files.")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    # initialize components
    LOGGER.info("Initializing components.")
    search_space = _init_component(cfg, "search_space")
    whole_dataset = _init_component(cfg, "dataset")
    model = _init_component(cfg, "final_model",
                            search_space=search_space,
                            device=device)
    trainer = _init_component(cfg, "final_trainer",
                              dataset=whole_dataset,
                              model=model,
                              device=device,
                              gpus=gpu_list)

    # start training
    LOGGER.info("Start training.")
    trainer.setup(load, save_every, train_dir)
    trainer.train()


@main.command(help="Dump the sample configuration.")
@click.argument("out_file", required=True, type=str)
def gen_sample_config(out_file):
    with open(out_file, "w") as out_f:
        for comp_name in ["search_space", "dataset",
                          "controller", "weights_manager", "trainer"]:
            out_f.write(utils.component_sample_config_str(comp_name, prefix="# "))
            out_f.write("\n")


@main.command(help="Dump the sample configuration for final training.")
@click.argument("out_file", required=True, type=str)
def gen_final_sample_config(out_file):
    import aw_nas.final #pylint: disable=unused-import
    with open(out_file, "w") as out_f:
        for comp_name in ["search_space", "dataset",
                          "final_model", "final_trainer"]:
            out_f.write(utils.component_sample_config_str(comp_name, prefix="# "))
            out_f.write("\n")


if __name__ == "__main__":
    main()
