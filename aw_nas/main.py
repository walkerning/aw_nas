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
import torch
from torch.backends import cudnn
import yaml

from aw_nas import utils
from aw_nas.utils.vis_utils import WrapWriter
from aw_nas.utils import RegistryMeta
from aw_nas.utils import logger as _logger

LOGGER = _logger.getChild("main")


def _init_component(cfg, registry_name, **addi_args):
    type_ = cfg[registry_name + "_type"]
    cfg = cfg.get(registry_name + "_cfg", None)
    if not cfg:
        cfg = {}
    addi_args.update(cfg)
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
def search(gpu, seed, cfg_file, load, save_every, train_dir, vis_dir):
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
        # backup config file
        train_dir = utils.makedir(train_dir)
        shutil.copyfile(cfg_file, os.path.join(train_dir, "config.yaml"))

        # add log file handler
        log_file = os.path.join(train_dir, "search.log")
        _logger.addFile(log_file)

    LOGGER.info("CWD: %s", os.getcwd())
    LOGGER.info("CMD: %s", " ".join(sys.argv))

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


@main.command(help="Dump the sample configuration.")
@click.argument("out_file", required=True, type=str)
def gen_sample_config(out_file):
    with open(out_file, "w") as out_f:
        for comp_name in ["search_space", "dataset",
                          "controller", "weights_manager", "trainer"]:
            out_f.write(utils.component_sample_config_str(comp_name, prefix="# "))
            out_f.write("\n")


if __name__ == "__main__":
    main()
