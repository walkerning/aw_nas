# -*- coding: utf-8 -*-
"""
The main entrypoint of aw_nas.
"""

import random
import yaml
import click

import numpy as np
import torch
from torch.backends import cudnn

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

@click.group(help="The awnas NAS framework command line interface. "
             "Use `AWNAS_LOG_LEVEL` environment variable to modify the log level.")
@click.option("--gpu", default=0, type=int,
              help="the gpu to run training on")
def main(gpu):
    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        LOGGER.info('GPU device = %d' % gpu)
    else:
        LOGGER.warn('No GPU available, use CPU!!')

@main.command()
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--load", default=None, type=str,
              help="the directory to load checkpoint")
@click.option("--save-every", default=None, type=int,
              help="the number of epochs to save checkpoint every")
@click.option("--train-dir", default=None, type=str,
              help="the directory to save checkpoints")
def search(seed, cfg_file, load, save_every, train_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    with open(cfg_file, "r") as f:
        cfg = yaml.load(f)

    # initialize components
    search_space = _init_component(cfg, "search_space")
    weights_manager = _init_component(cfg, "weights_manager",
                                      search_space=search_space, device=device)
    controller = _init_component(cfg, "controller",
                                 search_space=search_space, device=device)
    whole_dataset = _init_component(cfg, "dataset")

    # initialize, setup, run trainer
    trainer = _init_component(cfg, "trainer", weights_manager=weights_manager,
                              controller=controller, dataset=whole_dataset)
    trainer.setup(load, save_every, train_dir)
    trainer.train()


if __name__ == "__main__":
    main() #pylint: disable=no-value-for-parameter
