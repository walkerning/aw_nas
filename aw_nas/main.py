# -*- coding: utf-8 -*-
"""
The main entrypoint of aw_nas.
"""

import os
import sys
import random
import shutil
import functools

import click
import numpy as np
import setproctitle
import torch
from torch.backends import cudnn
import yaml

from aw_nas.dataset import AVAIL_DATA_TYPES
from aw_nas.common import rollout_from_genotype_str
from aw_nas import utils, BaseRollout
from aw_nas.utils.vis_utils import WrapWriter
from aw_nas.utils import RegistryMeta
from aw_nas.utils import logger as _logger
from aw_nas.utils.exception import expect

# patch click.option to show the default values
click.option = functools.partial(click.option, show_default=True)

LOGGER = _logger.getChild("main")

def _onlycopy_py(src, names):
    return [name for name in names if not \
            (name == "VERSION" or name.endswith(".py") or os.path.isdir(os.path.join(src, name)))]

def _init_component(cfg, registry_name, **addi_args):
    type_ = cfg[registry_name + "_type"]
    cfg = cfg.get(registry_name + "_cfg", None)
    if not cfg:
        cfg = {}
    # config items will override addi_args items
    addi_args.update(cfg)
    LOGGER.info("Component [%s] type： %s", registry_name, type_)
    cls = RegistryMeta.get_class(registry_name, type_)
    if LOGGER.level < 20: # logging is at debug level
        whole_cfg_str = cls.get_current_config_str(cfg)
        LOGGER.debug("%s %s config:\n%s", registry_name, type_,
                     utils.add_text_prefix(whole_cfg_str, "  "))
    return cls(**addi_args)


def _set_gpu(gpu):
    if gpu is None:
        return
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
        vis_dir = utils.makedir(vis_dir, remove=True)
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
        train_dir = utils.makedir(train_dir, remove=True)
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
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

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
    _data_type = whole_dataset.data_type()
    if _data_type == "sequence":
        # get the num_tokens
        num_tokens = whole_dataset.vocab_size
        LOGGER.info("Dataset %s: vocabulary size: %d", whole_dataset.NAME, num_tokens)
        weights_manager = _init_component(cfg, "weights_manager",
                                          search_space=search_space,
                                          device=device,
                                          num_tokens=num_tokens)
    else:
        weights_manager = _init_component(cfg, "weights_manager",
                                          search_space=search_space, device=device)
    # check weights_manager support for data type
    expect(_data_type in weights_manager.supported_data_types())

    controller = _init_component(cfg, "controller",
                                 search_space=search_space, device=device)

    # check type of rollout match
    expect(weights_manager.rollout_type() == controller.rollout_type(),
           ("The type of the rollouts produced/received by the "
            "controller/weights_manager should match! ({} VS. {})")\
           .format(weights_manager.rollout_type(), controller.rollout_type()))

    # initialize, setup, run trainer
    LOGGER.info("Initializing trainer and starting the search.")
    trainer = _init_component(cfg, "trainer", weights_manager=weights_manager,
                              controller=controller, dataset=whole_dataset)

    # check trainer support for data type
    expect(_data_type in trainer.supported_data_types())
    # check type of rollout match
    expect(controller.rollout_type() in trainer.supported_rollout_types(),
           ("The type of the rollouts handled by the controller/weights_manager"
            " is not supported by the trainer! ({} VS. {})")\
           .format(controller.rollout_type(), trainer.supported_rollout_types()))

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
    if train_dir:
        # backup config file, and if in `develop` mode, also backup the aw_nas source code
        train_dir = utils.makedir(train_dir, remove=True)
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
        device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")

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

    _data_type = whole_dataset.data_type()
    if _data_type == "sequence":
        # get the num_tokens
        num_tokens = whole_dataset.vocab_size
        LOGGER.info("Dataset %s: vocabulary size: %d", whole_dataset.NAME, num_tokens)
        model = _init_component(cfg, "final_model",
                                search_space=search_space,
                                device=device,
                                num_tokens=num_tokens)
    else:
        model = _init_component(cfg, "final_model",
                                search_space=search_space,
                                device=device)
    # check model support for data type
    expect(_data_type in model.supported_data_types())

    trainer = _init_component(cfg, "final_trainer",
                              dataset=whole_dataset,
                              model=model,
                              device=device,
                              gpus=gpu_list)
    # check trainer support for data type
    expect(_data_type in trainer.supported_data_types())

    # start training
    LOGGER.info("Start training.")
    trainer.setup(load, save_every, train_dir)
    trainer.train()


@main.command(help="Eval a final-trained model.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--load", required=True, type=str,
              help="the directory to load checkpoint")
@click.option("--split", "-s", multiple=True, type=str,
              help="evaluate on these dataset splits")
@click.option("--gpus", default="0", type=str,
              help="the gpus to run training on, split by single comma")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
def eval(cfg_file, load, split, gpus, seed): #pylint: disable=redefined-builtin
    setproctitle.setproctitle("awnas-eval config: {}; load: {}; cwd: {}"\
                              .format(cfg_file, load, os.getcwd()))

    # set gpu
    gpu_list = [int(g) for g in gpus.split(",")]
    if not gpu_list:
        _set_gpu(None)
        device = "cpu"
    else:
        _set_gpu(gpu_list[0])
        device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")

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
    whole_dataset = _init_component(cfg, "dataset")

    trainer = _init_component(cfg, "final_trainer",
                              dataset=whole_dataset,
                              model=None,
                              device=device,
                              gpus=gpu_list)

    # check trainer support for data type
    _data_type = whole_dataset.data_type()
    expect(_data_type in trainer.supported_data_types())

    # start training
    LOGGER.info("Start eval.")
    trainer.setup(load)
    for split_name in split:
        trainer.evaluate_split(split_name)


def _dump(rollout, dump_mode, of):
    if dump_mode == "list":
        yaml.safe_dump([list(rollout.genotype._asdict().values())], of)
    elif dump_mode == "str":
        yaml.safe_dump([str(rollout.genotype)], of)
    else:
        raise Exception("Unexpected dump_mode: {}".format(dump_mode))

@main.command(help="Sample architectures, by directly pickle loading controller from path")
@click.option("--load", required=True, type=str,
              help="the file to load controller")
@click.option("-o", "--out-file", required=True, type=str,
              help="the file to write the derived genotypes to")
@click.option("-n", default=1, type=int,
              help="number of architectures to derive")
@click.option("--save-plot", default=None, type=str,
              help="If specified, save the plot of the rollouts to this path")
@click.option("--gpu", default=0, type=int,
              help="the gpu to run deriving on")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--dump-mode", default="str", type=click.Choice(["list", "str"]))
def sample(load, out_file, n, save_plot, gpu, seed, dump_mode):
    LOGGER.info("CWD: %s", os.getcwd())
    LOGGER.info("CMD: %s", " ".join(sys.argv))

    setproctitle.setproctitle("awnas-sample load: {}; cwd: {}".format(load, os.getcwd()))

    # set gpu
    _set_gpu(gpu)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    # set seed
    if seed is not None:
        LOGGER.info("Setting random seed: %d.", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    # create the directory for saving plots
    if save_plot is not None:
        save_plot = utils.makedir(save_plot)

    controller_path = os.path.join(load)
    # load the model on cpu
    controller = torch.load(controller_path, map_location=torch.device("cpu"))
    # then set the device
    controller.set_device(device)

    rollouts = controller.sample(n)
    with open(out_file, "w") as of:
        for i, r in enumerate(rollouts):
            if save_plot is not None:
                r.plot_arch(
                    filename=os.path.join(save_plot, str(i)),
                    label="Derive {}".format(i)
                )
            of.write("# ---- Arch {} ----\n".format(i))
            _dump(r, dump_mode, of)
            of.write("\n")

@main.command(help="Eval architecture from file.")
@click.argument("cfg_file", required=True, type=str)
@click.argument("arch_file", required=True, type=str)
@click.option("--load", required=True, type=str,
              help="the directory to load checkpoint")
@click.option("--gpu", default=0, type=int,
              help="the gpu to run training on")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--save-plot", default=None, type=str,
              help="If specified, save the plot of the rollouts to this path")
@click.option("--steps", default=None, type=int,
              help="number of batches to eval for each arch, default to be the whole derive queue.")
def eval_arch(cfg_file, arch_file, load, gpu, seed, save_plot, steps):
    setproctitle.setproctitle("awnas-eval-arch config: {}; arch_file: {}; load: {}; cwd: {}"\
                              .format(cfg_file, arch_file, load, os.getcwd()))

    # set gpu
    _set_gpu(gpu)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

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

    # load genotypes
    LOGGER.info("Loading archs from file: %s", arch_file)
    with open(arch_file, "r") as f:
        genotypes = yaml.safe_load(f)
    assert isinstance(genotypes, (list, tuple))

    # initialize components
    LOGGER.info("Initializing components.")
    search_space = _init_component(cfg, "search_space")
    controller = _init_component(cfg, "controller",
                                 search_space=search_space, device=device)

    # create the directory for saving plots
    if save_plot is not None:
        save_plot = utils.makedir(save_plot)

    whole_dataset = _init_component(cfg, "dataset")
    _data_type = whole_dataset.data_type()
    if _data_type == "sequence":
        # get the num_tokens
        num_tokens = whole_dataset.vocab_size
        LOGGER.info("Dataset %s: vocabulary size: %d", whole_dataset.NAME, num_tokens)
        weights_manager = _init_component(cfg, "weights_manager",
                                          search_space=search_space,
                                          device=device,
                                          num_tokens=num_tokens)
    else:
        weights_manager = _init_component(cfg, "weights_manager",
                                          search_space=search_space, device=device)

    # initialize, setup, run trainer
    trainer = _init_component(cfg, "trainer", weights_manager=weights_manager,
                              controller=controller, dataset=whole_dataset)
    LOGGER.info("Loading from disk...")
    trainer.setup(load=load)

    # evaluate these rollouts using evaluator
    LOGGER.info("Eval...")
    rollouts = [rollout_from_genotype_str(geno, search_space) for geno in genotypes]
    rollouts = trainer.derive(len(rollouts), rollouts=rollouts, steps=steps)
    for i, r in enumerate(rollouts):
        LOGGER.info("Arch %3d: %.3f", i, r.get_perf())

@main.command(help="Derive architectures.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--load", required=True, type=str,
              help="the directory to load checkpoint")
@click.option("-o", "--out-file", required=True, type=str,
              help="the file to write the derived genotypes to")
@click.option("-n", default=1, type=int,
              help="number of architectures to derive")
@click.option("--save-plot", default=None, type=str,
              help="If specified, save the plot of the rollouts to this path")
@click.option("--test", default=False, type=bool, is_flag=True,
              help="If false, only the controller is loaded and use to sample rollouts; "
              "Otherwise, weights_manager/trainer is also loaded and test these rollouts.")
@click.option("--steps", default=None, type=int,
              help="number of batches to eval for each arch, default to be the whole derive queue.")
@click.option("--gpu", default=0, type=int,
              help="the gpu to run deriving on")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--dump-mode", default="str", type=click.Choice(["list", "str"]))
def derive(cfg_file, load, out_file, n, save_plot, test, steps, gpu, seed, dump_mode):
    LOGGER.info("CWD: %s", os.getcwd())
    LOGGER.info("CMD: %s", " ".join(sys.argv))

    setproctitle.setproctitle("awnas-derive config: {}; load: {}; cwd: {}"\
                              .format(cfg_file, load, os.getcwd()))

    # set gpu
    _set_gpu(gpu)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

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
    controller = _init_component(cfg, "controller",
                                 search_space=search_space, device=device)

    # create the directory for saving plots
    if save_plot is not None:
        save_plot = utils.makedir(save_plot)

    if not test:
        controller_path = os.path.join(load, "controller")
        controller.load(controller_path)
        rollouts = controller.sample(n)
        with open(out_file, "w") as of:
            for i, r in enumerate(rollouts):
                if save_plot is not None:
                    r.plot_arch(
                        filename=os.path.join(save_plot, str(i)),
                        label="Derive {}".format(i)
                    )
                of.write("# ---- Arch {} ----\n".format(i))
                _dump(r, dump_mode, of)
                of.write("\n")
    else:
        whole_dataset = _init_component(cfg, "dataset")
        _data_type = whole_dataset.data_type()
        if _data_type == "sequence":
            # get the num_tokens
            num_tokens = whole_dataset.vocab_size
            LOGGER.info("Dataset %s: vocabulary size: %d", whole_dataset.NAME, num_tokens)
            weights_manager = _init_component(cfg, "weights_manager",
                                              search_space=search_space,
                                              device=device,
                                              num_tokens=num_tokens)
        else:
            weights_manager = _init_component(cfg, "weights_manager",
                                              search_space=search_space, device=device)

        # initialize, setup, run trainer
        trainer = _init_component(cfg, "trainer", weights_manager=weights_manager,
                                  controller=controller, dataset=whole_dataset)
        LOGGER.info("Loading from disk...")
        trainer.setup(load=load)
        LOGGER.info("Deriving and testing...")
        rollouts = trainer.derive(n, steps)
        accs = [r.get_perf() for r in rollouts]
        idxes = np.argsort(accs)[::-1]
        with open(out_file, "w") as of:
            for i, idx in enumerate(idxes):
                rollout = rollouts[idx]
                if save_plot is not None:
                    rollout.plot_arch(
                        filename=os.path.join(save_plot, str(i)),
                        label="Derive {}; Reward {:.3f}".format(i, rollout.get_perf())
                    )
                of.write("# ---- Arch {} (Reward {}) ----\n".format(i, rollout.get_perf()))
                _dump(rollout, dump_mode, of)
                of.write("\n")

@main.command(help="Dump the sample configuration.")
@click.argument("out_file", required=True, type=str)
@click.option("-d", "--data-type", default=None, type=click.Choice(AVAIL_DATA_TYPES),
              help="only dump the configs of the components support this data type")
@click.option("-r", "--rollout-type", default=None,
              type=click.Choice(list(BaseRollout.all_classes_().keys())),
              help="only dump the configs of the components support this rollout type")
def gen_sample_config(out_file, data_type, rollout_type):
    with open(out_file, "w") as out_f:
        for comp_name in ["search_space", "dataset",
                          "controller", "weights_manager", "trainer"]:
            filter_funcs = []
            if data_type is not None:
                if comp_name == "dataset":
                    filter_funcs.append(lambda cls: data_type == cls.data_type())
                elif comp_name in {"weights_manager", "trainer"}:
                    filter_funcs.append(lambda cls: data_type in cls.supported_data_types())
            if rollout_type is not None:
                if comp_name in {"weights_manager", "controller"}:
                    filter_funcs.append(lambda cls: rollout_type == cls.rollout_type())
                if comp_name == "trainer":
                    filter_funcs.append(lambda cls: data_type in cls.supported_data_types())

            out_f.write(utils.component_sample_config_str(comp_name, prefix="# ",
                                                          filter_funcs=filter_funcs))
            out_f.write("\n")


@main.command(help="Dump the sample configuration for final training.")
@click.argument("out_file", required=True, type=str)
@click.option("-d", "--data-type", default=None, type=click.Choice(AVAIL_DATA_TYPES),
              help="only dump the configs of the components support this data type")
def gen_final_sample_config(out_file, data_type):
    with open(out_file, "w") as out_f:
        for comp_name in ["search_space", "dataset",
                          "final_model", "final_trainer"]:
            filter_funcs = []
            if data_type is not None:
                if comp_name == "dataset":
                    filter_funcs.append(lambda cls: data_type == cls.data_type())
                elif comp_name in {"final_model", "final_trainer"}:
                    filter_funcs.append(lambda cls: data_type in cls.supported_data_types())

            out_f.write(utils.component_sample_config_str(comp_name, prefix="# ",
                                                          filter_funcs=filter_funcs))
            out_f.write("\n")


if __name__ == "__main__":
    main()
