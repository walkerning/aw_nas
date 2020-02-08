# -*- coding: utf-8 -*-
"""
The main entrypoint of aw_nas.
"""

from __future__ import print_function

import os
import sys
import random
import shutil
import functools
import multiprocessing as mp

import click
import yaml
import numpy as np
import setproctitle
import torch
from torch.backends import cudnn

from aw_nas.dataset import AVAIL_DATA_TYPES
from aw_nas import utils, BaseRollout
from aw_nas.common import rollout_from_genotype_str
from aw_nas.utils.vis_utils import WrapWriter
from aw_nas.utils import RegistryMeta
from aw_nas.utils import logger as _logger
from aw_nas.utils.exception import expect

# patch click.option to show the default values
click.option = functools.partial(click.option, show_default=True)

# subclass `click.Group` to list commands in order
class _OrderedCommandGroup(click.Group):
    def __init__(self, *args, **kwargs):
        self.cmd_names = []
        super(_OrderedCommandGroup, self).__init__(*args, **kwargs)

    def list_commands(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super(_OrderedCommandGroup, self).list_commands(ctx)
        return sorted(commands, key=self.cmd_names.index)

    def command(self, *args, **kwargs):
        def decorator(func):
            cmd = super(_OrderedCommandGroup, self).command(*args, **kwargs)(func)
            self.cmd_names.append(cmd.name)
            return cmd
        return decorator

LOGGER = _logger.getChild("main")

def _onlycopy_py(src, names):
    return [name for name in names if not \
            (name == "VERSION" or name.endswith(".py") or os.path.isdir(os.path.join(src, name)))]

def _init_components_from_cfg(cfg, device, evaluator_only=False, controller_only=False,
                              from_controller=False, search_space=None, controller=None):
    """
    Initialize components using configuration.
    Order:
    `search_space`, `controller`, `dataset`, `weights_manager`, `objective`, `evaluator`, `trainer`
    """
    rollout_type = cfg["rollout_type"]
    if not from_controller:
        search_space = _init_component(cfg, "search_space")
        if not evaluator_only:
            controller = _init_component(cfg, "controller",
                                         search_space=search_space, device=device,
                                         rollout_type=rollout_type)
            if controller_only:
                return search_space, controller
    else: # continue components initialization from controller stage
        assert search_space is not None and controller is not None

    # dataset
    whole_dataset = _init_component(cfg, "dataset")
    _data_type = whole_dataset.data_type()

    # weights manager
    if _data_type == "sequence":
        # get the num_tokens
        num_tokens = whole_dataset.vocab_size
        LOGGER.info("Dataset %s: vocabulary size: %d", whole_dataset.NAME, num_tokens)
        weights_manager = _init_component(cfg, "weights_manager",
                                          search_space=search_space,
                                          device=device,
                                          rollout_type=rollout_type,
                                          num_tokens=num_tokens)
    else:
        weights_manager = _init_component(cfg, "weights_manager",
                                          search_space=search_space, device=device,
                                          rollout_type=rollout_type)
    expect(_data_type in weights_manager.supported_data_types())

    # objective
    objective = _init_component(cfg, "objective", search_space=search_space)

    # evaluator
    evaluator = _init_component(cfg, "evaluator", dataset=whole_dataset,
                                weights_manager=weights_manager, objective=objective,
                                rollout_type=rollout_type)
    expect(_data_type in evaluator.supported_data_types())

    if evaluator_only:
        return search_space, whole_dataset, weights_manager, objective, evaluator

    # trainer
    LOGGER.info("Initializing trainer and starting the search.")
    trainer = _init_component(cfg, "trainer",
                              evaluator=evaluator, controller=controller,
                              rollout_type=rollout_type)

    return search_space, whole_dataset, weights_manager, objective, evaluator, controller, trainer

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
        LOGGER.warning('No GPU available, use CPU!!')


@click.group(cls=_OrderedCommandGroup,
             help="The awnas NAS framework command line interface. "
             "Use `AWNAS_LOG_LEVEL` environment variable to modify the log level.")
@click.option("--local_rank", default=-1, type=int,
              help="the rank of this process")
def main(local_rank):
    if local_rank > -1:
        torch.cuda.set_device(local_rank)

# ---- Search, Sample, Derive using trainer ----
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
@click.option("--interleave-report-every", default=50, type=int,
              help="the number of interleave steps to report every, "
              "only work in interleave training mode")
@click.option("--train-dir", default=None, type=str,
              help="the directory to save checkpoints")
@click.option("--vis-dir", default=None, type=str,
              help="the directory to save tensorboard events. "
              "need `tensorboard` extra, `pip install aw_nas[tensorboard]`")
@click.option("--develop", default=False, type=bool, is_flag=True,
              help="in develop mode, will copy the `aw_nas` source files into train_dir for backup")
def search(cfg_file, gpu, seed, load, save_every, interleave_report_every,
           train_dir, vis_dir, develop):
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
    trainer = _init_components_from_cfg(cfg, device)[-1]

    # setup trainer and train
    trainer.setup(load, save_every, train_dir, writer=writer,
                  interleave_report_every=interleave_report_every)
    trainer.train()


def _dump(rollout, dump_mode, of):
    if dump_mode == "list":
        yaml.safe_dump([list(rollout.genotype._asdict().values())], of)
    elif dump_mode == "str":
        yaml.safe_dump([str(rollout.genotype)], of)
    else:
        raise Exception("Unexpected dump_mode: {}".format(dump_mode))

@main.command(help="Random sample architectures.")
@click.argument("cfg_file", required=True, type=str)
@click.option("-o", "--out-file", required=True, type=str,
              help="the file to write the derived genotypes to")
@click.option("-n", default=1, type=int,
              help="number of architectures to derive")
@click.option("--gpu", default=0, type=int,
              help="the gpu to run deriving on")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--dump-mode", default="str", type=click.Choice(["list", "str"]))
@click.option("--unique", default=False, type=bool, is_flag=True,
              help="make sure rollout samples are unique")
def random_sample(cfg_file, out_file, n, gpu, seed, dump_mode, unique):
    LOGGER.info("CWD: %s", os.getcwd())
    LOGGER.info("CMD: %s", " ".join(sys.argv))

    setproctitle.setproctitle("awnas-random-sample cfg: {}; cwd: {}".format(cfg_file, os.getcwd()))

    # set gpu
    _set_gpu(gpu)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    # set seed
    if seed is not None:
        LOGGER.info("Setting random seed: %d.", seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    res = _init_components_from_cfg(cfg, device, evaluator_only=True)
    ss = res[0]
    
    sampled = 0
    ignored = 0
    rollouts = []
    genotypes = []
            
    while sampled < n:
        if unique:
            r = ss.random_sample()
            if r.genotype in genotypes:
                ignored += 1
                LOGGER.info(
                        "(ignored %d) Ignore duplicated arch", ignored)
            else:
                sampled += 1
                LOGGER.info(
                        "(choosed %d) Choose arch",
                        sampled)
                rollouts.append(r)
                genotypes.append(r.genotype)
        else:
            r = ss.random_sample()
            rollouts.append(r)
            genotypes.append(r.genotype)

    with open(out_file, "w") as of:
        for i, r in enumerate(rollouts):
            of.write("# ---- Arch {} ----\n".format(i))
            _dump(r, dump_mode, of)
            of.write("\n")


@main.command(help="Sample architectures, pickle loading controller directly.")
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
@click.option("--prob-thresh", default=None, type=float,
              help="only return rollout with bigger prob than this threshold")
@click.option("--unique", default=False, type=bool, is_flag=True,
              help="make sure rollout samples are unique")
def sample(load, out_file, n, save_plot, gpu, seed, dump_mode, prob_thresh, unique):
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

    if prob_thresh or unique:
        sampled = 0
        ignored = 0
        rollouts = []
        genotypes = []
        while sampled < n:
            rollout_cands = controller.sample(n - sampled)
            for r in rollout_cands:
                assert "log_probs" in r.info
                log_prob = np.array([utils.get_numpy(cg_lp) for cg_lp in r.info["log_probs"]]).sum()
                if np.exp(log_prob) < prob_thresh:
                    ignored += 1
                    LOGGER.info(
                        "(ignored %d) Ignore arch prob %.3e (< %.3e)",
                        ignored, np.exp(log_prob), prob_thresh)
                elif r.genotype in genotypes:
                    ignored += 1
                    LOGGER.info(
                        "(ignored %d) Ignore duplicated arch", ignored)
                else:
                    sampled += 1
                    LOGGER.info(
                        "(choosed %d) Choose arch prob %.3e (>= %.3e)",
                        sampled, np.exp(log_prob), prob_thresh)
                    rollouts.append(r)
                    genotypes.append(r.genotype)
    else:
        rollouts = controller.sample(n)

    with open(out_file, "w") as of:
        for i, r in enumerate(rollouts):
            if save_plot is not None:
                r.plot_arch(
                    filename=os.path.join(save_plot, str(i)),
                    label="Derive {}".format(i)
                )
            if "log_probs" in r.info:
                log_prob = np.array([utils.get_numpy(cg_lp) for cg_lp in r.info["log_probs"]]).sum()
                of.write("# ---- Arch {} log_prob: {:.3f} prob: {:.3e} ----\n".format(
                    i, log_prob, np.exp(log_prob)))
            else:
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
@click.option("--save-state-dict", default=None, type=str,
              help="If specified, save the sub state dict of the rollouts to this path; "
              "Only tested for CNN now.")
@click.option("--steps", default=None, type=int,
              help="number of batches to eval for each arch, default to be the whole derive queue.")
def eval_arch(cfg_file, arch_file, load, gpu, seed, save_plot, save_state_dict, steps):
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

    # initialize and load evaluator
    res = _init_components_from_cfg(cfg, device, evaluator_only=True)
    search_space = res[0] #pylint: disable=unused-variable
    evaluator = res[-1]
    path = os.path.join(load, "evaluator")
    LOGGER.info("Loading evalutor from %s", path)
    evaluator.load(path)

    # create the directory for saving plots
    if save_plot is not None:
        save_plot = utils.makedir(save_plot)

    # evaluate these rollouts using evaluator
    LOGGER.info("Eval...")
    rollouts = [rollout_from_genotype_str(geno, search_space) for geno in genotypes]
    num_r = len(rollouts)

    for i, r in enumerate(rollouts):
        evaluator.evaluate_rollouts([r], is_training=False,
                                    eval_batches=steps,
                                    return_candidate_net=save_state_dict)[0]
        if save_state_dict is not None:
            # save state dict of the candidate network (active members only)
            # corresponding to each rollout to `save_state_dict` path
            torch.save(r.candidate_net.state_dict(),
                       os.path.join(save_state_dict, str(i)))
        if save_plot is not None:
            r.plot_arch(
                filename=os.path.join(save_plot, str(i)),
                label="Derive {}; Reward {:.3f}".format(i, r.get_perf(name="reward"))
            )
        print("Finish test {}/{}\r".format(i+1, num_r), end="")
    for i, r in enumerate(rollouts):
        LOGGER.info("Arch %3d: %s", i, "; ".join(
            ["{}: {:.3f}".format(n, v) for n, v in r.perf.items()]))


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
    search_space, controller = _init_components_from_cfg(cfg, device, controller_only=True)

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
        trainer = _init_components_from_cfg(cfg, device, from_controller=True,
                                            search_space=search_space, controller=controller)[-1]

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


# ---- Multiprocess Train, Test using final_trainer ----
@main.command(help="Train an architecture.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--load", default=None, type=str,
              help="the checkpoint to load")
@click.option("--load-state-dict", default=None, type=str,
              help="the checkpoint (state dict) to load")
@click.option("--save-every", default=None, type=int,
              help="the number of epochs to save checkpoint every")
@click.option("--train-dir", default=None, type=str,
              help="the directory to save checkpoints")
def mptrain(seed, cfg_file, load, load_state_dict, save_every, train_dir):
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
    device = torch.cuda.current_device()
    torch.distributed.init_process_group(backend="nccl")

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
    objective = _init_component(cfg, "objective", search_space=search_space)
    trainer = _init_component(cfg, "final_trainer",
                              dataset=whole_dataset,
                              model=model,
                              device=device,
                              gpus=[device],
                              objective=objective)
    # check trainer support for data type
    expect(_data_type in trainer.supported_data_types())

    # start training
    LOGGER.info("Start training.")
    trainer.setup(load, load_state_dict, save_every, train_dir)
    trainer.train()


# ---- Train, Test using final_trainer ----
@main.command(help="Train an architecture.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--gpus", default="0", type=str,
              help="the gpus to run training on, split by single comma")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
@click.option("--load", default=None, type=str,
              help="the checkpoint to load")
@click.option("--load-state-dict", default=None, type=str,
              help="the checkpoint (state dict) to load")
@click.option("--save-every", default=None, type=int,
              help="the number of epochs to save checkpoint every")
@click.option("--train-dir", default=None, type=str,
              help="the directory to save checkpoints")
def train(gpus, seed, cfg_file, load, load_state_dict, save_every, train_dir):
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
    objective = _init_component(cfg, "objective", search_space=search_space)
    trainer = _init_component(cfg, "final_trainer",
                              dataset=whole_dataset,
                              model=model,
                              device=device,
                              gpus=gpu_list,
                              objective=objective)
    # check trainer support for data type
    expect(_data_type in trainer.supported_data_types())

    # start training
    LOGGER.info("Start training.")
    trainer.setup(load, load_state_dict, save_every, train_dir)
    trainer.train()


@main.command(help="Test a final-trained model.")
@click.argument("cfg_file", required=True, type=str)
@click.option("--load", type=str,
              help="the checkpoint to load")
@click.option("--load-state-dict", type=str,
              help="the checkpoint (state dict) to load")
@click.option("--split", "-s", multiple=True, required=True, type=str,
              help="evaluate on these dataset splits")
@click.option("--gpus", default="0", type=str,
              help="the gpus to run training on, split by single comma")
@click.option("--seed", default=None, type=int,
              help="the random seed to run training")
def test(cfg_file, load, load_state_dict, split, gpus, seed): #pylint: disable=redefined-builtin
    assert (load is None) + (load_state_dict is None) == 1, \
        "One and only one of `--load` and `--load-state-dict` arguments is required."

    setproctitle.setproctitle("awnas-test config: {}; load: {}; cwd: {}"\
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
    search_space = _init_component(cfg, "search_space")
    objective = _init_component(cfg, "objective", search_space=search_space)
    trainer = _init_component(cfg, "final_trainer",
                              dataset=whole_dataset,
                              model=_init_component(
                                  cfg, "final_model",
                                  search_space=search_space,
                                  device=device) if load_state_dict else None,
                              device=device,
                              gpus=gpu_list,
                              objective=objective)

    # check trainer support for data type
    _data_type = whole_dataset.data_type()
    expect(_data_type in trainer.supported_data_types())

    # start training
    LOGGER.info("Start eval.")
    trainer.setup(load, load_state_dict)
    for split_name in split:
        trainer.evaluate_split(split_name)


# ---- Utility for generating sample configuration file ----
@main.command(help="Dump the sample configuration.")
@click.argument("out_file", required=True, type=str)
@click.option("-d", "--data-type", default=None, type=click.Choice(AVAIL_DATA_TYPES),
              help="only dump the configs of the components support this data type")
@click.option("-r", "--rollout-type", default=None,
              type=click.Choice(list(BaseRollout.all_classes_().keys())),
              help="only dump the configs of the components support this rollout type")
def gen_sample_config(out_file, data_type, rollout_type):
    with open(out_file, "w") as out_f:
        out_f.write("# rollout_type: {}\n".format(rollout_type if rollout_type else ""))
        for comp_name in ["search_space", "dataset", "controller", "evaluator",
                          "weights_manager", "objective", "trainer"]:
            filter_funcs = []
            if data_type is not None:
                if comp_name == "dataset":
                    filter_funcs.append(lambda cls: data_type == cls.data_type())
                elif comp_name in {"evaluator", "weights_manager", "objective"}:
                    filter_funcs.append(lambda cls: data_type in cls.supported_data_types())
            if rollout_type is not None:
                if comp_name in {"controller", "weights_manager", "evaluator", "trainer"}:
                    filter_funcs.append(lambda cls: rollout_type in cls.supported_rollout_types())

            out_f.write(utils.component_sample_config_str(comp_name, prefix="# ",
                                                          filter_funcs=filter_funcs))
            out_f.write("\n")


@main.command(help="Dump the sample configuration for final training.")
@click.argument("out_file", required=True, type=str)
@click.option("-d", "--data-type", default=None, type=click.Choice(AVAIL_DATA_TYPES),
              help="only dump the configs of the components support this data type")
def gen_final_sample_config(out_file, data_type):
    with open(out_file, "w") as out_f:
        for comp_name in ["search_space", "dataset", "final_model", "final_trainer", "objective"]:
            filter_funcs = []
            if data_type is not None:
                if comp_name == "dataset":
                    filter_funcs.append(lambda cls: data_type == cls.data_type())
                elif comp_name in {"final_model", "final_trainer", "objective"}:
                    filter_funcs.append(lambda cls: data_type in cls.supported_data_types())

            out_f.write(utils.component_sample_config_str(comp_name, prefix="# ",
                                                          filter_funcs=filter_funcs))
            out_f.write("\n")


if __name__ == "__main__":
    main()
