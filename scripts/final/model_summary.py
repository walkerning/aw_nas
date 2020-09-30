"""
the scripts to get the model flops/params for awnas
$1 - the input cfg file
"""

from __future__ import print_function

import os
import sys

import yaml
import torch
from torch.backends import cudnn

from torchviz import make_dot

import aw_nas
from aw_nas import utils
from aw_nas.dataset import AVAIL_DATA_TYPES
from aw_nas import utils, BaseRollout
from aw_nas.common import rollout_from_genotype_str
from aw_nas.utils.common_utils import _OrderedCommandGroup
from aw_nas.utils.vis_utils import WrapWriter
from aw_nas.utils import RegistryMeta
from aw_nas.utils import logger as _logger
from aw_nas.utils.exception import expect

LOGGER = _logger.getChild("main")


def _init_components_from_cfg(
    cfg,
    device,
    evaluator_only=False,
    controller_only=False,
    from_controller=False,
    search_space=None,
    controller=None,
):
    """
    Initialize components using configuration.
    Order:
    `search_space`, `controller`, `dataset`, `weights_manager`, `objective`, `evaluator`, `trainer`
    """
    rollout_type = cfg["rollout_type"]
    if not from_controller:
        search_space = _init_component(cfg, "search_space")
        if not evaluator_only:
            controller = _init_component(
                cfg,
                "controller",
                search_space=search_space,
                device=device,
                rollout_type=rollout_type,
            )
            if controller_only:
                return search_space, controller
    else:  # continue components initialization from controller stage
        assert search_space is not None and controller is not None

    # dataset
    whole_dataset = _init_component(cfg, "dataset")
    _data_type = whole_dataset.data_type()

    # weights manager
    if _data_type == "sequence":
        # get the num_tokens
        num_tokens = whole_dataset.vocab_size
        LOGGER.info("Dataset %s: vocabulary size: %d", whole_dataset.NAME, num_tokens)
        weights_manager = _init_component(
            cfg,
            "weights_manager",
            search_space=search_space,
            device=device,
            rollout_type=rollout_type,
            num_tokens=num_tokens,
        )
    else:
        weights_manager = _init_component(
            cfg,
            "weights_manager",
            search_space=search_space,
            device=device,
            rollout_type=rollout_type,
        )
    expect(_data_type in weights_manager.supported_data_types())

    # objective
    objective = _init_component(cfg, "objective", search_space=search_space)

    # evaluator
    evaluator = _init_component(
        cfg,
        "evaluator",
        dataset=whole_dataset,
        weights_manager=weights_manager,
        objective=objective,
        rollout_type=rollout_type,
    )
    expect(_data_type in evaluator.supported_data_types())

    if evaluator_only:
        return search_space, whole_dataset, weights_manager, objective, evaluator

    # trainer
    LOGGER.info("Initializing trainer and starting the search.")
    trainer = _init_component(
        cfg,
        "trainer",
        evaluator=evaluator,
        controller=controller,
        rollout_type=rollout_type,
    )

    return (
        search_space,
        whole_dataset,
        weights_manager,
        objective,
        evaluator,
        controller,
        trainer,
    )


def _init_component(cfg, registry_name, **addi_args):
    type_ = cfg[registry_name + "_type"]
    cfg = cfg.get(registry_name + "_cfg", None)
    if not cfg:
        cfg = {}
    # config items will override addi_args items
    addi_args.update(cfg)
    LOGGER.info("Component [%s] type： %s", registry_name, type_)
    cls = RegistryMeta.get_class(registry_name, type_)
    if LOGGER.level < 20:  # logging is at debug level
        whole_cfg_str = cls.get_current_config_str(cfg)
        LOGGER.debug(
            "%s %s config:\n%s",
            registry_name,
            type_,
            utils.add_text_prefix(whole_cfg_str, "  "),
        )
    return cls(**addi_args)


def _set_gpu(gpu):
    if gpu is None:
        return
    if torch.cuda.is_available():
        set_reproducible = bool(os.environ.get("AWNAS_REPRODUCIBLE", False))
        if set_reproducible:
            LOGGER.info(
                "AWNAS_REPRODUCIBLE environment variable set. Disable cudnn.benchmark, "
                "enable cudnn.deterministic for better reproducibility"
            )
        torch.cuda.set_device(gpu)
        if set_reproducible:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
        cudnn.enabled = True
        LOGGER.info("GPU device = %d" % gpu)
    else:
        LOGGER.warning("No GPU available, use CPU!!")


try:
    cfg_from = sys.argv[1]
except IndexError:
    cfg_from = "./config.yaml"

with open(cfg_from, "r") as f:
    cfg = yaml.safe_load(f)

cfg["final_trainer_cfg"]["multiprocess"] = False

gpu_list = []

if not gpu_list:
    _set_gpu(None)
    device = "cpu"
else:
    _set_gpu(gpu_list[0])
    device = torch.device(
        "cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu"
    )


seed = 2020

# initialize components
LOGGER.info("Initializing components.")
search_space = _init_component(cfg, "search_space")
whole_dataset = _init_component(cfg, "dataset")

_data_type = whole_dataset.data_type()

model = _init_component(cfg, "final_model", search_space=search_space, device=device)

# check model support for data type
expect(_data_type in model.supported_data_types())
objective = _init_component(cfg, "objective", search_space=search_space)
trainer = _init_component(
    cfg,
    "final_trainer",
    dataset=whole_dataset,
    model=model,
    device=device,
    gpus=gpu_list,
    objective=objective,
)
# check trainer support for data type
expect(_data_type in trainer.supported_data_types())

# start training
LOGGER.info("Start training.")
# trainer.setup(load, load_state_dict, save_every, train_dir)

if cfg["dataset_type"] == "cifar10":
    dummy_input = torch.rand([2, 3, 32, 32]).to(device)
elif cfg["dataset_type"] == "imagenet":
    dummy_input = torch.rand([2, 3, 224, 224]).to(device)
else:
    raise AssertionError("Dataset not supported")

output = trainer.model.forward(dummy_input)
dot = make_dot(output, params=dict(trainer.model.named_parameters()))

dot.format = "pdf"
dot.render("./test-torchviz")

flops = trainer.model.total_flops / 1.0e6
bi_flops = trainer.model.bi_flops / 1.0e6

model_params = utils.count_parameters(trainer.model, count_binary=True) / 1.0e6
print("param size = {} M | bi-param {} M".format(model_params[0], model_params[1]))
print("flops = {} M | bi-flops {} M".format(flops, bi_flops))
