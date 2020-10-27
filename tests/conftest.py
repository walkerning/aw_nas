import os
import pytest
import torch

class MockUnderlyWriter(object):
    def add_scalar(self, tag, value):
        """Add scalar!"""
        return "add_scalar_{}_{}".format(tag, value)

    def add_scalars(self, main_tag, value):
        """Add scarlas!"""
        return "add_scalars_{}_{}".format(main_tag, value)

@pytest.fixture
def writer():
    return MockUnderlyWriter()

@pytest.fixture
def super_net(request):
    cfg = getattr(request, "param", {})
    scfg = cfg.pop("search_space_cfg", {})
    is_dp = cfg.pop("dataparallel", False)
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import SuperNet
    search_space = get_search_space(cls="cnn", **scfg)
    device = "cuda"
    if is_dp and int(torch.__version__.split(".")[1]) > 2:
        # For torch>1.2.0, this candidate_member_mask is not supported when work will data parallel
        cfg["candidate_member_mask"] = False
    net = SuperNet(search_space, device, **cfg)
    return net

@pytest.fixture
def diff_super_net(request):
    cfg = getattr(request, "param", {})
    scfg = cfg.pop("search_space_cfg", {})
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import DiffSuperNet
    search_space = get_search_space(cls="cnn", **scfg)
    device = "cuda"
    net = DiffSuperNet(search_space, device, **cfg)
    return net

@pytest.fixture
def rnn_super_net(request):
    cfg = getattr(request, "param", {})
    num_tokens = cfg.pop("num_tokens", 10)
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import RNNSuperNet
    search_space = get_search_space(cls="rnn")
    device = "cuda"
    net = RNNSuperNet(search_space, device, num_tokens, **cfg)
    return net

@pytest.fixture
def rnn_diff_super_net(request):
    cfg = getattr(request, "param", {})
    num_tokens = cfg.pop("num_tokens", 10)
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import RNNDiffSuperNet
    search_space = get_search_space(cls="rnn")
    device = "cuda"
    net = RNNDiffSuperNet(search_space, device, num_tokens, **cfg)
    return net

@pytest.fixture
def morphism(request):
    cfg = getattr(request, "param", {})
    scfg = cfg.pop("search_space_cfg", {})
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import MorphismWeightsManager
    search_space = get_search_space(cls="cnn", **scfg)
    device = "cuda"
    net = MorphismWeightsManager(search_space, device, "mutation")
    return net

from aw_nas.rollout.mutation import Population

sample_config = """
## ---- Component search_space ----
# ---- Type cnn ----
search_space_type: cnn
search_space_cfg:
  # Schedulable attributes: 
  num_cell_groups: 2
  num_init_nodes: 2
  num_layers: 20
  cell_layout: null
  reduce_cell_groups:
  - 1
  num_steps: 4
  num_node_inputs: 2
  shared_primitives:
  - none
  - max_pool_3x3
  - avg_pool_3x3
  - skip_connect
  - sep_conv_3x3
  - sep_conv_5x5
  - dil_conv_3x3
  - dil_conv_5x5
  cell_shared_primitives: null
# ---- End Type cnn ----
## ---- End Component search_space ----

## ---- Component dataset ----
# ---- Type cifar10 ----
dataset_type: cifar10
dataset_cfg:
  # Schedulable attributes: 
  cutout: null
# ---- End Type cifar10 ----
## ---- End Component dataset ----

## ---- Component final_model ----
# ---- Type cnn_final_model ----
final_model_type: cnn_final_model
final_model_cfg:
  # Schedulable attributes: dropout_path_rate
  num_classes: 10
  init_channels: 20
  layer_channels: []
  stem_multiplier: 3
  dropout_rate: 0.1
  dropout_path_rate: 0.2
  auxiliary_head: false
  auxiliary_cfg: null
  use_stem: conv_bn_3x3
  stem_stride: 1
  stem_affine: true
  cell_use_preprocess: true
  cell_pool_batchnorm: false
  cell_group_kwargs: null
  cell_independent_conn: false
  schedule_cfg: null
# ---- End Type cnn_final_model ----
## ---- End Component final_model ----

## ---- Component final_trainer ----
# ---- Type cnn_trainer ----
final_trainer_type: cnn_trainer
final_trainer_cfg:
  # Schedulable attributes: 
  epochs: 50
  batch_size: 96
  optimizer_type: SGD
  optimizer_kwargs: null
  learning_rate: 0.05
  momentum: 0.9
  warmup_epochs: 0
  optimizer_scheduler:
    T_max: 50
    eta_min: 0.001
    type: CosineAnnealingLR
  weight_decay: 0.0003
  no_bias_decay: false
  grad_clip: 5.0
  auxiliary_head: false
  auxiliary_weight: 0.0
  add_regularization: false
  save_as_state_dict: false
  eval_no_grad: true
  schedule_cfg: null
# ---- End Type cnn_trainer ----
## ---- End Component final_trainer ----

## ---- Component objective ----
# ---- Type classification ----
objective_type: classification
objective_cfg:
  # Schedulable attributes: 
  {}
# ---- End Type classification ----
## ---- End Component objective ----
"""

import yaml
import collections
import numpy as np
from six import StringIO
from aw_nas.rollout.mutation import ConfigTemplate, ModelRecord

class StubPopulation(Population):
    def __init__(self, search_space, num_records=3, config_template=sample_config):
        cfg_template = ConfigTemplate(yaml.load(StringIO(config_template)))
        model_records = collections.OrderedDict()
        for ind in range(num_records):
            rollout = search_space.random_sample()
            cfg = cfg_template.create_cfg(rollout.genotype)
            model_records[ind] = ModelRecord(rollout.genotype, cfg, search_space,
                                             info_path="/a_no_exist_path/{}.yaml".format(ind),
                                             checkpoint_path="/a_not_exist_path/{}".format(ind),
                                             finished=True,
                                             confidence=1,
                                             perfs={"acc": np.random.rand(),
                                                    "loss": np.random.uniform(0, 10)})
        super(StubPopulation, self).__init__(search_space, model_records, cfg_template)

@pytest.fixture
def init_population_dir(tmp_path, request):
    import torch
    from aw_nas.common import get_search_space
    from aw_nas import utils
    from aw_nas.main import _init_component

    cfg = getattr(request, "param", {})
    scfg = cfg.pop("search_space_cfg", {})
    search_space = get_search_space(cls="cnn", **scfg)
    path = utils.makedir(os.path.join(tmp_path, "init_population_dir"))
    ckpt_dir = utils.makedir(os.path.join(tmp_path, "init_ckpt_path"))

    # dump config template
    with open(os.path.join(path, "template.yaml"), "w") as wf:
        wf.write(sample_config)

    # generate mock records, ckpts
    num_records = cfg.get("num_records", 3)
    cfg_template = ConfigTemplate(yaml.load(StringIO(sample_config)))
    model_records = collections.OrderedDict()
    for ind in range(num_records):
        rollout = search_space.random_sample()
        cfg = cfg_template.create_cfg(rollout.genotype)
        ckpt_path = os.path.join(ckpt_dir, str(ind))
        cnn_model = _init_component(
            cfg, "final_model", search_space=search_space, device=torch.device("cpu"))
        torch.save(cnn_model, ckpt_path)
        model_records[ind] = ModelRecord(rollout.genotype, cfg, search_space,
                                         checkpoint_path=ckpt_path, finished=True,
                                         confidence=1,
                                         perfs={"acc": np.random.rand(),
                                                "loss": np.random.uniform(0, 10)})
    # initialize population
    population = Population(search_space, model_records, cfg_template)
    # save population
    population.save(path, 0)

    # ugly: return ss for reference
    return (path, search_space)

@pytest.fixture
def population(request):
    cfg = getattr(request, "param", {})
    init_dirs = cfg.get("init_dirs", None)
    scfg = cfg.pop("search_space_cfg", {})
    s_type = cfg.pop("search_space_type", "cnn")
    cfg_template = cfg.pop("cfg_template", sample_config)
    from aw_nas.common import get_search_space
    search_space = get_search_space(s_type, **scfg)
    if init_dirs:
        population = Population.init_from_dirs(init_dirs, search_space)
    else:
        population = StubPopulation(search_space, num_records=cfg.get("num_records", 3),
                  config_template=cfg_template)
    return population
        
@pytest.fixture
def ofa_super_net(request):
    cfg = getattr(request, "param", {})
    scfg = cfg.pop("search_space_cfg", {})
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import OFASupernet
    search_space = get_search_space(cls="ofa", **scfg)
    device = "cuda"
    net = OFASupernet(search_space, device, rollout_type="ofa", **cfg)
    return net

@pytest.fixture(scope="session")
def nasbench_search_space():
    from aw_nas.common import get_search_space
    # might take several minutes
    search_space = get_search_space("nasbench-101")
    return search_space

@pytest.fixture
def nasbench_201(request):
    cfg = getattr(request, "param", {})
    scfg = cfg.pop("search_space_cfg", {})
    from aw_nas.common import get_search_space
    from aw_nas.btcs.nasbench_201 import NB201SharedNet
    search_space = get_search_space(cls="nasbench-201", **scfg)
    device = "cuda"
    net = NB201SharedNet(search_space, device, **cfg)
    return net

