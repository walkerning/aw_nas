#pylint: disable-all
import os

import yaml
import pytest

# we use environments variable to mark slow instead of register new pytest marks here.
AWNAS_TEST_SLOW = os.environ.get("AWNAS_TEST_SLOW", None)

sample_cfg_str = """
## ---- Component search_space ----
# ---- Type cnn ----
search_space_type: cnn
search_space_cfg:
  # Schedulable attributes: 
  num_cell_groups: 2
  num_init_nodes: 2
  num_layers: 5
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
  init_channels: 10
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

@pytest.mark.skipif(not AWNAS_TEST_SLOW, reason="tune evaluator is slow")
def test_bf_tune_evaluator(tmp_path):
    from aw_nas.objective import ClassificationObjective
    from aw_nas.evaluator.bftune import BFTuneEvaluator
    from aw_nas import get_search_space

    ss = get_search_space("cnn")
    objective = ClassificationObjective(ss)
    t_cfg_fname = os.path.join(tmp_path, "template.yaml")
    with open(t_cfg_fname, "w") as cfg_f:
        cfg_f.write(sample_cfg_str)
    evaluator = BFTuneEvaluator(None, None, objective, template_cfg_file=t_cfg_fname,
                                save_every=10, bf_checkpoints=[1, 2, 3])
    rollout = ss.random_sample()
    rollout.train_dir = os.path.join(tmp_path, str(hash(rollout)))
    print("train dir: ", rollout.train_dir)
    rollout = evaluator.evaluate_rollouts([rollout], is_training=True)[0]
    print("perf after stage 0: ", rollout.perf["reward"])

    rollout = evaluator.evaluate_rollouts([rollout], is_training=True)[0]
    print("perf after stage 1: ", rollout.perf["reward"])
