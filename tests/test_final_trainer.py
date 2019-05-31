import os
import yaml
import torch
import pytest

SAMPLE_CFG_STR = """
## ---- Component search_space ----
# ---- Type rnn ----
search_space_type: rnn
search_space_cfg:
  # Schedulable attributes: 
  num_cell_groups: 1
  num_init_nodes: 1
  num_layers: 1
  num_steps: 8
  num_node_inputs: 1
  loose_end: false
  shared_primitives:
  - tanh
  - relu
  - sigmoid
  - identity
# ---- End Type rnn ----
## ---- End Component search_space ----

## ---- Component dataset ----
# ---- Type ptb ----
dataset_type: ptb
dataset_cfg:
  # Schedulable attributes: 
  data_dir: ./data/ptb
  shuffle: false
# ---- End Type ptb ----
## ---- End Component dataset ----

## ---- Component final_model ----
# ---- Type rnn_model ----
final_model_type: rnn_model
final_model_cfg:
  # Schedulable attributes: 
  genotypes: "cell=[('sigmoid', 0, 1), ('relu', 1, 2), ('relu', 1, 3), ('identity', 1,4), ('tanh', 2,5), ('sigmoid', 5,6), ('tanh', 3,7), ('relu', 5,8)], cell_concat=range(1, 9)"
  num_emb: 100
  num_hid: 100
  tie_weight: true
  decoder_bias: true
  share_primitive_weights: true
  share_from_weights: true
  batchnorm_step: false
  batchnorm_edge: false
  batchnorm_out: false
  max_grad_norm: 0.25
  dropout_emb: 0.1
  dropout_inp0: 0.2
  dropout_inp: 0.75
  dropout_hid: 0.25
  dropout_out: 0.75
# ---- End Type rnn_model ----
## ---- End Component final_model ----

## ---- Component final_trainer ----
# ---- Type rnn_trainer ----
final_trainer_type: rnn_trainer
final_trainer_cfg:
  # Schedulable attributes: 
  epochs: 10
  batch_size: 64
  eval_batch_size: 10
  learning_rate: 20.0
  momentum: 0.0
  optimizer_scheduler: null
  weight_decay: 8.e-7
  bptt_steps: 35
  reset_hidden_prob: null
  rnn_act_reg: 0.0
  rnn_slowness_reg: 1.e-3
  random_bptt: true
  valid_decay_window: 5
  schedule_cfg: null
# ---- End Type rnn_trainer ----
## ---- End Component final_trainer ----
"""

AWNAS_TEST_SLOW = os.environ.get("AWNAS_TEST_SLOW", None)

@pytest.mark.skipif(not AWNAS_TEST_SLOW, reason="parse corpus might be slow, by default not test")
def test_rnn_final_trainer_load_save(tmp_path):
    from aw_nas.main import _init_component
    from aw_nas.final import RNNGenotypeModel

    cfg = yaml.safe_load(SAMPLE_CFG_STR)

    device = "cuda:0"
    search_space = _init_component(cfg, "search_space")
    dataset = _init_component(cfg, "dataset")
    num_tokens = dataset.vocab_size
    rnn_model = _init_component(cfg, "final_model", search_space=search_space,
                                device=device, num_tokens=num_tokens)
    trainer = _init_component(cfg, "final_trainer", device=device, dataset=dataset,
                              model=rnn_model, gpus=[0])
    trainer.save(tmp_path)
    ori_params = list(rnn_model.parameters())
    assert ori_params == trainer.optimizer.param_groups[0]["params"]
    trainer.load(tmp_path)
    assert id(ori_params[0]) not in [id(p) for p in trainer.optimizer.param_groups[0]["params"]]
    assert list(trainer.model.parameters()) == trainer.optimizer.param_groups[0]["params"]
    # data is the same
    assert all((p == ori_p).all() for p, ori_p in zip(trainer.model.parameters(), ori_params))

    del search_space, ori_params, trainer, rnn_model

    model_path = os.path.join(tmp_path, "model.pt")
    model = torch.load(model_path, map_location=torch.device("cpu"))
    assert isinstance(model, RNNGenotypeModel)
