## A breakup of the ENAS configuration

## Component `search space`

> aw_nas adopts the ENAS-like search space scheme, described as the config search_space section

``` yaml
search_space_type: cnn
search_space_cfg:
  # Schedulable attributes: 
  cell_layout: [0, 0, 1, 0, 1, 0, 1, 0]
  num_cell_groups: 2
  num_init_nodes: 1
  num_layers: 8
  num_node_inputs: 1
  num_steps: 1
  reduce_cell_groups:
  - 1 
  shared_primitives:
  - none
  - max_pool_3x3
  - avg_pool_3x3
  - skip_connect
  - sep_conv_3x3
  - sep_conv_5x5
  - dil_conv_3x3
  - dil_conv_5x5
```

* num_cell_groups(2): there are ```$num_cell_groups``` kinds of cells(blocks/layers)
* num_layers(8): the arch is composed ```num_layers``` cells stacked together
* reduce_cell_groups(1): the 2nd type of cell is reduction cell
* cell_layout([0,0,1,0,1,0,1,0]): the 3rd, 5th, 7th of 8 cells are the reduction cell, others are normal
* num_node_inputs(1): there are ```num_node_inputs``` input node for the cell
* num_steps(1): there are ```num_steps``` internal node(the node except for the input/output_node) for the cell 

* *For cell computation, multiple cell inputs are added together and the outputs are concatenated together*

![](./pics/enas_cell.jpg)


Actually, Also, many examples in ```$AW_NAS_HOME/examples/baselines```

## Component Final Model

> define the architecture for final training

``` yaml
## ---- Component final_model ----
# ---- Type cnn_genotype ----
final_model_type: cnn_final_model
final_model_cfg:
  # Schedulable attributes: dropout_path_rate
  genotypes:
      CNNGenotype(normal_0=[('skip_connect', 0, 2), ('sep_conv_3x3', 1, 2), ('max_pool_3x3',
      2, 3), ('max_pool_3x3', 0, 3), ('none', 0, 4), ('sep_conv_5x5', 1, 4), ('skip_connect',
      2, 5), ('sep_conv_3x3', 3, 5)], reduce_1=[('max_pool_3x3', 0, 2), ('sep_conv_5x5',
      1, 2), ('max_pool_3x3', 0, 3), ('dil_conv_3x3', 1, 3), ('max_pool_3x3', 0, 4), ('max_pool_3x3',
      2, 4), ('skip_connect', 3, 5), ('max_pool_3x3', 0, 5)], normal_0_concat=[2, 3, 4,
      5], reduce_1_concat=[2, 3, 4, 5])
  auxiliary_cfg: null
  auxiliary_head: True
  dropout_path_rate: 0.2
  dropout_rate: 0.1
  init_channels: 36
  num_classes: 10
  schedule_cfg:
    dropout_path_rate:
      type: add
      start: 0
      every: 1
      step: 0.000333 # 0.2/600
  stem_multiplier: 3
# ---- End Type cnn_genotype ----
## ---- End Component final_model ----
```

* genotypes: num of genotypes aligns with ```$num_cell_groups```, the name of genotype is the op defined and registered in the ```$aw_nas/ops/```
* auxiliary_cfg(0.4): only works when auxiliary is true, the ```$auxiliary_cfg``` value is the weight of the auxiliary head
* layer_channels([64, 64, 64, 128, 128, 256, 256, 512, 512]): num of the ```$layer_channels``` should be the same as the ```$num_layers```, denoting the width of certain cell(block/layer)
* use_stem("conv_bn_relu_3x3"): the type of the stem block
* stem_multiplier(1): stack multiple stem blocks
* init_channels(64): the num of channels feed into the arch, (if no stem is used, the ```$init_channels``` should be 3, as the RGB of the input image, otherwise, its value should align with the stem block width)

## Component Trainer

``` yaml
## ---- Component final_trainer ----
# ---- Type cnn_trainer ----
final_trainer_type: cnn_trainer
final_trainer_cfg:
  # Schedulable attributes: 
  auxiliary_head: true
  auxiliary_weight: 0.4
  batch_size: 128
  epochs: 600
  grad_clip: 5.0
  learning_rate: 0.05
  momentum: 0.9
  no_bias_decay: false
  optimizer_scheduler:
    eta_min: 0.001
    T_max: 600
    type: CosineAnnealingLR
  schedule_cfg: null
  warmup_epochs: 0
  weight_decay: 0.0003
  save_as_state_dict: true
# ---- End Type cnn_trainer ----
## ---- End Component final_trainer ----
```

* Training hyper-params 

---

# Config for Searching

> the example config is $AW_NAS_HOME/examples/basic/enas.yaml

In searching each component has a ```rollout_type```:
  * Example
    * eNAS - discrete
    * darts - differentiable
    * nasbench101
    * you could use command ```awnas registry -t rollout``` to refer to supported rollout types
  * **All ```rollout_type``` should be the same for each component**


## Component Search Space

(same as above mentioned search space)

## Component Controller

> aw_nas supports many types of controllers, controller samples arch(rollout) from search space 

``` yaml
## ---- Component controller ----
# ---- Type rl ----
controller_type: rl
controller_cfg:
  # Schedulable attributes: 
  rollout_type: discrete
  mode: eval
  independent_cell_group: false
  # ---- Type embed_lstm ----
  controller_network_type: embed_lstm
  controller_network_cfg:
    # Schedulable attributes: softmax_temperature, force_uniform
    num_lstm_layers: 1
    controller_hid: 100
    attention_hid: 100
    softmax_temperature: null
    tanh_constant: 1.1
    op_tanh_reduce: 2.5
    force_uniform: false
    schedule_cfg: null
  # ---- End Type embed_lstm ----
  # ---- Type pg ----
  rl_agent_type: pg
  rl_agent_cfg:
    # Schedulable attributes: 
    alpha: 0.99
    gamma: 1.0
    entropy_coeff: 0.01
    max_grad_norm: null
    batch_update: true
  # ---- End Type pg ----
# ---- End Type rl ----
## ---- End Component controller ----
```


## Component Weight-manager

> the weight manager fills weights into arch(rollout) sampled by the controller

``` yaml
## ---- Component weights_manager ----
# ---- Type supernet ----
weights_manager_type: supernet
weights_manager_cfg:
  # Schedulable attributes:
  gpus: [0] 
  rollout_type: discrete
  num_classes: 10
  init_channels: 20
  stem_multiplier: 3
  max_grad_norm: 5.0
  dropout_rate: 0.1
  candidate_member_mask: true
  candidate_cache_named_members: true
  candidate_virtual_parameter_only: true
# ---- End Type supernet ----
## ---- End Component weights_manager ----
```

* init_channels: the initial number of channels 
* gpus([0]): the weight-manager supports data-parallel when the ```$gpus``` is [1,...,N], then use ```--gpu 0``` arg when calling awnas search. (the awnas train adopts a different scheme, directly using ```--gpus [0,1,2,3]``` when calling ```awnas train```) 

## Component Evaluator

> the evaluator takes the arch(by controller), filled with params(by weight-manager), outputs the arch performance(reward) for updating the controller

``` yaml
## ---- Component evaluator ----
# ---- Type mepa ----
evaluator_type: mepa
evaluator_cfg:
  # Schedulable attributes: controller_surrogate_steps, mepa_surrogate_steps, mepa_samples
  rollout_type: discrete
  batch_size: 128
  controller_surrogate_steps: 0
  mepa_surrogate_steps: 0
  mepa_optimizer:
    lr: 0.05
    momentum: 0.9
    type: SGD
    weight_decay: 0.0003
  mepa_scheduler:
    eta_min: 0.0
    T_max: 200
    type: CosineAnnealingLR
  surrogate_optimizer: null
  surrogate_scheduler: null
  mepa_samples: 1
  data_portion:
  - 0.0
  - 0.8
  - 0.2
  mepa_as_surrogate: false
  bptt_steps: 35
  schedule_cfg: null
# ---- End Type mepa ----
## ---- End Component evaluator ----
```

* (description of data-portion)
  * data-potion: donot use the 1st 
  * train valid
* (pseudo code here)

* **Noted that aw_nas intrinsically supports the mepa evaluator**(for more details of mepa, please refer to [this paper]())
* mepa optimizer & scheduler are used for updating the shared-weights
* mepa_samples: numbers of architecture Monte-Carlo samples in every supernet training step
* when the ```$controller_surrogate_steps $mepa_surrogate_steps``` are all zero, it will be the naive evaluator


## Component Trainer

> trainer describes the training of controller in the searching

``` yaml
## ---- Component trainer ----
# ---- Type simple ----
trainer_type: simple
trainer_cfg:
  # Schedulable attributes: controller_samples, derive_samples
  rollout_type: differentiable
  epochs: 50
  test_every: 100
  controller_optimizer:
    lr: 3.e-4
    betas: [0.5,0.999]
    weight_decay: 1.e-3
    type: Adam
  controller_scheduler: null
  controller_samples: 1
  derive_samples: 8
  rollout_batch_size: 1
  evaluator_steps: null
  controller_steps: null
  controller_train_every: 1
  controller_train_begin: 1
  interleave_controller_every: 1
  schedule_cfg: null
# ---- End Type simple ----
## ---- End Component trainer ----
```

* controller_optimizer: describe the optimizer for optimizing the controller
* （?）interleave_controller_every: the frequency of updating the controller 




