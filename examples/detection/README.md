
## NAS for Detection Task

There are some examples of detection task. Like classification task, there are two main steps to be done: search an optimal sub-network in the search space and post-training. Currently, we only implement OFA + SSD on detection task, which includes supernet training, searching, and final training. This flow is similar to DetNAS [Chen, Yukang, et al., 2019].

Note that these configurations are for the COCO dataset, change `dataset_type` from `coco` to `voc`, and the `num_classes` configurations from 90 to 20 to run on VOC dataset.

### Supernet Training

The first step is training a supernet pregressively. There way we implement pregressive training is using `schdule_cfg` to configure the sample space of sub-networks, which is defined in `examples/detection/ssd_supernet_training.yaml`. 

There are a several of things you may need to edit in configure YAML file:

* `search_space_cfg.schedule_cfg.*.boundary`: the value of boundary means which epoch to set the value to correspond one. For example, `kernel_choice: {boundary: [1, 81], value: [[7], [7, 5, 3]]}` means that the 1st epoch setting `kernel_choice` as `[7]` and waiting util the 81st epoch setting `kernel_choice` as `[7, 5, 3]`. The epoch interval depends the network setting and the dataset.
* `evaluator_cfg`:
  * `batch_size`: the batch size during the supernet training process.
  * `mepa_optimizer`: we use `CosineAnnealingWarmRestarts` as the default optimizer that you can see details in [here](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts). The learning rate declines as consine curve and change back to `lr` immediately after `T_max` epoches. It would be better to set the `T_max` as the epoch interval of pregressive training.
  * `data_portion`: we usually set it `[0., 0.9, 0.1]`, which means using 90% training data to training supernet, and then using the rest 10% training data to search.
* `weights_manager_cfg.search_backbone_cfg`:
  * `backbone_type`: we implement two backbone, `mbv2_backbone` and `mbv3_backbone`. They use different configures.
  * `backbone_cfg.pretrained_path`: load the (ImageNet or other dataset) pretrained model for backbone.
  * `backbone_cfg.num_classes`: the number of classes of dataset. It does not include the background class, and the background label will be add to `0`. The number of finally predicted classes always includes the background, which will be `num_classes + 1`.


#### Distributed Parallelized Training

We implement a multiprocess decorator to support distributed parallelized training for `weights_manager`. `controller` also needs to be parallelized and synchronized to behave as expected. However, parallelized controller is not supported yet, which will cause weird behaviors. For example, controller in differnt process samples different rollout to assemble candidate net, then `forward` will get stuck because different batchnorm layers are waiting to be synchronized across devices. Considering this situation, we convert `BatchNorm2d` in weights manager to `SyncBatchNorm2d` only if the environment variable `AWNAS_SET_SYNC_BN` is set to avoid getting stuck.


```
from aw_nas.utils.parallel_utils import parallelize

@parallelize()
class CustomWeightsManager(object):
    
    def __init__(self, *args, **kwargs):
        """
        It is unnecessary to define `multiprocess` in `__init__` argument list.
        Decorator will do that in wrapper.
        """

```


#### Distillation
Distillation is an essential part during the OFA training process, which can pass the knowledge of the supernet to sub-networks to gain improvements. Currently, we implement [Adaptive Distillation Loss](https://arxiv.org/abs/1901.00366) with focal loss only.

At first we used all of logits of teacher as the soft labels that has the exactly same number with the hard labels. However, we noticed there was a significant decrease after applying distillation. Thus, we discarded all negative and ignored samples that are indicated by the hard labels (anchors whose IoU with GTboxes are lower than 0.5), and only applied the soft loss on positive samples, then we gained 1%~2% mAP improvement on VOC.

There are three different ways to build a teacher model: using supernet (`search` command only), loading a pytorch model, or initial an `aw_nas` final model.

```yaml
objective_cfg:
  ...
  soft_losses_cfg:
    teacher_cfg: ["supernet" or "$PYTORCH_MODEL_PATH" or a dict]
    losses_cfg:
      type: adaptive_distillation_loss
      cfg:
        alpha: 1.5
        gamma: 1.0
        temperature: 2.
        loss_coef: 0.1
```
If you want use the supernet as the teacher model, set `teacher_cfg` as `supernet`(**Recommened**). Or you can set an arbitrary network as the teacher model, only if `teacher_cfg` is set as the path of the pytorch model (the whole pytorch model, instead of the `state_dict`.)
Or you can initialize a aw_nas final model by passing a final model config to `teacher_cfg` as following:

```yaml
teacher_cfg:
  state_dict: $PATH_TO_MODEL
  teacher_final_type: ssd_final_model
  teacher_final_cfg:
    num_classes: 20
    feature_levels: [4, 5]
    backbone_type: ofa_final_model
    backbone_cfg:
      genotypes: "cell_0=1,cell_1=2,cell_2=2,cell_3=2,cell_4=2,cell_5=2,cell_0_block_0=(1, 3),cell_1_block_0=(3, 3),cell_1_block_1=(3, 3),cell_2_block_0=(3, 5),cell_2_block_1=(3, 5),cell_3_block_0=(4, 7),cell_3_block_1=(3, 3),cell_4_block_0=(3, 5),cell_4_block_1=(4, 3),cell_5_block_0=(6, 3),cell_5_block_1=(6, 7),cell_1_block_2=(0, 0),cell_1_block_3=(0, 0),cell_2_block_2=(0, 0),cell_2_block_3=(0, 0),cell_3_block_2=(0, 0),cell_3_block_3=(0, 0),cell_4_block_2=(0, 0),cell_4_block_3=(0, 0),cell_5_block_2=(0, 0),cell_5_block_3=(0, 0)"
      backbone_type: mbv3_backbone
      backbone_cfg:
        layer_channels: [16, 16, 24, 40, 80, 112, 160, 960, 1280]
        strides: [1, 2, 2, 2, 1, 2]
        mult_ratio: 1.
        kernel_sizes: [3, 5, 7]
        pretrained_path: null
    head_type: ssd_head_final_model
    head_cfg:
      expansions: [0.5, 0.5, 0.5, 0.5]
      channels: [512, 256, 256, 128]
      aspect_ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
      pretrained_path: null
    supernet_state_dict: null
```



#### Usage
```sh
aw_nas search examples/detection/ssd_supernet_training.yaml --train-dir ${dir} --save-every ${num} [--load ${previous_trained_dir}]
```

### Async Searching

Because it is very costly to measure mAP on dataset and GPUs will be at IDEL state, we implement multiprocess searching which is defined in `examples/detection/ssd_async_search.yaml` 

The only thing you may need to change is:
* `trainer_cfg.num_epochs`: the number of epochs you want to search, but the start epoch is according to the final epoch of the previous supernet training.
* `trainer_cfg.parallelism` and `trainer_cfg.dispatcher_cfg.gpu_ids`: the number of multiprocess and gpu ids.


#### Usage

```sh
aw_nas search examples/detection/ssd_async_search.yaml --train-dir ${dir} --save-every ${num} --load ${trained_dir}
```


### Post Training
Once you get an optimal sub-network, you can use it theoretically. But we recommend to training it on the whole training dataset to improve the accuracy.

In `examples/detection/ssd_final.yaml`, you may need to change:
* `final_model_cfg.backbone_cfg.genotypes`: the genotype of the subnetwork. You can copy the genotype from search result.
* `final_model_cfg.supernet_state_dict`: the path of state dict of supernet during the supernet training process. 
* `final_trainer_cfg`:
  * `batch_size`
  * `epochs`
  * `learning_rate`
  * `base_net_lr`: only available when `freeze_base_net` is `true`, not recommended.

#### Usage
```sh
aw_nas train examples/detection/ssd_final.yaml --train-dir ${dir} --save-every ${num}
```


## References
* **DetNAS**: Chen, Yukang, et al. "DetNAS: Backbone search for object detection." Advances in Neural Information Processing Systems. 2019.
* **Adaptive Distillation** Tang, Shitao, et al. "Learning efficient detector with semi-supervised adaptive distillation." arXiv preprint arXiv:1901.00366 (2019).
