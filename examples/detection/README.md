
## NAS for Detection Task

There are some examples of detection task. Like classification task, there are two main steps to be done: search an optimal sub-network in the search space and post-training. Currently, we only implement OFA + SSD on detection task, which includes supernet training, searching, and final training. This flow is similar to DetNAS [Chen, Yukang, et al., 2019].

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