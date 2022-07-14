CLOSE: Curriculum Learning On the Sharing Extent Towards Better One-shot NAS
--------

If you find this work/code helpful, please cite:
```
@inproceedings{
zhou2022close,
title={CLOSE: Curriculum Learning On the Sharing Extent Towards Better One-shot NAS},
author={Zixuan Zhou and Xuefei Ning and Yi Cai and Jiashu Han and Yiping Deng and Yuhan Dong and Huazhong Yang and Yu Wang},
booktitle={The 17th European Conference on Computer Vision (ECCV)},
year={2022}
}
```



## Code
The implementation of CLOSENet for the two topological NAS benchmarks, i.e., NAS-Bench-201 and NAS-Bench-301, are under the sub-package `aw_nas.btcs` and `aw_nas.weights_manager`, respectively. The training configurations are provided under the sub-directories: [nb201](./nb201), [nb301](./nb301). To run the training/estimation of CLOSENet, one can simply use the `awnas search`/`awnas eval-arch` instruction with the training configurations provided above. 

The implementation of CLOSENet for the non-topological search spaces ResNet/ResNeXt is in a [plugin code snippet](./nds/plugin/germ_nds.py). Check the [nds](./nds) subdirectory for the training configurations on NDS ResNet and ResNeXt.

The training configurations are named by the corresponding hyperparameters. For example, `./nb301/node_cl-2468_lr-4_mlp-200x3_wit.yaml` corresponds to the CLOSENet with five GLOW blocks, which are added in 1, 201, 401, 601, 801 epoch, respectively. The learning rate is restarted in 401 epoch following the SRT. The MLP in the GATE module is constructed by three 200-dim fully-connected layers. And the WIT is applied.

There are several helper scripts provided under this directory. They are

* `run_supernet_training.py`: Run multiple `awnas search` on multiple GPUs.
* `run_derive_ckpts.py`: Run multiple `awnas derive` (NB201) or `awnas eval-arch` (NB301/NDS-ResNet/NDS-ResNeXt-A)  on multiple GPUs.

Please check the help messages as well as the codes to see how to run these helper scripts.


## Instruction
To run the training of CLOSENet, one can use the following instruction (for example, on NAS-Bench-301):
```sh
$ awnas search ./nb301/node_cl-2468_lr-4_mlp-200x3_wit.yaml --gpu [gpu id] --seed [random seed] --save-every 50 --train-dir results/nb301_closenet_train/
```

After training, one can use the following instruction to estimate the architectures' performances:
```sh
$ awnas eval-arch ./nb301/node_cl-2468_lr-4_mlp-200x3_wit.yaml $AWNAS_HOME/data/nasbench-301/nb301_archs.yaml --load results/nb301_closenet_train/1000/ --gpu [gpu id] --seed [random seed] --dump-rollouts results/nb301_closenet_train/derive_results_1000.pkl
```
