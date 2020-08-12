## A Generic Graph-based Neural Architecture Encoding Scheme for Predictor-based NAS (GATES), in ECCV 2020

If you find this work/repo helpful, please cite:
```
@misc{ning2020generic,
    title={A Generic Graph-based Neural Architecture Encoding Scheme for Predictor-based NAS},
    author={Xuefei Ning and Yin Zheng and Tianchen Zhao and Yu Wang and Huazhong Yang},
    year={2020},
    eprint={2004.01899},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

The implementations of GATES (detailed in Appendix. Sec.1) in OON/OOE search spaces are in `aw_nas/utils/torch_utils.py`: `DenseGraphFlow`: OON, e.g., NAS-Bench-101; `DenseGraphOpEdgeFlow`: OOE, e.g., ENAS; `DenseGraphSimpleOpEdgeFlow`: simple OOE, e.g., NAS-Bench-201.

We provide the sample codes/logs/configs of training different predictors on NAS-Bench-101, and the sample logs/configs of trainig different predictors on NAS-Bench-201 (Corresponding to the Sec. 4.1 ~ Sec. 4.2 in the paper). The sample configs/logs are in two directories: `nb101_cfgs`, and `nb201_cfgs` from [this url](https://cloud.tsinghua.edu.cn/d/7204cb83f8384f9aa82a/).

* Under `nb101_cfgs/tr_5e-4` are the example configurations and logs on NAS-Bench-101. 0.05% (190) of the training data are used to train the predictors. You could run `bash print_res.sh nb101_cfgs` to print the average test kendall's tau of the last 5 epochs of these exps.
* Under `nb201_cfgs/` are the example configurations and logs on NAS-Bench-201. 1%/10% of the training data are used to train the predictors. You could run `bash print_res.sh nb201_cfgs` to print the average test kendall's tau of the last 5 epochs of these exps.

> You can also use the sample `aw_nas` code provided in the cloud drive to run the test, while this repository provides the full `aw_nas` codes.

## Run the training exps on NAS-Bench-101
Folow the following steps to run experiments.

#### Install
* Install `awnas` following the [instructions](../../../README.md).
* Install the NAS-Bench-101 package following the instruction https://github.com/google-research/nasbench.
* Also, download the `nasbench_only108.tfrecord` and put it under `~/awnas/data/nasbench-101/`.

#### Dump nb101 data
Dump nb101 pickle data (all patched to 7x7):
```
python dump_nb101_allv.py
```

Two pickle files would be generated under the current working dir: `nasbench_allv.pkl`, `nasbench_allv_valid.pkl`

#### Modify configurations
Optionally, make some changes to the `config.yaml` files under `nb101_cfgs/tr_5e-4`, for example
  * To use 1% of the training archs rather than 0.05% (190). Modify `train_ratio: 0.0005` to `train_ratio: 0.01` in these configs.
  * To use the regression loss instead of the hinge ranking loss, modify `compare: true` to `compare: false` in these configs.
  * To use different number of GATES/GCN layers, change the `arch_network_cfg.arch_embedder_cfg.gcn_out_dims` list.
  * See the config file for more configurations!

#### Run exp
Run `python scripts/nasbench/train_nasbench_pkl.py <yaml_config_file> --train-dir <train_result_dir> --gpu 0`
