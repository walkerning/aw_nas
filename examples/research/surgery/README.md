Evaluating Efficient Performance Estimators of Neural Architectures
--------

If you find this work/code helpful, please cite:
```
@misc{ning2020surgery,
    title={Evaluating Efficient Performance Estimators of Neural Architectures},
    author={Xuefei Ning and Changcheng Tang and Wenshuo Li and Zixuan Zhou and Shuang Liang and Huazhong Yang and Yu Wang},
    year={2020},
    eprint={2008.03064},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```



The implementation of the NAS components for the three benchmarks, including the search space, GT evaluator, weights manager, (optional) architecture embedder and so on, are under the sub-package `aw_nas.btcs`. Check the READMEs in the sub-directories: [nb101](./nb101), [nb201](./nb201), and [nb301](./nb301) for more instructions on how to run one-shot training, architecture rewards estimation (`derive` and `eval-arch`), and criteria evaluation of the estimation results.

The implementation of the non-topological search spaces ResNet/ResNeXt is in a [plugin code snippet](./nds/plugin/germ_nds.py). Check the [nds][./nds] subdirectory for more instructions on how to run OS training/evaluation on NDS ResNet and ResNeXt.

There are many other assets (logs, configuration files, helper scripts) used in our paper, and the main assets can be found at [this URL](https://cloud.tsinghua.edu.cn/d/965b3ae1f80b45e9ba21/).



There are several helper scripts provided under this directory. They are

* `run_supernet_training.py`: Run multiple `awnas search` on multiple GPUs.
* `run_derive_ckpts.py`: Run multiple `awnas derive` (NB201) or `awnas eval-arch` (NB301/NB101)   on multiple GPUs.
* `run_oneshot_batch.py` : Run `awnas eval-arch`, and return the rewards as a list, in which each item is the reward in a valdiation batch (This script add a configuration `cfg["objective_cfg"]["aggregate_as_list"] = True`).
* `run_zero_shot.py`: Run zero-shot estimations. By default, multiple random initialization seeds would be used (20, 2020, 202020). Optionally, one can specificy `--evaluators` to calculate the zero-shot indicators on pretrained one-shot supernets.
* `evaluation.py`: Evaluate the `derive` (NB201) or `eval-arch` (NB301/101) results, and dump the corresponding criteria (i.e., One-shot average, Kendall's Tau, SpearmanR, P@top/bottom Ks, B/WR@K, etc.) into a pickle file with the `_statistics.pkl` suffix.

Please check the help messages as well as the codes to see how to run these helper scripts.

### NAS-Bench-201

See instructions under `nb201/`.

### NAS-Bench-301
See instructions under `nb301/`.

### NAS-Bench-1shot1-Sub3

See instructions under `nb101/`.

### NDS ResNet and ResNeXt (Non-topological search spaces)

See instructions under `nds/`.

### Zero-shot Estimators (ZSEs)

We appreciate the authors of [Zero-Cost Proxies for Lightweight NAS, ICLR2021] for their efforts in providing the research codes.
In order to evaluate ZSEs, we integrate their lib into our framework in `aw_nas/objective/zerocost.py`. Thus [their lib](https://github.com/SamsungLabs/zero-cost-nas) should be installed for evaluating ZSEs.
To evaluate `relu_logdet` and `relu` scores, one should use [our fork](https://github.com/zhouzx17/zero-cost-nas) of their lib.

The zeroshot configuration files on different search spaces are `nb101/zeroshot.yaml`, `nb201/zeroshot.yaml`, `nb301/zeroshot.yaml`, `nds/resnet_zeroshot.yaml`, and `resnexta_zeroshot.yaml`.
