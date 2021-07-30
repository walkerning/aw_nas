Nasbench-201
--------

### Prepare

1. Install the NB201 package following instructions [here](https://github.com/D-X-Y/NAS-Bench-201).
2. Download the data file `NAS-Bench-201-v1_0-e61699.pth`, and put it under the path `$AWNAS_HOME/data/nasbench-201/` (without AWNAS_HOME explicitly overrode, this path is `$HOME/awnas/data/nasbench-201/`).
3. Several other data files are used in the evaluation script and the de-isomorphism sampling (`examples/research/surgery/data/non-isom.txt`). These files (`non-isom.txt`, `iso_dict.yaml`, `iso_dct.txt`, `deiso_dict.txt`) should be downloaded from [this url](https://cloud.tsinghua.edu.cn/d/97a8f29e58cc4e87a3d3/), and also put under the path `$AWNAS_HOME/data/nasbench-201/`.
Also, run `ln -s $AWNAS_HOME/data/nasbench-201/non-isom.txt $AWNAS_HOME/data/nasbench-201/non-isom5.txt`.

### Train one-shot supernets

`awnas search --gpu 0 --seed [random seed] --save-every 50 --train-dir results/oneshot-example/ examples/research/surgery/nb201/deiso_plateaulr.yaml`

One can modify the configurations in `deiso_plateaulr.yaml`, for example, 1) To use `S` architecture samples in every supernet update, change `evaluator_cfg.mepa_samples` to `S`; 2) To adjust wheher or not to use de-isomorphism sampling, add `deiso: true` and the architecture list file in the controller component cfg, as follows

```
controller_type: nasbench-201-rs
controller_cfg:
  rollout_type: nasbench-201
  deiso: true
  mode: eval
```

`examples/research/surgery/nb201/run_supernet_training.sh` is a helper script to run the previous `awnas search` command, and can be run with `bash examples/research/surgery/nb201/run_supernet_training.sh <cfg_file.yaml> <seed>`.

Also, to run multiple supernet training processes using multiple different configurations in a batched manner (on multiple GPUs), check `examples/research/surgery/run_supernet_training.py`.

### Derive architectures using one-shot supernets

```
awnas derive --load results/oneshot-example/1000/ --out-file results/oneshot-example/derive_results.yaml --gpu 0 -n 6466 --test --seed [random seed] --runtime-save examples/research/surgery/nb201/deiso_derive.yaml
```

The `--runtime-save` option is optional, and it enables `awnas derive` to continue from a previously interrupted derive process.

To run multiple `derive` processes using multiple different checkpoints in a batched manner (on multiple GPUs), check `examples/research/surgery/run_derive_ckpts.py`.

### Get the evaluation results

`python examples/research/surgery/evaluation.py results/oneshot-example/derive_results.yaml --type deiso`


### Plotting the results

Coming soon

