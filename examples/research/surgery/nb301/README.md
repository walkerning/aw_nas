Nasbench-301
--------

### Train one-shot supernets

`examples/research/surgery/nb301/config.yaml` is the basic supernet-training configuration file on NB301. Run the following command to start a supernet-training process:

`````
awnas search --gpu 0 --seed [random seed] --save-every 50 --train-dir results/nb301/results_supernet_training examples/research/surgery/nb301/config.yaml
`````

Also, to run multiple supernet training processes using multiple different configurations in a batched manner (on multiple GPUs), check `examples/research/surgery/run_supernet_training.py`.

### Eval-arch using one-shot supernets

Given a supernet/evaluator checkpoint, and a YAML file containing a list of architecture genotypes, one can run the following command to estimate the one-shot rewards of these architectures:

```
awnas eval-arch examples/research/surgery/nb301/config.yaml archs.yaml --load [supernet/evaluator checkpoint] --dump-rollouts results/nb301/eval_results.pkl --gpu 0 --seed 123
```

The architecture file `arch.yaml` used in our paper can be found under the dir `assets/nb301/nb301_assets` at [this url](https://cloud.tsinghua.edu.cn/d/965b3ae1f80b45e9ba21/).

To run multiple `eval-arch` processes using multiple different checkpoints in a batched manner (on multiple GPUs), check `examples/research/surgery/run_derive_ckpts.py`.

### Get the evaluation results

One does not need to install the NB301 benchmark while running the previous commands. However, when calculating the evaluation results, the GT performances of architectures are needed, thus [the NB301 benchmark package](https://github.com/automl/nasbench301) needs to be installed.

After successful installation, run ``python examples/research/surgery/evaluation.py results/nb301/eval_results.pkl --type nb301`` to dump the evaluation results (e.g., Kendall's Tau, SpearmanR, P@top/bottomKs, B/WR@Ks) to `results/nb301/eval_results_statistics.pkl`.
