Nasbench-1shot1-Sub3
--------

### Prepare
Install [nasbench101](https://github.com/google-research/nasbench) and [nasbench101-1shot](https://github.com/automl/nasbench-1shot1).


### Train one-shot supernets
`examples/research/surgery/nb101/oneshot_cfg.yaml` is the basic supernet-training configuration file on NB101-1shot. Run the following command:
```
awnas search --gpu 0 --seed [random seed] --save-every 10 --train-dir results/nb101/supernet examples/research/surgery/nb101/oneshot_cfg.yaml
```

Before running the command, `nb101_14k.yaml` file should be download for sampling in training procedure.


### Derive architectures using one-shot supernets
Given a supernet/evaluator checkpoint, and a YAML file containing a list of architecture genotypes, one can run the following command to estimate the one-shot rewards of these architectures:
```
awnas eval-arch examples/research/surgery/nb101/oneshot_cfg.yaml archs.yaml --load [supernet/evaluator checkpoint] --dump-rollouts results/nb101/eval_results.pkl --gpu 0 --seed 123
```


### Get the evaluation results

`python examples/research/surgery/evaluation.py results/nb101/eval_results.pkl --type nb101`



### Note

Actually, the results reported in our paper are runned using the official codes provided by [NB101-1shot1](https://github.com/automl/nasbench-1shot1) (with a small bug fixed). There are differences between our supernet implementation and theirs:

