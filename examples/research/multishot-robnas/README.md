## Multi-shot NAS for Discovering Adversarially Robust Convolutional Neural Architectures at Targeted Capacities
--------

If you find this work/code helpful, please cite:
```
@article{ning2020multi,
  title={Multi-shot NAS for Discovering Adversarially Robust Convolutional Neural Architectures at Targeted Capacities},
  author={Ning, Xuefei and Zhao, Junbo and Li, Wenshuo and Zhao, Tianchen and Zheng, Yin and Yang, Huazhong and Wang, Yu},
  journal={arXiv preprint arXiv:2012.11835},
  year={2020}
}
```

The adversarial robustness plugin is in `../../plugins/robustness/`. There are several python files and `cfgs` dir under this directory. The python files are:

* `1_robustness.py`: Adversarial robustness objectives, adversarial attacks. Note that the diff supernet defined in this file is not used, it is for the original DARTS/ENAS search space. However, we use a dense micro search space that allows densely-connected pattern (each node can choose an arbitrary number of input nodes, instead of only 2).
* `robustness_ss.py`: Search space, rollout.
* `robustness_weights_manager.py`: The parameter-sharing weights manager of the dense cell search space. The cache is used for reusing the generated adversarial examples for the same images if no weights are updated, otherwise `awnas` will meaninglessly repeat several nearly-identical adversarial example generation for metric reporting and loss calculation. Note that upon each weight update, the cache is cleared (see `step_current_gradients` and `step`). The BN calibration is also implemented in this file (search for `calib_bn`).
* `robustness_final_model`: Final model.
* `2_multi_evaluator.py`: The multi-shot evaluator that uses interpolation to estimate reward at a targeted capacity.
* `robustness_arch_embedder.py`: LSTM/GATES topology encoder.
* `test_distance_attack.py`: Run distance attacks on trained final models. Generate appropriate test configurations and run `awnas test`.
* `test_acc_attack.py`: Run PGD/FGSM attacks on trained final models. Generate appropriate test configurations and run `awnas test`.
* `test_black_attack.py`: Run blackbox PGD/FGSM attacks on trained final models. Generate appropriate test configurations and run `awnas test`.
* `test_rob_ss.py`: A unit test script that will not be ignored by awnas plugin mechanism. However, one can run `pytest -x test_rob_ss.py` to conduct some unit testing.


### Environmental Setup
First, you should install `awnas`. Then, to use the adversarial robustness plugin codes, you can soft link the robustness plugin to a position under the directory `${AWNAS_HOME}/plugins` (If the environment variable `AWNAS_HOME` is not specified, it is by default `${HOME}/awnas/`):
```
ln -s `readlink -f ../../plugins/robustness` ${HOME}/awnas/plugins
```

If you are using multiple versions of plugin codes, you can use an alternative directory as the AWNAS home directory for your experimental runs.
```
mkdir my_workspace
mkdir -p my_workspace/awnas/plugins
ln -s ${HOME}/awnas/data my_workspace/awnas # soft link the data dir
ln -s `readlink -f ../../plugins/robustness` my_workspace/awnas/plugins/ # soft link the plugin dir to make it active
export AWNAS_HOME=`readlink -f my_workspace/awnas` # a relative path to your later working path can also be used
```

After these soft linking and environment setting, `awnas` will load plugins from `my_workspace/awnas/plugins` instead of `${HOME}/awnas/plugins`.

### Run supernet adversarial training
As described in our paper, the first step of our working flow is adversarially training several supernets with different initial channel numbers. 

We provide an example configuration file to adversarially train a supernet with initial channel number 24 using seven-step PGD at `../../plugins/robustness/cfgs/example_supernet_training.yaml`. 

By simply modifying the `init_channels` defined in  `weights_manager_cfg`, the supernet with specified initial channel numbers can be trained.

To run the training process, simply run `bash run_cfg.sh ../../plugins/robustness/cfgs/example_supernet_training.yaml`. The logs and results are saved under directory `results_search/<exp_name>` by default, and `<exp_name>` will be the file-extension-stripped basename of the configuration file by default (in this case `example_supernet_training`). Check the `run_cfg.sh` script for details.

> NOTE: Since the supernet training stage and the search stage are decoupled in thw workflow. One must use the same training/validation data splits in two stages. You can simply choose to set `shuffle_data_before_split: false` in configuration files of both stages (which I have already set as default in the example configurations).
Nevertheless, if you'd like to shuffle the dataset before split it into training/validation splits, you should use the same `shuffle_seed`, or use the same `shuffle_indice_file`. [This indice file](https://cloud.tsinghua.edu.cn/f/fa8e252640ee4c03bd30/?dl=1) is used in our paper.

----
### Run predictor-based search
As described in our paper, although the FGSM attack is not suitable for the adversarial training phase, it makes a reasonable proxy objective of using PGD attack during the search phase. And the predictor-based search is conducted in the stagewise search space, and thanks to several evaluation proxies (objective proxy: PGD-FGSM, dataset proxy: half validation data queue) and the better exploration (smaller search space, and better predictor-based search strategy), we regard the predictor-based search flow to be more effective.

We provide sample predictor-based search configuration files (on stagewise search space) at `../../plugins/robustness/cfgs/stagewise_predbased/search-*.yaml`. The configuration file name contains information of different NAS components:

* `search-24c_stagewisess_pgd_gates.yaml`:
  * Evaluation: 24-channel supernet evaluation
  * Search space: stagewise
  * Objective: PGD attack
  * Predictor (in Controller): GATES
* `search-multishoteval_stagewisess_pgd_gates_2000M.yaml`
  * Evaluation: Multishot evaluation targeting at 2000M FLOPs
  * Search space: stagewise
  * Objective: PGD attack
  * Predictor (in Controller): GATES
* `search-multishoteval_stagewisess_fgsm_gates_2000M.yaml`
  * Evaluation: Multishot evaluation targeting at 2000M FLOPs
  * Search space: stagewise
  * Objective: FGSM attack
  * Predictor (in Controller): GATES

To run a multi-shot search targeting at 2000M using the FGSM proxy attack, simply run `bash run_cfg.sh ../../plugins/robustness/search-multishoteval_stagewisess_fgsm_gates_2000M.yaml`. The logs and results are saved under directory `results_search/<exp_name>` by default, and `<exp_name>` will be the file-extension-stripped basename of the configuration file by default (in this case, `search-multishoteval_stagewisess_fgsm_gates_2000M`). Check the `run_cfg.sh` script for details.

### Predictor-based diagnoistics

The `test_pred.py` script is a predictor diagnostic script. You can also check the comments in the script. It prints the reward/acc_clean/acc_adv/predicted score range of the rollouts sampled in each stage (50 topologies per stage; 4 stage random sample, 8 stage predictor-based sample). Also, for the predictor Pi trained at each stage i (from 4 to 12), we calculate the correlation of the Pi predicted scores and actual multi-shot reward on topologies from 12 stages. Try run `python test_pred.py results_search/search-multishoteval_stagewisess_fgsm_gates_2000M` after the search. Example output logs are also uploaded [here](https://cloud.tsinghua.edu.cn/d/cd4cb962a60343ff9f55/). And we can see that predictor-based search indeed help: No matter what evaluation strategy is used (multi-shot, one-shot), the topology rewards range is more and more concentrated towards higher values. Several facts about the predictor-based search that are not emphasized in our paper are:

* One should use the evolutionary inner search method instead of the random sample inner search method. During our experiment, we first use the random-sample inner search method. This weak inner search method will fail to find topologies with higher predicted scores, not to mention that we also expect these topologies to have higher rewards if the predictor is appropriately trained.
* The cellwise search space is too large for a predictor to make good predictions out of just 200 initial architectures (50x4). This is also one of the reasons why we choose a stagewise search space in the accelerated search flow. Fortunately, we can still reuse the K trained supernets, since the stagewise search space is a sub-search-space of the cellwise one.
* The graph-based (GATES) encoder is indeed more suitable for topological search spaces (e.g., our search space) than the LSTM encoder (All defined in `robustness_archembedder.py`).

----
### Run final model adversarial training
We provide several example configuration files to adversarially train models using seven-step PGD at `../../plugins/robustness/cfgs/final/train-*.yaml`. 
* `train-cifar100-resnet18.yaml`:
  * Model: ResNet18
  * Dataset: CIFAR-100
* `train-cifar10-MSRobNet-1000.yaml`
  * Model: MSRobNet-1000
  * Dataset: CIFAR-10

Other example configuration files are uploaded [here](https://cloud.tsinghua.edu.cn/d/e94e4e3760004da9a141/). And some of the checkpoints are uploaded [here](https://cloud.tsinghua.edu.cn/d/2c174606b5fd431c84c0/).

To run a final adversarial training, use the `run_final_cfg.sh` script. For example, run `bash run_final_cfg.sh ../../plugins/robustness/cfgs/final/train-cifar10-MSRobNet-1000.yaml`. The logs and results are saved under directory `results_search/<exp_name>` by default, and `<exp_name>` will be the file-extension-stripped basename of the configuration file by default (in this case, `train-cifar10-MSRobNet-1000`). Check the `run_final_cfg.sh` script for details.

----
### Run final model testing
* We provide convenient scripts `../../plugins/robustness/test_acc_attack.py` for testing final models with PGD/FGSM attack.
* We provide convenient scripts `../../plugins/robustness/test_distance_attack.py` for testing final models with the distance attack. 
* We provide convenient scripts `../../plugins/robustness/test_black_attack.py` for testing final models with blackbox attack.

Run `python ../../plugins/robustness/test_*_attack.py --help` for details.
