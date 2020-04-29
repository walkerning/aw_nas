# aw_nas

## Introduction

### NAS

### Components of a NAS system

There are multiple actors that are working together in a NAS system, and they can be categorized into these components:

* search space
* controller
* weights manager
* evaluator
* objective


The interface between these components are somehow well-defined. We use a class `awnas.rollout.base.BaseRollout` to represent the interface object between all these components. Usually, a search space defines one or more rollout types (a subclass of `BaseRollout`). For example, the basic cell-based search space `cnn` (class `awnas.common.CNNSearchSpace`) corresponds to two rollout types: `discrete`discrete rollouts that are used in RL-based, EVO-based controllers, etc. (class `awnas.rollout.base.Rollout`); `differentiable` differentiable rollouts that are used in gradient-based NAS (class `awnas.rollout.base.DifferentiableRollout`).

*TODO*: need a supported components list?

## Install

Using virtual python environment is encouraged. For example, with Anaconda, you could run `conda create -n awnas python==3.7.3 pip` first.

To install `awnas`, run `pip install -r requirements.txt`.

Architecture plotting relies on the `graphviz` package, make sure `graphiz` is installed, e.g. on Ubuntu, you can run `sudo apt-get install graphviz`.

## Usage

After installation, you can run `awnas --help` to see what sub-commands are available.

Output of an example run (version 0.3.dev0):

```
02/18 12:28:53 PM btc              WARNING: Error importing module nasbench: No module named 'nasbench'
Should install the NASBench 101 package following https://github.com/google-research/nasbench
02/18 12:28:53 PM btc              WARNING: Error importing module nasbench_201: No module named 'nas_201_api'
Should install the NASBench 201 package following https://github.com/D-X-Y/NAS-Bench-201
02/18 12:28:53 PM plugin              INFO: Check plugins under /home/foxfi/awnas/plugins
02/18 12:28:53 PM plugin              INFO: Loaded plugins:
Usage: awnas [OPTIONS] COMMAND [ARGS]...

  The awnas NAS framework command line interface. Use `AWNAS_LOG_LEVEL`
  environment variable to modify the log level.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  search                   Searching for architecture.
  random-sample            Random sample architectures.
  sample                   Sample architectures, pickle loading controller...
  eval-arch                Eval architecture from file.
  derive                   Derive architectures.
  train                    Train an architecture.
  test                     Test a final-trained model.
  gen-sample-config        Dump the sample configuration.
  gen-final-sample-config  Dump the sample configuration for final training.
```

### Prepare data

When running `awnas` program, it will assume the data of a dataset with `name=<NAME>` under `AWNAS_DATA/<NAME>`, in which `AWNAS_DATA` base directory is read from the environment variable `AWNAS_DATA`. If the environment variable is not specified, the default is `AWNAS_HOME/data`, in which `AWNAS_HOME` is an environment variable default to be `~/awnas`.

* Cifar10: No specific preparation needed.
* PTB: `bash scripts/get_data.sh ptb`, the ptb data will be downloaded under `${DATA_BASE}/ptb` directory. By default `${DATA_BASE}` will be `~/awnas/data`.
* Tiny-ImageNet: `bash scripts/get_data.sh tiny-imagenet`, the tiny-imagenet data will be downloaded under `${DATA_BASE}/tiny-imagenet` directory.

### Run NAS search

Try running a ENAS search (the results (including configuration backup, search log) in `<TRAIN_DIR>`):

```
awnas search examples/enas.yaml --gpu 0 --save-every 10 --train-dir <TRAIN_DIR>
```

There are several sections in the configuration file that describe the configurations of different components in the NAS framework. For example, in `examples/enas.yaml`, different configuration sections are organized as follows:

1. a cell-based CNN **search space**: The search space is an extended version from the 5-primitive micro search space in the original ENAS paper.
2. cifar-10 **dataset**
3. RL-learned **controller** with the `embed_lstm` RNN network
4. shared weights based **evaluator**
5. shared weights based **weights manager**: super net
6. classification **objective**
7. **trainer**: the orchestration of the overall NAS search flow

To generate a sample configuration file for searching, try ``awnas gen-sample-config`` utility. For example, if you want a sample search configuration for searching on NAS-Bench-101, run

```
awnas gen-sample-config -r nasbench-101 -d image ./sample_nb101.yaml
```

Then, check the `sample_nb101.yaml` file, for all the component types, all choices that declare to support the `nasbench-101` rollout type would be listed in the file. Delete what you do not need, uncomment what you need, change the default settings, and then that config can be used to run NAS on NAS-Bench-101.

### Derive

```
awnas derive
```

### Final Training of Cell-based Architecture

`awnas.final` sub-package provides the final training functionality of cell-based architectures. `examples/cnn_templates/final_template.yaml` is a commonly-used configuration template for final training a cell-based architecture. To use that template, fill the ``final_model_cfg.genotypes` field with the genotype string derived from the search process. A genotype string example is
```
CNNGenotype(normal_0=[('dil_conv_3x3', 1, 2), ('skip_connect', 1, 2), ('sep_conv_3x3', 0, 3), ('sep_conv_3x3', 2, 3), ('skip_connect', 3, 4), ('sep_conv_3x3', 0, 4), ('sep_conv_5x5', 1, 5), ('sep_conv_5x5', 0, 5)], reduce_1=[('max_pool_3x3', 0, 2), ('dil_conv_5x5', 0, 2), ('avg_pool_3x3', 1, 3), ('avg_pool_3x3', 2, 3), ('sep_conv_5x5', 1, 4), ('avg_pool_3x3', 1, 4), ('sep_conv_3x3', 1, 5), ('dil_conv_5x5', 3, 5)], normal_0_concat=[2, 3, 4, 5], reduce_1_concat=[2, 3, 4, 5])
```

### Plugin mechanism


## Hardware related: Hardware profiling and parsing

TODO: insert a good-looking figure, and describe the interface files

`aw_nas` provide a command-line interface `awnas-hw` to orchestrate the hardware-related objective (e.g., latency, energy, etc.) profiling and parsing flow. A complete workflow example is illustrated as follows.

TODO: a complete step-by-step example

`BaseHardwareObjectiveModel`

### Implement the interface for new search spaces
We provide a mixin class `ProfilingSearchSpace`. This interface has two methods that must be implemented:
* `generate_profiling_primitives`: profiling cfgs => return the profiling primitive list
* `parse_profiling_primitives`: primitive hw-related objective list, profiling/hwobj model cfgs => hwobj model

You might need to implement the hardware-related objective model class for the new search space. You can reuse some codes in `aw_nas/hardware/hw_obj_models`.

### Implement the interface for new hardwares
To implement hardware-specific compilation and parsing process, create a new class inheriting `BaseHardwareCompiler`, implement the `compile` and `hwobj_net_to_primitive` methods. As stated before, you can put your new hardware implementation python file into the `AWNAS_HOME/plugins`, to make it accessible by `aw_nas`.

## Develop New Components

TOOD (The detailed interface of different components, to write new components, these interface method must be overrided.)
