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

The interface between these components are somehow well-defined. TODO (what is Rollout)

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

### Derive

## Develop New Components

TOOD (The detailed interface of different components, to write new components, these interface method must be overrided.)