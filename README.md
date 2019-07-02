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

`pip install -r requirements.txt`

Make sure graphiz is installed, e.g. on Ubuntu, you can run `sudo apt-get install graphviz`.

## Usage

After installation, you can run `awnas --help` to see what sub-commands are available.

### Prepare data

When running `awnas` program, it will assume the data of a dataset with `name=<NAME>` under `AWNAS_DATA/<NAME>`, in which `AWNAS_DATA` base directory is read from the environment variable `AWNAS_DATA`. If the environment variable is not specified, the default is `AWNAS_HOME/data`, in which `AWNAS_HOME` is an environment variable default to be `~/awnas`.

* Cifar10: No specific preparation needed.
* PTB: `bash scripts/get_data.sh ptb`, the ptb data will be downloaded under `${DATA_BASE}/ptb` directory. By default `${DATA_BASE}` will be `~/awnas/data`.
* Tiny-ImageNet: `bash scripts/get_data.sh tiny-imagenet`, the tiny-imagenet data will be downloaded under `${DATA_BASE}/tiny-imagenet` directory.

### Run NAS search

### Derive

## Develop New Components

TOOD (The detailed interface of different components, to write new components, these interface method must be overrided.)