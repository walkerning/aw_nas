from __future__ import print_function

import os
import re
import random
import shutil
import logging
import argparse
import subprocess

import yaml
import numpy as np

DERIVE_N = 10

def _get_genotype_substr(genotypes):
    return re.search(r".+?Genotype\((.+)\)", genotypes).group(1)

def _get_perf(log, type_="cnn"):
    if type_ == "cnn":
        out = subprocess.check_output("grep -Eo 'valid_acc [0-9.]+' {}".format(log) + \
                                      " | tail -n 1 | awk '{print $NF}'", shell=True)
        logging.info(out)
        acc = float(out)
        return acc
    raise NotImplementedError("unknown type: {}".format(type_))

def call_search(cfg, gpu, seed, train_dir, vis_dir, save_every):
    if seed is None:
        seed = random.randint(1, 999999)
    logging.info("train seed: %s", str(seed))
    logging.info(("awnas search {cfg} --gpu {gpu} --seed {seed} --save-every {save_every} "
                  "--train-dir {train_dir} --vis-dir {vis_dir}")\
                 .format(cfg=cfg, gpu=gpu, seed=seed,
                         train_dir=train_dir, vis_dir=vis_dir, save_every=save_every))

    subprocess.check_call(("awnas search {cfg} --gpu {gpu} --seed {seed} --save-every {save_every} "
                           "--train-dir {train_dir} --vis-dir {vis_dir}")\
                          .format(cfg=cfg, gpu=gpu, seed=seed,
                                  train_dir=train_dir, vis_dir=vis_dir, save_every=save_every),
                          shell=True)

# derive
def call_derive(cfg, gpu, seed, load, out_file, n):
    if seed is None:
        seed = random.randint(1, 999999)
    logging.info("train seed: %s", str(seed))
    logging.info(("awnas derive {cfg} --load {load} --gpu {gpu} --seed {seed}"
                  " --test -n {n} -o {out_file}")\
                 .format(cfg=cfg, load=load, gpu=gpu, seed=seed,
                         out_file=out_file, n=n))

    subprocess.check_call(("awnas derive {cfg} --load {load} --gpu {gpu} --seed {seed}"
                           " --test -n {n} -o {out_file}")\
                          .format(cfg=cfg, load=load, gpu=gpu, seed=seed,
                                  out_file=out_file, n=n),
                          shell=True)

# train
def call_train(cfg, gpu, seed, train_dir, save_every):
    if seed is None:
        seed = random.randint(1, 999999)
    logging.info("train seed: %s", str(seed))
    save_str = "" if save_every is None else "--save-every {}".format(save_every)
    logging.info(("awnas train {cfg} --gpus {gpu} --seed {seed} {save_str} "
                  "--train-dir {train_dir}")\
                 .format(cfg=cfg, gpu=gpu, seed=seed,
                         train_dir=train_dir, save_str=save_str))

    subprocess.check_call(("awnas train {cfg} --gpus {gpu} --seed {seed} {save_str} "
                           "--train-dir {train_dir}")\
                          .format(cfg=cfg, gpu=gpu, seed=seed,
                                  train_dir=train_dir, save_str=save_str),
                          shell=True)

def make_surrogate_cfgs(derive_out_file, template_file, sur_dir):
    with open(template_file, "r") as f:
        cfg_template = yaml.load(f)
    with open(derive_out_file, "r") as f:
        genotypes_list = yaml.load(f)
    for ind, genotypes in enumerate(genotypes_list):
        sur_fname = os.path.join(sur_dir, "{}.yaml".format(ind))
        genotypes = _get_genotype_substr(genotypes)
        cfg_template["final_model_cfg"]["genotypes"] = genotypes
        with open(sur_fname, "w") as of:
            yaml.safe_dump(cfg_template, of)

def get_sur_perfs(sur_dir):
    final_perfs = []
    for ind in range(DERIVE_N):
        surrogate_dir = os.path.join(sur_dir, str(ind))
        log = os.path.join(surrogate_dir, "train.log")
        final_perfs.append(_get_perf(log, type_=args.type))
    return final_perfs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", required=True)
parser.add_argument("--exp-name", required=True, type=str)
parser.add_argument("--type", default="cnn", choices=["cnn", "rnn"], type=str, help="(default: %(default)s)")
parser.add_argument("--base-dir", default=os.path.abspath(os.path.expanduser("~/awnas/results")),
                    type=str, help="results will be saved to `base_dir`/`exp_name` (default: %(default)s)")

parser.add_argument("--seed", type=int, help="the default seeds of all tasks, "
                    "if not specified explicitly.")

parser.add_argument("--search-cfg", required=True, type=str)
parser.add_argument("--search-memory", default=6000, type=int)
parser.add_argument("--search-util", default=30, type=int)
parser.add_argument("--search-seed", default=None, type=int)
parser.add_argument("--search-save-every", default=20, type=int)

parser.add_argument("--derive-memory", default=3000, type=int)
parser.add_argument("--derive-util", default=0, type=int)
parser.add_argument("--derive-seed", default=123, type=int)

parser.add_argument("--train-surrogate-cfg", required=True, type=str, help="train surrogate config file")
parser.add_argument("--train-surrogate-memory", default=6000, type=int)
parser.add_argument("--train-surrogate-util", default=0, type=int)
parser.add_argument("--train-surrogate-seed", default=None, type=int)

parser.add_argument("--train-final-cfg", required=True, type=str, help="train final config file")
parser.add_argument("--train-final-memory", default=10000, type=int)
parser.add_argument("--train-final-util", default=70, type=int)
parser.add_argument("--train-final-seed", default=None, type=int)

args = parser.parse_args()
args.search_cfg = os.path.abspath(args.search_cfg)
args.train_surrogate_cfg = os.path.abspath(args.train_surrogate_cfg)
args.train_final_cfg = os.path.abspath(args.train_final_cfg)

gpu = args.gpu
exp_name = args.exp_name

# result dirs
result_dir = os.path.join(args.base_dir, exp_name)
search_dir = os.path.join(result_dir, "search")
sur_dir = os.path.join(result_dir, "train_surrogate")
final_dir = os.path.join(result_dir, "train_final")

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir))
    os.makedirs(search_dir)
    os.makedirs(sur_dir)
    os.makedirs(final_dir)

search_cfg = os.path.join(result_dir, "search.yaml")
train_surrogate_template = os.path.join(result_dir, "train_surrogate.template")
train_final_template = os.path.join(result_dir, "train_final.template")
shutil.copy(args.search_cfg, search_cfg)
shutil.copy(args.train_surrogate_cfg, train_surrogate_template)
shutil.copy(args.train_final_cfg, train_final_template)

# # search
vis_dir = os.path.join(result_dir, "vis")
call_search(search_cfg, gpu, args.search_seed, search_dir, vis_dir, args.search_save_every)

# derive
max_epoch = max([int(n) for n in os.listdir(search_dir) if n.isdigit()])
final_checkpoint = os.path.join(search_dir, str(max_epoch))
derive_out_file = os.path.join(search_dir, "derive.yaml")
call_derive(search_cfg, gpu, args.derive_seed, final_checkpoint, derive_out_file, DERIVE_N)

# make surrogate cfgs
make_surrogate_cfgs(derive_out_file, train_surrogate_template, sur_dir)

# train surrogate
for index in range(DERIVE_N):
    sur_fname = os.path.join(sur_dir, "{}.yaml".format(index))
    train_sur_dir = os.path.join(sur_dir, str(index))
    call_train(sur_fname, gpu, args.train_surrogate_seed, train_sur_dir, save_every=None)

# choose best
sur_perfs = get_sur_perfs(sur_dir)
best_ind = np.argmax(sur_perfs)
with open(derive_out_file, "r") as f:
    genotypes_list = yaml.load(f)
best_geno = _get_genotype_substr(genotypes_list[best_ind])
with open(os.path.join(sur_dir, "sur_res.txt"), "w") as of:
    of.write("\n".join(["{} {}".format(ind, perf)
                        for ind, perf in
                        sorted(list(enumerate(sur_perfs)), key=lambda item: -item[1])]))

# dump configuration of final train
with open(train_final_template, "r") as f:
    base_cfg = yaml.load(f)
base_cfg["final_model_cfg"]["genotypes"] = best_geno
train_final_cfg = os.path.join(final_dir, "train.yaml")
with open(train_final_cfg, "w") as of:
    yaml.safe_dump(base_cfg, of)

# train final
total_epochs = base_cfg["final_trainer_cfg"]["epochs"]
train_final_dir = os.path.join(final_dir, "train")
call_train(train_final_cfg, gpu, args.train_final_seed, train_final_dir, save_every=total_epochs // 4)
log = os.path.join(train_final_dir, "train.log")
final_valid_perf = _get_perf(log, type_=args.type)
