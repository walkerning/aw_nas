"""
Example:
Start the ray head on 4 gpus: 0, 2, 3, 4
`CUDA_VISIBLE_DEVICES=0,2,3,4 ray start --head --num-cpus 10 --num-gpus 4 --object-store-memory 10000000`

Then, for every search-derive-sur-final flow, run:
`python ray_pipeline.py --redis-addr <head redis server address> <other args>`

The redis server address of ray head will be printed by the `ray start` command.

..warning::
Currently, all remote tasks write to the filesystem directly, so if there are multiple nodes used,
make sure all these files are on a shared filesystem between these nodes.

Suppose there is a new node with a shared filesystem, to add this node into the ray cluster:
`CUDA_VISIBLE_DEICES=1,2,4,5,6 ray start --redis-address <head redis server address> --num-cpus 10 --num-gpus=5 --object-store-memory 10000000`
"""
#pylint: disable-all

from __future__ import print_function

import os
import re
import sys
import time
import random
import signal
import shutil
import argparse
import subprocess
from functools import wraps
from multiprocessing import Process

import yaml
import numpy as np
import ray
import psutil

# DERIVE_N = 10

class KillSignal(ray.experimental.signal.Signal):
    pass

@ray.remote
class Killer(object):
    def send_kill(self):
        ray.experimental.signal.send(KillSignal())
        print("finished sending kill signals, please wait for some seconds for all these tasks to exit")

def _get_gpus(gpu_ids):
    return ",".join(map(str, gpu_ids))

def _get_genotype_substr(genotypes):
    return re.search(r".+?Genotype\((.+)\)", genotypes).group(1)

def _get_perf(log, type_="cnn"):
    if type_ == "cnn":
        out = subprocess.check_output("grep -Eo 'valid_acc [0-9.]+' {}".format(log) + \
                                      " | tail -n 1 | awk '{print $NF}'", shell=True)
        acc = float(out)
        print("get perf %s: %f" %(log, acc))
        return acc
    raise NotImplementedError("unknown type: {}".format(type_))

def make_surrogate_cfgs(derive_out_file, template_file, sur_dir):
    with open(template_file, "r") as f:
        cfg_template = yaml.safe_load(f)
    with open(derive_out_file, "r") as f:
        genotypes_list = yaml.safe_load(f)
    for ind, genotypes in enumerate(genotypes_list):
        sur_fname = os.path.join(sur_dir, "{}.yaml".format(ind))
        genotypes = _get_genotype_substr(genotypes)
        cfg_template["final_model_cfg"]["genotypes"] = genotypes
        with open(sur_fname, "w") as of:
            yaml.safe_dump(cfg_template, of)

@ray.remote(num_gpus=1, max_calls=1)
def call_search(cfg, seed, train_dir, vis_dir, data_dir, save_every, killer):
    if seed is None:
        seed = random.randint(1, 999999)
    gpu = _get_gpus(ray.get_gpu_ids())
    print("train seed: %s" % str(seed))

    # setup gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["AWNAS_DATA"] = data_dir

    # call aw_nas.main::main
    cmd = ("search {cfg} --gpu 0 --seed {seed} --save-every {save_every} "
           "--train-dir {train_dir} --vis-dir {vis_dir}")\
        .format(cfg=cfg, seed=seed, save_every=save_every,
                train_dir=train_dir, vis_dir=vis_dir)
    print("CUDA_VISIBLE_DEVICES={} AWNAS_DATA={} awnas {}".format(gpu, data_dir, cmd))
    print("check {} for log".format(os.path.join(train_dir, "search.log")))
    def _run_main(*args):
        sys.stdout = open("/dev/null", "w")
        from aw_nas.main import main
        main(*args)
    proc = Process(target=_run_main, args=(re.split(r"\s+", cmd),))
    proc.start()

    # wait for proc finish or killed
    while 1:
        time.sleep(10)
        if proc.is_alive():
            sigs = ray.experimental.signal.receive([killer], timeout=1)
            if sigs:
                print("call_search: receive kill signal from killer, kill the working processes")
                process = psutil.Process(proc.pid)
                for c_proc in process.children(recursive=True):
                    c_proc.kill()
                process.kill()
                exit_status = 1
                break
        else:
            exit_status = proc.exitcode
            break
    if exit_status != 0:
        raise subprocess.CalledProcessError(exit_status, cmd)

    max_epoch = max([int(n) for n in os.listdir(train_dir) if n.isdigit()])
    final_checkpoint = os.path.join(train_dir, str(max_epoch))
    return final_checkpoint

def random_derive(cfg, seed, out_file, n):
    with open(cfg, "r") as cfg_f:
        cfg_dct = yaml.safe_load(cfg_f)
    from aw_nas.common import get_search_space
    ss = get_search_space(cfg_dct["search_space_type"], **cfg_dct["search_space_cfg"])
    with open(out_file, "w") as of:
        for i in range(n):
            rollout = ss.random_sample()
            yaml.safe_dump([str(rollout.genotype)], of)
    return out_file

# derive
@ray.remote(num_gpus=1, max_calls=1)
def call_derive(cfg, seed, load, out_file, data_dir, n, killer):
    if seed is None:
        seed = random.randint(1, 999999)
    gpu = _get_gpus(ray.get_gpu_ids())
    print("derive seed: %s" % str(seed))

    # setup gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["AWNAS_DATA"] = data_dir

    # call aw_nas.main::main
    cmd = ("derive {cfg} --load {load} --gpu 0 --seed {seed}"
           " --test -n {n} -o {out_file}")\
        .format(cfg=cfg, load=load, seed=seed, out_file=out_file, n=n)
    print("CUDA_VISIBLE_DEVICES={} AWNAS_DATA={} awnas {}".format(gpu, data_dir, cmd))
    def _run_main(*args):
        sys.stdout = open("/dev/null", "w")
        from aw_nas.main import main
        main(*args)
    proc = Process(target=_run_main, args=(re.split(r"\s+", cmd),))
    proc.start()

    # wait for proc finish or killed
    while 1:
        time.sleep(10)
        if proc.is_alive():
            sigs = ray.experimental.signal.receive([killer], timeout=1)
            if sigs:
                print("call_derive: receive kill signal from killer, kill the working processes")
                process = psutil.Process(proc.pid)
                for c_proc in process.children(recursive=True):
                    c_proc.kill()
                process.kill()
                exit_status = 1
                break
        else:
            exit_status = proc.exitcode
            break
    if exit_status != 0:
        raise subprocess.CalledProcessError(exit_status, cmd)

    return out_file

# train
@ray.remote(num_gpus=1, max_calls=1)
def call_train(cfg, seed, train_dir, data_dir, save_every, killer):
    if seed is None:
        seed = random.randint(1, 999999)
    print("train seed: %s" % str(seed))
    gpus = ray.get_gpu_ids()
    gpu = _get_gpus(gpus)
    save_str = "" if save_every is None else "--save-every {}".format(save_every)

    # setup gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["AWNAS_DATA"] = data_dir

    # call aw_nas.main::main
    cmd = ("train {cfg} --gpus {range_gpu}"
           " --seed {seed} {save_str} --train-dir {train_dir}")\
        .format(cfg=cfg, seed=seed, train_dir=train_dir, save_str=save_str,
                range_gpu=",".join(map(str, range(len(gpus)))))
    print("CUDA_VISIBLE_DEVICES={} AWNAS_DATA={} awnas {}".format(gpu, data_dir, cmd))
    print("check {} for log".format(os.path.join(train_dir, "train.log")))
    def _run_main(*args):
        sys.stdout = open("/dev/null", "w")
        from aw_nas.main import main
        main(*args)
    proc = Process(target=_run_main, args=(re.split(r"\s+", cmd),))
    proc.start()

    # wait for proc finish or killed
    while 1:
        time.sleep(10)
        if proc.is_alive():
            sigs = ray.experimental.signal.receive([killer], timeout=1)
            if sigs:
                print("call_train: receive kill signal from killer, kill the working processes")
                process = psutil.Process(proc.pid)
                for c_proc in process.children(recursive=True):
                    c_proc.kill()
                process.kill()
                exit_status = 1
                break
        else:
            exit_status = proc.exitcode
            break
    if exit_status != 0:
        raise subprocess.CalledProcessError(exit_status, cmd)

    return os.path.join(train_dir, "train.log")

STAGES = ["search", "derive", "train_surrogate", "train_final"]

## --- main ---
parser = argparse.ArgumentParser()
parser.add_argument("--redis-addr", required=True, type=str,
                    help="the redis server address of ray head")
parser.add_argument("--exp-name", required=True, type=str)
parser.add_argument("--type", default="cnn", choices=["cnn", "rnn"], type=str,
                    help="(default: %(default)s)")
parser.add_argument("--base-dir", default=os.path.abspath(os.path.expanduser("~/awnas/results")),
                    type=str,
                    help="results will be saved to `base_dir`/`exp_name` (default: %(default)s)")

parser.add_argument("--seed", default=None, type=int, help="the default seeds of all tasks, "
                    "if not specified explicitly.")

parser.add_argument("--search-cfg", required=True, type=str)
parser.add_argument("--search-seed", default=None, type=int)
parser.add_argument("--search-save-every", default=20, type=int)

parser.add_argument("--derive-seed", default=123, type=int)

parser.add_argument("--train-surrogate-cfg", type=str,
                    help="train surrogate config file (default: %(default)s)",
                    default="./examples/cnn_templates/sur_template.yaml")
parser.add_argument("--train-surrogate-seed", default=None, type=int)

parser.add_argument("--train-final-cfg", type=str,
                    help="train final config file (default: %(default)s)",
                    default="./examples/cnn_templates/final_template.yaml")
parser.add_argument("--train-final-seed", default=None, type=int)
parser.add_argument("--random-derive", default=False, action="store_true",
                    help="random derive architecture from search space, if true, "
                    "will not run search stage")
parser.add_argument("--start-stage", default="search", type=str,
                    choices=STAGES, help="Start from an intermediage stage, "
                    "by default start from the very beggining: search")
parser.add_argument("--end-stage", default="train_final", type=str,
                    choices=STAGES, help="End when after this stage, "
                    "by default end after: train_final")
parser.add_argument("--start-surrogate-index", default=0, type=int,
                    help="When START_STAGE == `train_surrogate', "
                    "can specificy start train from which surrogate cfg")
parser.add_argument("--derive-n", default=10, type=int,
                    help="Number of dervied arch after search.")
parser.add_argument("--data", type=str, default=os.path.expanduser("~/awnas/data"),
                    help="the data base dir(exclude the dataset name), "
                    "will set `AWNAS_DATA` environment variable accordingly. (default: %(default)s)")

cmd_args = parser.parse_args()
ray.init(redis_address=cmd_args.redis_addr) # connect to ray head
killer = Killer.remote() # create the killer actor
killed = 0 # killed flag for current client process

def check_killed():
    # check killed in the client process
    # cannot be used in remote fnction, which is serialized, unserialized, called in remote workers
    if killed:
        exit(1)

def terminate_procs(sig, frame):
    global killed
    print("sending kill signals, please wait for some seconds for all these tasks to exit")
    killer.send_kill.remote()
    killed = 1

def _exit():
    print("End exp {}".format(exp_name))
    sys.exit(0)

signal.signal(signal.SIGINT, terminate_procs)
signal.signal(signal.SIGTERM, terminate_procs)

cmd_args.search_cfg = os.path.abspath(cmd_args.search_cfg)
cmd_args.train_surrogate_cfg = os.path.abspath(cmd_args.train_surrogate_cfg)
cmd_args.train_final_cfg = os.path.abspath(cmd_args.train_final_cfg)

exp_name = cmd_args.exp_name
DERIVE_N = cmd_args.derive_n

# seed
if cmd_args.seed is not None:
    cmd_args.train_final_seed = cmd_args.train_final_seed if cmd_args.train_final_seed is not None else cmd_args.seed
    cmd_args.train_surrogate_seed = cmd_args.train_surrogate_seed if cmd_args.train_surrogate_seed is not None else cmd_args.seed
    cmd_args.search_seed = cmd_args.search_seed if cmd_args.search_seed is not None else cmd_args.seed
    cmd_args.derive_seed = cmd_args.derive_seed if cmd_args.derive_seed is not None else cmd_args.seed

# result dirs
result_dir = os.path.join(cmd_args.base_dir, exp_name)
search_dir = os.path.join(result_dir, "search")
sur_dir = os.path.join(result_dir, "train_surrogate")
final_dir = os.path.join(result_dir, "train_final")

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(search_dir):
    os.makedirs(search_dir)
if not os.path.exists(sur_dir):
    os.makedirs(sur_dir)
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

search_cfg = os.path.join(result_dir, "search.yaml")
train_surrogate_template = os.path.join(result_dir, "train_surrogate.template")
train_final_template = os.path.join(result_dir, "train_final.template")

# intermediate output files
# they will be repoped with the exactly the same values for dependency specification,
# if the pipeline aborts at some stages, can directly use these to start from middle
exist_search_ckpts = [int(n) for n in os.listdir(search_dir) if n.isdigit()]
if exist_search_ckpts:
    max_epoch = max(exist_search_ckpts)
    final_checkpoint = os.path.join(search_dir, str(max_epoch))
else:
    final_checkpoint = None
derive_out_file = os.path.join(search_dir, "derive.yaml")
sur_train_logs = [os.path.join(sur_dir, str(index), "train.log") for index in range(DERIVE_N)]

# start/end stage
start_stage = STAGES.index(cmd_args.start_stage)
if cmd_args.random_derive:
    start_stage = max(start_stage, 1)
    cmd_args.start_stage = STAGES[start_stage]
    print("random_derive is True, will not run search stage")
end_stage = STAGES.index(cmd_args.end_stage)
assert start_stage <= end_stage, "--start-stage must be before --end-stage"
print("Start the NAS pipeline from stage: {}".format(cmd_args.start_stage))
print("End the NAS pipeline after stage: {}".format(cmd_args.end_stage))

shutil.copy(cmd_args.search_cfg, search_cfg)
shutil.copy(cmd_args.train_surrogate_cfg, train_surrogate_template)
shutil.copy(cmd_args.train_final_cfg, train_final_template)
if cmd_args.start_stage != "train_surrogate":
    assert cmd_args.start_surrogate_index == 0, \
        "Can only specificy STRAT_SURROGATE_INDEX when START_STAGE"\
        " == `train_surrogate' ({})".format(cmd_args.start_stage)
try:
    if start_stage <= 0: # search
        # search
        vis_dir = os.path.join(result_dir, "vis")
        final_checkpoint = call_search.remote(search_cfg, cmd_args.search_seed,
                                              search_dir, vis_dir,
                                              cmd_args.data, cmd_args.search_save_every, killer)
        final_checkpoint = ray.get(final_checkpoint)
        check_killed()
    if end_stage <= 0: # search
        _exit()

    if start_stage <= 1: # derive
        # derive
        if cmd_args.random_derive:
            derive_out_file = random_derive(search_cfg, cmd_args.derive_seed,
                                            derive_out_file, DERIVE_N)
        else:
            derive_out_file = call_derive.remote(search_cfg, cmd_args.derive_seed,
                                                 final_checkpoint, derive_out_file,
                                                 cmd_args.data, DERIVE_N, killer)
            derive_out_file = ray.get(derive_out_file)

        check_killed()
    if end_stage <= 1: # derive
        _exit()

    if start_stage <= 2: # train_surrogate
        # make surrogate cfgs
        make_surrogate_cfgs(derive_out_file, train_surrogate_template, sur_dir)
        
        check_killed()

        # train surrogate
        runned_sur_train_logs = []
        for index in range(cmd_args.start_surrogate_index, DERIVE_N):
            sur_fname = os.path.join(sur_dir, "{}.yaml".format(index))
            train_sur_dir = os.path.join(sur_dir, str(index))
            runned_sur_train_logs.append(call_train.remote(sur_fname, cmd_args.train_surrogate_seed,
                                                           train_sur_dir, cmd_args.data,
                                                           save_every=None, killer=killer))

        runned_sur_train_logs = ray.get(runned_sur_train_logs)
        sur_train_logs = sur_train_logs[:cmd_args.start_surrogate_index] + runned_sur_train_logs

        check_killed()
    if end_stage <= 2: # train_surrogate
        _exit()

    # if start_stage <= 3: # always true
    # choose best
    sur_perfs = [_get_perf(log, type_=cmd_args.type) for log in sur_train_logs]
    best_ind = np.argmax(sur_perfs)
    print("best surrogate performances: {}; ({})".format(sur_perfs[best_ind], sur_perfs))
    with open(derive_out_file, "r") as f:
        genotypes_list = yaml.safe_load(f)
    best_geno = _get_genotype_substr(genotypes_list[best_ind])
    with open(os.path.join(sur_dir, "sur_res.txt"), "w") as of:
        of.write("\n".join(["{} {}".format(ind, perf)
                            for ind, perf in
                            sorted(list(enumerate(sur_perfs)), key=lambda item: -item[1])]))

    check_killed()

    # dump configuration of final train
    with open(train_final_template, "r") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["final_model_cfg"]["genotypes"] = best_geno
    train_final_cfg = os.path.join(final_dir, "train.yaml")
    with open(train_final_cfg, "w") as of:
        yaml.safe_dump(base_cfg, of)

    check_killed()

    # train final
    total_epochs = base_cfg["final_trainer_cfg"]["epochs"]
    train_final_dir = os.path.join(final_dir, "train")
    final_log = ray.get(call_train.remote(train_final_cfg, cmd_args.train_final_seed,
                                          train_final_dir, cmd_args.data,
                                          save_every=total_epochs // 4, killer=killer))
    final_valid_perf = _get_perf(final_log, type_=cmd_args.type)
    # if end_stage <= 3: # always true
    _exit()
except ray.exceptions.RayTaskError as e:
    print("EXIT! exp {}: Exception when executing task: ".format(exp_name), e)
    sys.exit(1)

