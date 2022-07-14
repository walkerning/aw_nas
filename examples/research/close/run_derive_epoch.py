# pylint: disable=invalid-name,missing-docstring,redefined-outer-name
# -- coding: utf-8 -*-

import os
import copy
import argparse
import subprocess
import multiprocessing
from io import StringIO
import yaml

GPUs = [4, 5, 6, 7]

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=None)
parser.add_argument("--arch-file", default=None)
parser.add_argument("--gpus", default=None, type=str)
args, base_dirs = parser.parse_known_args()

if args.gpus is not None:
    GPUs = [int(g) for g in args.gpus.split(",")]

num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)


def _worker(p_id, gpu_id, queue):
    while 1:
        token = queue.get()
        if token is None:
            break
        # ckpt_dir, res_file = token
        cfg_file, arch_file, ckpt_dir, out_file, derive_log = token
        # call eval-arch
        cmd = (
            "awnas eval-arch {} {} --load {} --dump-rollouts {} --gpu {} --seed 123 "
            ">{} 2>&1"
        ).format(cfg_file, arch_file, ckpt_dir, out_file, gpu_id, derive_log)
        print("Process #{}: ckpt {}; CMD: {}".format(p_id, ckpt_dir, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(p_id))


for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for base_dir in base_dirs:
    cfg_file = os.path.join(base_dir, "config.yaml")
    cfg = yaml.load(open(cfg_file, "r"))
    if "eval_arch" in cfg["weights_manager_cfg"].keys():
        cfg["weights_manager_cfg"]["eval_arch"] = True

    cfg_file = os.path.join(base_dir, "config_derive.yaml")
    yaml.dump(cfg, open(cfg_file, "w"))
    
    derive_logs = os.path.join(base_dir, "derive_logs")
    if os.path.exists(derive_logs) == False:
        os.makedirs(derive_logs)

    epoch = args.epoch

    ckpt_dir = os.path.join(base_dir, epoch)
    out_file = os.path.join(base_dir, "derive_results_{}.pkl".format(epoch))
    derive_log = os.path.join(base_dir, "derive_logs" ,"{}.log".format(epoch))
    queue.put((cfg_file, args.arch_file, ckpt_dir, out_file, derive_log))

# close all the workers
for _ in range(num_processes):
    queue.put(None)
