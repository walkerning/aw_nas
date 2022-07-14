# -- coding: utf-8 -*-

import os
import subprocess
import multiprocessing
import argparse

GPUs = [0, 1, 2, 3, 4, 5, 6, 7]
parser = argparse.ArgumentParser()
parser.add_argument("--result-dir", required=True)
parser.add_argument("--seeds", default=None, type=str)
parser.add_argument("--save-every", required=True)
parser.add_argument("--load", default=None)
parser.add_argument("--gpus", default=None)
args, cfgs = parser.parse_known_args()
if args.gpus is not None:
    GPUs = [int(g) for g in args.gpus.split(",")]
cfgs = [os.path.abspath(cfg) for cfg in cfgs]
common_path = os.path.commonpath(cfgs)
os.makedirs(args.result_dir, exist_ok=True)
rel_res_dirs = [os.path.relpath(cfg, common_path).replace(".yaml", "") for cfg in cfgs]
res_dirs = [os.path.join(args.result_dir, rel_dir) for rel_dir in rel_res_dirs]
seeds = [20, 2020, 202020]
if args.seeds is not None:
    seeds = [int(s) for s in args.seeds.split(",")]

for res_dir in res_dirs:
    os.makedirs(res_dir, exist_ok=True)

num_processes = len(GPUs)
print(
    "Num process: {}. Num exp: {}. Would save to: {}".format(
        num_processes, len(cfgs), res_dirs
    )
)

queue = multiprocessing.Queue(maxsize=num_processes)


def _worker(p_id, gpu_id, queue):
    while 1:
        token = queue.get()
        if token is None:
            break
        cfg_file, res_dir, seed = token
        # os.makedirs(res_dir, exist_ok=True)
        # log_file = os.path.join(res_dir, "search_tail.log")
        load_str = "" if args.load is None else "--load {}".format(args.load)
        cmd = (
            "awnas search {} --gpu {} --seed {} --save-every {} --train-dir {} {}"
            " >/dev/null 2>&1"
        ).format(
            cfg_file, gpu_id, seed, args.save_every, res_dir, load_str
        )  # , log_file)
        print("Process #{}: cfg {}; CMD: {}".format(p_id, cfg_file, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(p_id))


for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for cfg, res_dir in zip(cfgs, res_dirs):
    for seed in seeds:
        #res_seed_dir = os.path.join(res_dir, str(seed))
        cfg_name = os.path.basename(cfg)
        cfg_name = cfg_name.replace(".yaml", "")
        res_seed_dir = os.path.join(args.result_dir, cfg_name, str(seed))
        queue.put((cfg, res_seed_dir, seed))

# close all the workers
for _ in range(num_processes):
    queue.put(None)
