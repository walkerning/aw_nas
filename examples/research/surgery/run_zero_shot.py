# -- coding: utf-8 -*-
#pylint: disable=invalid-name

import os
import subprocess
import multiprocessing
import argparse
import itertools

GPUs = [1, 2, 6]
parser = argparse.ArgumentParser()
parser.add_argument("eval_cfg")
parser.add_argument("arch_file")
parser.add_argument("--result-dir", required=True)
args = parser.parse_args()

all_cfgs = list(itertools.product([20, 2020, 202020], [1, 5, 9]))
num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)
log_dir = os.path.join(args.result_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

def _worker(p_id, gpu_id, queue):
    while 1:
        cfg = queue.get()
        if cfg is None:
            break
        seed, n_batch = cfg
        out_file = os.path.join(args.result_dir, "seed{}_batch{}.pkl".format(seed, n_batch))
        eval_log = os.path.join(log_dir, "seed{}_batch{}.log".format(seed, n_batch))
        cmd = "awnas eval-arch {} {} --seed {} --steps {} --gpu {} --dump-rollouts {} >{} 2>&1"\
              .format(args.eval_cfg, args.arch_file, seed, n_batch, gpu_id, out_file, eval_log)
        print("Process #{}: GPU {} (seed: {} nbatch: {}); CMD: {}".format(
            p_id, gpu_id, seed, n_batch, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(p_id))

for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for cfg in all_cfgs:
    queue.put(cfg)

# close all the workers
for _ in range(num_processes):
    queue.put(None)
