# -- coding: utf-8 -*-

import os
import copy
import time
import yaml
import subprocess
import multiprocessing
import argparse

GPUs = [2,3,4,5,6]
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-file", required=True)
parser.add_argument("--result-dir", required=True)
args, ckpts = parser.parse_known_args()

ckpts = [os.path.abspath(ckpt) for ckpt in ckpts]
common_path = os.path.commonpath(ckpts)
res_yamls = [os.path.relpath(ckpt, common_path).strip("/") + ".yaml" for ckpt in ckpts]
os.makedirs(args.result_dir, exist_ok=True)
for res_yaml in res_yamls:
    if "/" in res_yaml:
        os.makedirs(os.path.join(args.result_dir, res_yaml.rsplit("/", 1)[0]), exist_ok=True)
print("Would save to:", res_yamls)

num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)

def _worker(p_id, gpu_id, queue):
    while 1:
        token = queue.get()
        if token is None:
            break
        ckpt_dir, res_yaml = token
        out_file = os.path.join(args.result_dir, res_yaml)
        derive_log = out_file.replace(".yaml", ".log")
        cmd = "awnas derive {} --load {} --out-file {} --gpu {} -n 6466 --test --seed 123 --runtime-save >{} 2>&1".format(args.cfg_file, ckpt_dir, out_file, gpu_id, derive_log)
        print("Process #{}: ckpt {}; CMD: {}".format(p_id, ckpt_dir, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(p_id))

for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for ckpt, res_yaml in zip(ckpts, res_yamls):
    queue.put((ckpt, res_yaml))
    # print("Put in epoch {}".format(epoch))

# close all the workers
for _ in range(num_processes):
    queue.put(None)
