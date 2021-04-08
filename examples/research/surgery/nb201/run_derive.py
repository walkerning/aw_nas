# -- coding: utf-8 -*-

import os
import copy
import time
import yaml
import subprocess
import multiprocessing
import argparse

GPUs = [5, 5, 5, 5, 7, 7, 7, 7, 3, 6]
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir")
parser.add_argument("--result-dir", required=True)
parser.add_argument("--iso", default=False, action="store_true")
args = parser.parse_args()

# Generate derive config file, add "avoid_repeat: true"
derive_cfg_fname = os.path.join(args.ckpt_dir, "derive_config.yaml")
if not os.path.exists(derive_cfg_fname):
    with open(os.path.join(args.ckpt_dir, "config.yaml"), "r") as rf:
        search_cfg = yaml.load(rf)
    derive_cfg = copy.deepcopy(search_cfg)
    derive_cfg["controller_cfg"]["avoid_repeat"] = True
    with open(derive_cfg_fname, "w") as wf:
        yaml.dump(derive_cfg, wf)

derive_epochs = list(range(40, 1001, 40))
num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)
log_dir = os.path.join(args.result_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

def _worker(p_id, gpu_id, queue):
    while 1:
        epoch = queue.get()
        if epoch is None:
            break
        ckpt_dir = os.path.join(args.ckpt_dir, str(epoch))
        out_file = os.path.join(args.result_dir, "{}.yaml".format(epoch))
        derive_log = os.path.join(log_dir, "{}.log".format(epoch))
        arch_num = 15625 if args.iso else 6466
        cmd = ("awnas derive {} --load {} --out-file {} --gpu {} -n {} --test --seed 123"
               " --runtime-save >{} 2>&1").format(
                   derive_cfg_fname, ckpt_dir, out_file, gpu_id, arch_num, derive_log)
        print("Process #{}: GPU {} Get epoch: {}; CMD: {}".format(p_id, gpu_id, epoch, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(p_id))

for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for epoch in derive_epochs:
    queue.put(epoch)
    # print("Put in epoch {}".format(epoch))

# close all the workers
for _ in range(num_processes):
    queue.put(None)
