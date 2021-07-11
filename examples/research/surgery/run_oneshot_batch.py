# -- coding: utf-8 -*-
#pylint: disable=invalid-name

import os
import argparse
import subprocess
import multiprocessing

import yaml

GPUs = [1, 2, 6]
parser = argparse.ArgumentParser()


parser.add_argument("ckpt_dir")
parser.add_argument("--result-dir", required=True)
parser.add_argument("--batch-size", required=True, type=int)
parser.add_argument("--num-batch", default=10, type=int)
# below two default values are for nb201 eval-arch on eva10; change it to your need
parser.add_argument("--arch-file", default="archs.yaml")
parser.add_argument("--cfg-file", default=None)
args = parser.parse_args()

derive_epochs = [40, 200, 1000]
num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)
log_dir = os.path.join(args.result_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

if args.cfg_file is None:
    args.cfg_file = os.path.join(args.ckpt_dir, "derive_config.yaml")
with open(args.cfg_file, "r") as rf:
    cfg = yaml.load(rf)
cfg["evaluator_cfg"]["batch_size"] = args.batch_size
cfg["objective_cfg"]["aggregate_as_list"] = True
eval_cfg_fname = os.path.join(args.result_dir, "eval_config.yaml")
with open(eval_cfg_fname, "w") as wf:
    yaml.dump(cfg, wf)

def _worker(p_id, gpu_id, queue):
    while 1:
        epoch = queue.get()
        if epoch is None:
            break
        ckpt_dir = os.path.join(args.ckpt_dir, str(epoch))
        out_file = os.path.join(args.result_dir, "{}.pkl".format(epoch))
        eval_log = os.path.join(log_dir, "{}.log".format(epoch))
        cmd = ("awnas eval-arch {} {} --load {} --steps 10"
               "  --gpu {} --reset-dataloader-each-rollout --dump-rollouts {} >{} 2>&1").format(
                   eval_cfg_fname, args.arch_file, ckpt_dir, gpu_id, out_file, eval_log)
        print("Process #{}: GPU {} Get epoch: {}; CMD: {}".format(p_id, gpu_id, epoch, cmd))
        subprocess.check_call(cmd, shell=True)
    print("Process #{} end".format(p_id))

for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

for epoch in derive_epochs:
    queue.put(epoch)

# close all the workers
for _ in range(num_processes):
    queue.put(None)
