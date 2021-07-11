# -- coding: utf-8 -*-
# pylint: disable=invalid-name

import os
import copy
import subprocess
import multiprocessing
import argparse
import itertools

import yaml
import numpy as np

GPUs = [4, 5, 6, 7]
parser = argparse.ArgumentParser()
parser.add_argument("eval_cfg")
parser.add_argument("arch_file")
# parser.add_argument(
#     "--evaluators", default=None,
#     help="Comma-separated paths. If specified, we do not need to try different init seeds")
parser.add_argument("--result-dir", required=True)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--num-batch", default=10, type=int)
args, evaluators = parser.parse_known_args()

n_batches = args.num_batch
os.makedirs(args.result_dir, exist_ok=True)

# data shuffle indices file paths, None means no shuffle
# since we preserve the perfs of all batches, it seems we do not need
# to experiment with different shuffle indice files now!!!
data_shuffle_indices_files = [
    None,
    # "/home/eva_share_users/foxfi/surgery/nb201/zero_shot/shuffle2020.yaml",
    # "/home/eva_share_users/foxfi/surgery/nb201/zero_shot/shuffle202020.yaml"
]

with open(args.eval_cfg, "r") as rf:
    base_cfg = yaml.load(rf)

print("!! Change batch size to {}".format(args.batch_size))
base_cfg["evaluator_cfg"]["batch_size"] = args.batch_size
# base_cfg["weights_manager_cfg"]["dropout_rate"] = 0.

different_data_cfg_files = []
for data_shuffle_file in data_shuffle_indices_files:
    cfg = copy.deepcopy(base_cfg)
    if data_shuffle_file is not None:
        cfg["evaluator_cfg"]["shuffle_data_before_split"] = True
        cfg["evaluator_cfg"]["shuffle_indice_file"] = data_shuffle_file
        can_ds_fname = (
            os.path.basename(data_shuffle_file).split(".", 1)[0].replace("-", "_")
        )
    else:
        can_ds_fname = "ori"
    cfg_fname = os.path.join(args.result_dir, "{}.yaml".format(can_ds_fname))
    different_data_cfg_files.append(cfg_fname)
    cfg["objective_cfg"]["aggregate_as_list"] = True
    cfg["objective_cfg"]["aggregate_as_list"] = True
    cfg["weights_manager_cfg"]["candidate_eval_no_grad"] = False
    with open(cfg_fname, "w") as wf:
        yaml.dump(cfg, wf)

if evaluators:
    evaluator_paths = evaluators
    all_cfgs = list(
        itertools.product(
            evaluator_paths,
            # [None, 2020, 202020], # validation data shuffle seed (derive queue)
            different_data_cfg_files,  # the cfg file names with different data shuffle indices
            [n_batches],
        )
    )
else:
    all_cfgs = list(
        itertools.product(
            [20, 2020, 202020],  # weight init seed
            different_data_cfg_files,
            [n_batches],
        )
    )

num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)
print(
    "#Cfgs: {}; #Processes: {}; #GPUs: {}".format(
        len(all_cfgs), num_processes, len(np.unique(GPUs))
    )
)


def _worker(p_id, gpu_id, queue):
    while 1:
        cfg = queue.get()
        if cfg is None:
            break
        init_seed_or_eval_path, eval_cfg_fname, n_batch = cfg
        if isinstance(init_seed_or_eval_path, str):
            seed = 20
            load_str = "--load {} --seed {}".format(init_seed_or_eval_path, seed)
            eval_name = "eval{}".format(os.path.basename(init_seed_or_eval_path))
        else:
            seed = init_seed_or_eval_path
            load_str = "--seed {}".format(seed)
            eval_name = "seed{}".format(seed)
        eval_result_dir = os.path.join(args.result_dir, eval_name)
        os.makedirs(eval_result_dir, exist_ok=True)
        # output file name and evaluation log file name
        result_name = "data{}-seed{}-batch{}".format(
            os.path.basename(eval_cfg_fname).split(".", 1)[0], seed, n_batch
        )
        out_file = os.path.join(eval_result_dir, "{}.pkl".format(result_name))
        eval_log = os.path.join(eval_result_dir, "{}.log".format(result_name))
        cmd = (
            "awnas eval-arch {} {} {} --steps {} --gpu {} --reset-dataloader-each-rollout"
            " --dump-rollouts {} >{} 2>&1"
        ).format(
            eval_cfg_fname,
            args.arch_file,
            load_str,
            n_batch,
            gpu_id,
            out_file,
            eval_log,
        )
        print("Process #{}: GPU {} ({}); CMD: {}".format(p_id, gpu_id, cfg, cmd))
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
