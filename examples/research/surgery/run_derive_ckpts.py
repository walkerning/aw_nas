# pylint: disable=invalid-name,missing-docstring,redefined-outer-name
# -- coding: utf-8 -*-

import os
import copy
import argparse
import subprocess
import multiprocessing
from io import StringIO
import yaml

GPUs = [0, 1, 2, 3]
_HOME_DIR = os.path.abspath(os.path.expanduser(os.environ.get("AWNAS_HOME", "~/awnas")))

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-file", default=None)
parser.add_argument("--gpus", default=None)
parser.add_argument(
    "--cfg-ckpt-rel-path",
    default=None,
    help="The relative path of cfg file to ckpt dir",
)
parser.add_argument("--arch-file", default=None)
parser.add_argument(
    "--arch-file-ckpt-rel-path",
    default=None,
    help="The relative path of arch file to ckpt dir",
)
parser.add_argument("--result-dir", required=True)
parser.add_argument("--single-result-file", default=None)
parser.add_argument(
    "--subset",
    default=False,
    action="store_true",
    help="Only applicable when using derive (nb201)",
)
parser.add_argument(
    "--addi-controller-cfgs",
    default="",
    help="Add controller cfgs by yaml dict, " "Only relevant when using derive (nb201)",
)
parser.add_argument(
    "--iso",
    default=False,
    action="store_true",
    help="Only applicable when using derive (nb201)",
)
args, ckpts = parser.parse_known_args()
if args.gpus:
    GPUs = [int(gpu) for gpu in args.gpus.split(",")]

ckpts = [os.path.abspath(ckpt) for ckpt in ckpts]


def get_arch_files(ckpts, arch_file, rel_path):
    assert arch_file is not None or rel_path is not None, (
        "Use `eval-arch` for Nasbench-301/101, should provide --arch-file or"
        " --arch-file-ckpt-rel-path"
    )
    if not rel_path:
        arch_files = [arch_file] * len(ckpts)
    else:
        arch_files = [os.path.join(ckpt, rel_path) for ckpt in ckpts]
    return arch_files


arch_files = [None] * len(ckpts)

assert (
    args.cfg_file is not None or args.cfg_ckpt_rel_path is not None
), "Should provide --cfg-file or --cfg-ckpt-rel-path"

if args.addi_controller_cfgs:
    addi_controller_cfgs = yaml.load(StringIO(args.addi_controller_cfgs))
else:
    addi_controller_cfgs = None

if not args.cfg_ckpt_rel_path:
    with open(args.cfg_file, "r") as r_f:
        eval_or_derive_cfg = yaml.load(r_f)

    call_derive = False  # call derive or eval_arch
    derive_cfg_file = args.cfg_file
    if eval_or_derive_cfg["search_space_type"] == "nasbench-201":
        call_derive = True
        if addi_controller_cfgs:
            derive_cfg = copy.deepcopy(eval_or_derive_cfg)
            derive_cfg["controller_cfg"].update(addi_controller_cfgs)
            derive_cfg_file = "{}_derive.yaml".format(args.cfg_file.rsplit(".", 1)[0])
            with open(derive_cfg_file, "w") as wf:
                yaml.dump(derive_cfg, wf)
        else:
            derive_cfg = eval_or_derive_cfg

        if args.subset:
            if derive_cfg["controller_cfg"]["text_file"]:
                with open(derive_cfg["controller_cfg"]["text_file"], "r") as rf2:
                    arch_num = len(rf2.read().strip().split("\n"))
            else:
                arch_num = 15625 if args.iso else 6466
        else:
            # derive 6466 or 15625
            if not args.iso:
                derive_cfg["controller_cfg"]["text_file"] = os.path.join(
                    _HOME_DIR, "awnas/data/nasbench-201/non-isom.txt"
                )
                arch_num = 6466
            else:
                derive_cfg["controller_cfg"]["text_file"] = os.path.join(
                    _HOME_DIR, "awnas/data/nasbench-201/iso.txt"
                )
                arch_num = 15625
    else:
        arch_num = None  # call eval-arch would not be used (nb301/101)
    cfg_files = [derive_cfg_file] * len(ckpts)
    num_archs = [arch_num] * len(ckpts)
    call_derives = [call_derive] * len(ckpts)
else:
    cfg_files = []
    call_derives = []
    num_archs = []
    for ckpt in ckpts:
        cfg_file = os.path.join(ckpt, args.cfg_ckpt_rel_path)
        derive_cfg_file = cfg_file
        arch_num = None
        with open(cfg_file, "r") as r_f:
            eval_or_derive_cfg = yaml.load(r_f)
        call_derive = False  # call derive or eval_arch
        if eval_or_derive_cfg["search_space_type"] == "nasbench-201":
            call_derive = True
            if args.addi_controller_cfgs:
                derive_cfg = copy.deepcopy(eval_or_derive_cfg)
                derive_cfg["controller_cfg"].update(
                    yaml.load(StringIO(args.addi_controller_cfgs))
                )
                derive_cfg_file = "{}_derive.yaml".format(cfg_file.rsplit(".", 1)[0])
                with open(derive_cfg_file, "w") as wf:
                    yaml.dump(derive_cfg, wf)
            else:
                derive_cfg = eval_or_derive_cfg

            if args.subset:
                if derive_cfg["controller_cfg"]["text_file"]:
                    with open(derive_cfg["controller_cfg"]["text_file"], "r") as rf2:
                        arch_num = len(rf2.read().strip().split("\n"))
                else:
                    arch_num = 15625 if args.iso else 6466
            else:
                # derive 6466 or 15625
                if not args.iso:
                    derive_cfg["controller_cfg"]["text_file"] = os.path.join(
                        _HOME_DIR, "awnas/data/nasbench-201/non-isom.txt"
                    )
                    arch_num = 6466
                else:
                    derive_cfg["controller_cfg"]["text_file"] = os.path.join(
                        _HOME_DIR, "awnas/data/nasbench-201/iso.txt"
                    )
                    arch_num = 15625

        cfg_files.append(derive_cfg_file)
        num_archs.append(arch_num)
        call_derives.append(call_derive)

if not all(call_derives):
    # has nb301/101, need call eval-arch
    arch_files = get_arch_files(ckpts, args.arch_file, args.arch_file_ckpt_rel_path)

# ---- result paths ----
if len(ckpts) > 1:
    assert args.single_result_file is None
    common_path = os.path.commonpath(ckpts)

    res_files = [
        os.path.relpath(ckpt, common_path).strip("/") + (".yaml" if call_derive else ".pkl")
        for ckpt, call_derive in zip(ckpts, call_derives)
    ]
else:
    assert args.single_result_file is not None
    res_files = [args.single_result_file]

os.makedirs(args.result_dir, exist_ok=True)
for res_file in res_files:
    if "/" in res_file:
        os.makedirs(
            os.path.join(args.result_dir, res_file.rsplit("/", 1)[0]), exist_ok=True
        )
print("Would save to:", res_files)

num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)


def _worker(p_id, gpu_id, queue):
    while 1:
        token = queue.get()
        if token is None:
            break
        # ckpt_dir, res_file = token
        cfg_file, ckpt_dir, res_file, num_arch, call_derive, arch_file = token
        out_file = os.path.join(args.result_dir, res_file)
        derive_log = out_file.replace(".yaml", ".log").replace(".pkl", ".log")
        if call_derive:
            # call derive
            cmd = (
                "awnas derive {} --load {} --out-file {} --gpu {} -n {} --test --seed 123 "
                "--runtime-save >{} 2>&1"
            ).format(cfg_file, ckpt_dir, out_file, gpu_id, num_arch, derive_log)
        else:
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

for cfg_file, ckpt, res_file, num_arch, call_derive, arch_file in zip(
    cfg_files, ckpts, res_files, num_archs, call_derives, arch_files
):
    queue.put((cfg_file, ckpt, res_file, num_arch, call_derive, arch_file))

# close all the workers
for _ in range(num_processes):
    queue.put(None)
