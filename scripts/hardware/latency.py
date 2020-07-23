"""
Script to print CPU/GPU latency of GeneralGenotypeModel's blocks
"""

import argparse
import os
import sys

import yaml
import torch

from aw_nas.common import get_search_space
from aw_nas.final.general_model import GeneralGenotypeModel
from aw_nas.utils.elapse_utils import analyze_elapses


def profiling(config, device, forward_time=100):
    ss = get_search_space(config["search_space_type"], **config["search_space_cfg"])
    genotypes = config["final_model_cfg"]["genotypes"]
    model = GeneralGenotypeModel(ss, device, genotypes)

    shape = model.genotypes[0]["spatial_size"]
    inputs = torch.rand([1, model.genotypes[0]["C"], shape, shape]).to(device)
    
    use_cuda = device != "cpu"
    performance = analyze_elapses(model, inputs, use_cuda=use_cuda, forward_time=forward_time)
    for prim in performance["primitives"]:
        prim["latency"] = prim["elapse"]
        del prim["elapse"]
    print("async_elapse: ", performance["async_elapse"],
          "sync_elapse: ", performance["sync_elapse"])

    return performance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='+',
                        help='the path of awnas config files to be profiled.')
    parser.add_argument('--device', type=int, required=True,
                        help='the device id of gpu, if set -1, use cpu.')
    parser.add_argument('--perf_dir', type=str, required=True,
                        help='the file path to write down performance result.')
    parser.add_argument('--forward_time', type=int, default=100,
                        help='the directory to write performance result.')

    args = parser.parse_args()

    if args.device == -1:
        device = "cpu"
    elif args.device >= 0:
        device = "cuda:{}".format(args.device)
    else:
        raise ValueError("Expect device id >= -1, got {} instead.".format(args.device))

    os.makedirs(args.perf_dir, exist_ok=True)
    perfs = []
    for cfg_path in args.config:
        with open(cfg_path, 'r') as fr:
            cfg = yaml.load(fr)
        perf = profiling(cfg, device=device, forward_time=args.forward_time)
        perfs += perf["primitives"]
        with open(os.path.join(args.perf_dir, cfg_path.split("/")[-1]), "w") as fw:
            yaml.safe_dump(perf, fw)

    with open(os.path.join(args.perf_dir, "prof_primitvie.yaml"), "w") as fw:
            yaml.safe_dump(perfs, fw)

if __name__ == "__main__":
    main()
