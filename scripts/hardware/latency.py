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
    inputs = torch.rand([1, model.genotypes[0]["C"], shape, shape])
    
    return analyze_elapses(model, inputs, device=device, forward_time=forward_time)

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
    for cfg_path in args.config:
        with open(cfg_path, 'r') as fr:
            cfg = yaml.load(fr)
        perfs = profiling(cfg, device=device, forward_time=args.forward_time)
        _dir = os.path.join(args.perf_dir, cfg_path.split("/")[-1].split(".")[0])
        os.makedirs(_dir, exist_ok=True)
        for i, perf in enumerate(perfs):
            with open(os.path.join(_dir, "{}.yaml".format(i)), "w") as fw:
                yaml.safe_dump(perf, fw)


if __name__ == "__main__":
    main()
