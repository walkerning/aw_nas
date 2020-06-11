"""
Script to print CPU/GPU latency of GeneralGenotypeModel's blocks
"""

import sys

import yaml
import torch

from aw_nas.common import get_search_space
from aw_nas.final.general_model import GeneralGenotypeModel
from aw_nas.utils.elapse_utils import analyze_elapses

def main():
    with open(sys.argv[1], "r") as fr:
        config = yaml.load(fr)
    ss = get_search_space(config["search_space_type"], **config["search_space_cfg"])
    genotypes = config["final_model_cfg"]["genotypes"]
    model = GeneralGenotypeModel(ss, "cuda", genotypes)

    shape = model.genotypes[0]["spatial_size"]
    inputs = torch.rand([1, model.genotypes[0]["C"], shape, shape])
    gpu_performance = analyze_elapses(model, inputs.cuda(), use_cuda=True, forward_time=100)
    for prim in gpu_performance["primitives"]:
        print(prim["elapse"])
    print("async_elapse: ", gpu_performance["async_elapse"],
          "sync_elapse: ", gpu_performance["sync_elapse"])

    # profiling on CPU
    model = model.to("cpu")
    cpu_performance = analyze_elapses(model, inputs.cpu(), use_cuda=False, forward_time=100)
    for prim in cpu_performance["primitives"]:
        print(prim["elapse"])
    print("async_elapse: ", cpu_performance["async_elapse"],
          "sync_elapse: ", cpu_performance["sync_elapse"])

if __name__ == "__main__":
    main()
