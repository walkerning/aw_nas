# -*- coding: utf-8 -*-

import argparse
import torch
import copy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-file", required=True, help="output file")
args, ckpts = parser.parse_known_args()

num_ckpts = len(ckpts)

ckpt = torch.load(ckpts[0], map_location=torch.device("cpu"))
average_ckpt = copy.deepcopy(ckpt)
keys = ckpt["weights_manager"].keys()

for fname in ckpts[1:]:
    ckpt = torch.load(fname, map_location=torch.device("cpu"))
    for key in keys:
        if not "num_batches_tracked" in key:
            average_ckpt["weights_manager"][key] += ckpt["weights_manager"][key]
        else:
            average_ckpt["weights_manager"][key] = max(ckpt["weights_manager"][key], average_ckpt["weights_manager"][key])

for key in keys:
    if not "num_batches_tracked" in key:
        average_ckpt["weights_manager"][key] /= num_ckpts

torch.save(average_ckpt, args.out_file)
