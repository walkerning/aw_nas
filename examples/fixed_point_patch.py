"""
Script for patching fixed point modules.
"""
import numpy as np
import torch
from torch import nn
import nics_fix_pt.nn_fix as nfp

BITWIDTH = 8
FIX_METHOD = 1 # auto_fix

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {n: {
        "method": torch.autograd.Variable(torch.IntTensor(np.array([method])), requires_grad=False),
        "scale": torch.autograd.Variable(torch.IntTensor(np.array([scale])), requires_grad=False),
        "bitwidth": torch.autograd.Variable(torch.IntTensor(np.array([bitwidth])), requires_grad=False)
    } for n in names}

class FixedConv(nfp.Conv2d_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedConv, self).__init__(*args, **kwargs)

class FixedLinear(nfp.Linear_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedLinear, self).__init__(*args, **kwargs)

class FixedBatchNorm2d(nfp.BatchNorm2d_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedBatchNorm2d, self).__init__(*args, **kwargs)

nn.Conv2d = FixedConv
nn.BatchNorm2d = FixedBatchNorm2d
nn.Linear = FixedLinear
