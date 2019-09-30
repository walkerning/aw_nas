#pylint: disable-all
"""
Script for patch SubCandidateNet/CNNGenotypeModel.forward
"""
from contextlib import contextmanager
import six

import numpy as np
import torch
from torch import nn

from nics_fix_pt.quant import quantitize
from aw_nas.weights_manager.super_net import SubCandidateNet
from aw_nas.final.cnn_model import CNNGenotypeModel

BITWIDTH = 8
FIX_METHOD = 1 # auto_fix

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {n: {
        "method": torch.autograd.Variable(torch.IntTensor(np.array([method])), requires_grad=False),
        "scale": torch.autograd.Variable(torch.IntTensor(np.array([scale])), requires_grad=False),
        "bitwidth": torch.autograd.Variable(torch.IntTensor(np.array([bitwidth])), requires_grad=False)
    } for n in names}

# ---- patch ----
## Here do not patch forward, as activation is not quantized in calls to `forward`,
## it's not meaningful to quantize the weights during the calls to `forward` too
## It must be patched, or ``backward through a graph second time'' error will occur.
## Let's reset all the module attributes to the original parameters in `_parameters`.
SubCandidateNet.old_forward = SubCandidateNet.forward
CNNGenotypeModel.old_forward = CNNGenotypeModel.forward
def fix_forward(self, *args, **kwargs):
    with fix_params(self, original=True):
        return self.old_forward(*args, **kwargs)
SubCandidateNet.forward = fix_forward
CNNGenotypeModel.forward = fix_forward

# only patch `forward_one_step_callback`, not forward
SubCandidateNet.old_forward_one_step_callback = SubCandidateNet.forward_one_step_callback
CNNGenotypeModel.old_forward_one_step_callback = CNNGenotypeModel.forward_one_step_callback
def fix_forward_one_step_callback(self, inputs, callback):
    with fix_params(self):
        return self.old_forward_one_step_callback(inputs, callback)
SubCandidateNet.forward_one_step_callback = fix_forward_one_step_callback
CNNGenotypeModel.forward_one_step_callback = fix_forward_one_step_callback
# ---- end patch ----

def quantize(self, fix_cfg, original=False):
    for n, param in six.iteritems(self._parameters):
        if not isinstance(param, (torch.Tensor, torch.autograd.Variable)):
            continue
        if not original:
            # run quantize
            param, _ = quantitize(param, fix_cfg.get(n, {}), {}, name=n)
        object.__setattr__(self, n, param)
    for n, param in six.iteritems(self._buffers):
        if not isinstance(param, (torch.Tensor, torch.autograd.Variable)):
            continue
        if not original:
            # run quantize
            param, _ = quantitize(param, fix_cfg.get(n, {}), {}, name=n)
        object.__setattr__(self, n, param)

def setback_buffer(self):
    for n, param in six.iteritems(self._buffers):
        if param is not None:
            # set buffer back, as there will be no gradient, just in-place modification
            # FIXME: For fixed-point batch norm,
            # the running mean/var accumulattion is on quantitized mean/var,
            # which means it might fail to update the running mean/var
            # if the updating momentum is too small
            self._buffers[n] = getattr(self, n)

@contextmanager
def fix_params(module, original=False):
    if not hasattr(module, "_generated_fixed_cfgs"):
        module._generated_fixed_cfgs = {}
        for mod_prefix, mod in module.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                to_fix = ["weight", "bias"]
            # elif isinstance(mod, nn.BatchNorm2d):
            #     to_fix = ["weight", "bias", "running_mean", "running_var"]
            else:
                to_fix = []
            if to_fix:
                module._generated_fixed_cfgs[mod_prefix] = _generate_default_fix_cfg(
                    to_fix, method=FIX_METHOD, bitwidth=BITWIDTH)
    current_active_modules = dict(module.named_modules()) \
                             if not hasattr(module, "active_named_members")\
                                else dict(module.active_named_members(member="modules",
                                                                      prefix="super_net",
                                                                      check_visited=True))

    # FIXME: possible multi-access in multi-gpu
    to_fix_mods = set(module._generated_fixed_cfgs.keys()).intersection(
        current_active_modules.keys())
    for mod_prefix in to_fix_mods:
        fix_cfg = module._generated_fixed_cfgs[mod_prefix]
        mod = current_active_modules[mod_prefix]
        quantize(mod, fix_cfg, original=original)
    yield
    if not original:
        for mod_prefix in to_fix_mods:
            mod = current_active_modules[mod_prefix]
            setback_buffer(mod)

"""
Note there are a lot randomness in the search process. so all the number are just a ref.
|   | quantize                                         | 30 eva    | time quantize weight /   | #quantize | ratio |
|   | patch method                                     | step time | time feature inject 1e-4 | calls     |       |
|---+--------------------------------------------------+-----------+--------------------------+-----------+-------|
| 1 | old                                              | ~65       | 31.01/12.82              | ~68088    |   2.4 |
| 2 | new patch forward&fonestepcallback               | ~75       | 28.68/13.05              | ~63954    |   2.2 |
| 3 | new patch fonestepcallback                       | ~60       | 14.19/12.90              | ~31997    |   1.1 |
| x | new patch forward(set original)&fonestepcallback | ~60       | -                        | ~31997    |     - |

from 1->2, the quantization call reduction comes from avoiding quantizing unused params in one forward pass (`check_visited=True`)
and avoiding duplicated quantization calls when there are double connection in the rollout.
"""
