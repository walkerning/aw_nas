"""
Script for patching fixed point modules.
"""
import six
import numpy as np
import torch
from torch import nn
import nics_fix_pt.nn_fix as nfp
from nics_fix_pt.fix_modules import register_fix_module

BITWIDTH = 8
FIX_METHOD = 1 # auto_fix

INJECT = True
INJECT_PROB = 1e-1
INJECT_STEP = 1

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

class RRAMConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(RRAMConv, self).__init__(*args, **kwargs)
        self.step = 0
        self.fault_bit_list = np.array([2**x for x in range(8)], dtype=np.float32)
        self.random_tensor = None

    def forward(self, x):
        inject_prob = INJECT_PROB
        param = self.weight
        max_ = torch.max(torch.abs(param)).cpu().data
        if self.step == 0:
            random_tensor = param.new(param.size())\
						.random_(0, int(1. / inject_prob))
            scale = torch.ceil(torch.log(
                torch.max(torch.max(torch.abs(param)),
                      torch.tensor(1e-5).float().to(param.device))) / np.log(2.))
            step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]).to(param.device),
                                                 requires_grad=False),
                         (scale.float() - 7.))
            fault_ind = (random_tensor < 1)
            random_tensor.fill_(1.)
            param = param / step
            random_tensor[fault_ind] = torch.tensor(self.fault_bit_list[np.random.randint(
                       0, 8, size=fault_ind.sum().cpu().data)]).to(param.device)
        else:
            random_tensor = self.random_tensor
        self.step += 1
        if self.step >= INJECT_STEP:
            self.step = 0
        tmp_param = (param / random_tensor).int().float()
        mod = torch.fmod(tmp_param, 2.)
        random_pos = np.random.random(param.shape)
        
        param = param - (mod * random_tensor * (random_tensor > 1).float() * torch.tensor((random_pos > 0.1).astype(float), dtype=torch.float).to(param.device))\
		 + ((mod.abs() < 1e-4).float() * random_tensor * (random_tensor > 1).float() * ((param > 0).float() * 2 - 1) * torch.tensor((random_pos <= 0.1).astype(float), dtype=torch.float).to(param.device))#saf-1
        param = param * step
        # clip
        param.clamp_(min=-max_, max=max_)
        # for masked bp
        normal_mask = torch.ones_like(param)
        normal_mask[fault_ind] = 0
        masked = normal_mask * param
        param = (param - masked).detach() + masked

        object.__setattr__(self, "weight", param)
        out = super(RRAMConv, self).forward(x)
        return out

register_fix_module(RRAMConv)

class RRAMFixedConv(nfp.RRAMConv_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(RRAMFixedConv, self).__init__(*args, **kwargs)

class FixedLinear(nfp.Linear_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedLinear, self).__init__(*args, **kwargs)

class FixedBatchNorm2d(nfp.BatchNorm2d_fix):
    """
    FIXME:
      When training using multi gpu, the update for the buffers(running_mean/running_var)
      is done using data on only the main gpu; This will lead to discrepancy between
      one-gpu/multi-gpu training. Maybe can consider using `encoding.nn.BatchNorm2d`.

      Moreover, as the fixed-point configuration is shared between threads, and each gpu
      separately computes using its own data split, the unprotected concurrent read/write
      to this shared config lead to undefined behaviour. Maybe this config should be
      thread-local too. (Maybe this should be a fix in the nics_fix_pytorch lib).

      An temporary fix is not to quantize the `running_mean` and `running_var` here.
    """
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedBatchNorm2d, self).__init__(*args, **kwargs)

if INJECT:
    nn.Conv2d = RRAMFixedConv
else:
    nn.Conv2d = FixedConv
nn.BatchNorm2d = FixedBatchNorm2d
nn.Linear = FixedLinear
