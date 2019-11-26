import os
from contextlib import contextmanager
from functools import wraps

import six
import numpy as np
import torch
from torch import nn
from nics_fix_pt import quant
import pytest

def _add_call_counter(func):
    @wraps(func)
    def _func(*args, **kwargs):
        _func.num_calls += 1
        return func(*args, **kwargs)
    _func.num_calls = 0
    return _func

quant.quantitize = _add_call_counter(quant.quantitize)
from nics_fix_pt.quant import quantitize

@contextmanager
def patch_variable(patches):
    backup = []
    for module, name, value in patches:
        backup.append((module, name, getattr(module, name)))
        setattr(module, name, value)
    yield
    for module, name, value in backup:
        setattr(module, name, value)

def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))

def _supernet_sample_cand(net):
    ss = net.search_space

    rollout = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    cand_net = net.assemble_candidate(rollout)
    return cand_net

    
BITWIDTH = 8
FIX_METHOD = 1 # auto_fix

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {n: {
        "method": torch.autograd.Variable(torch.IntTensor(np.array([method])), requires_grad=False),
        "scale": torch.autograd.Variable(torch.IntTensor(np.array([scale])), requires_grad=False),
        "bitwidth": torch.autograd.Variable(torch.IntTensor(np.array([bitwidth])), requires_grad=False)
    } for n in names}

def fix_forward(self, *args, **kwargs):
    with fix_params(self):
        return self.old_forward(*args, **kwargs)

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
            elif isinstance(mod, nn.BatchNorm2d):
                to_fix = ["weight", "bias", "running_mean", "running_var"]
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
    quantized = {"conv": 0, "bn": 0, "linear": 0}
    for mod_prefix in to_fix_mods:
        fix_cfg = module._generated_fixed_cfgs[mod_prefix]
        mod = current_active_modules[mod_prefix]
        quantize(mod, fix_cfg, original=original)
        if isinstance(mod, nn.Conv2d):
            key = "conv"
        elif isinstance(mod, nn.Linear):
            key = "linear"
        elif isinstance(mod, nn.BatchNorm2d):
            key = "bn"
        quantized[key] += 1
    print("num quantized modules: ", quantized)
    yield
    if not original:
        for mod_prefix in to_fix_mods:
            mod = current_active_modules[mod_prefix]
            setback_buffer(mod)

@pytest.mark.parametrize("case", range(5))
def test_new_quantize(case):
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import SuperNet
    import numpy as np
    import torch
    from torch import nn
    import nics_fix_pt.nn_fix as nfp
    
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

    ss = get_search_space(cls="cnn", **{
        "num_steps": 4,
        "num_layers": 1,
        "num_cell_groups": 1,
        "cell_layout": [0],
        "reduce_cell_groups": [0],
    })
    device = "cuda:0"
    with patch_variable([(nn, "Conv2d", FixedConv),
                         (nn, "BatchNorm2d", FixedBatchNorm2d),
                         (nn, "Linear", FixedLinear)]):
        super_net_fixed = SuperNet(ss, device, dropout_rate=0)
    super_net = SuperNet(ss, device, dropout_rate=0)

    # use same float params
    snf_params = dict(super_net_fixed.named_parameters())
    for name, param in super_net.named_parameters():
        snf_params[name].data.copy_(param)

    # assemble the same subarch
    rollout = ss.random_sample()
    cand_net = super_net.assemble_candidate(rollout)
    cand_net.old_forward = cand_net.forward
    cand_net_fixed = super_net_fixed.assemble_candidate(rollout)
    cand_net.train()
    cand_net_fixed.train()

    lr = 1e-3
    FixedLinear.forward = _add_call_counter(FixedLinear.forward)
    FixedConv.forward = _add_call_counter(FixedConv.forward)
    FixedBatchNorm2d.forward = _add_call_counter(FixedBatchNorm2d.forward)
    optimizer = torch.optim.SGD(cand_net.parameters(), lr=lr)
    optimizer_fixed = torch.optim.SGD(cand_net_fixed.parameters(), lr=lr)
    # only fixed params
    for _ in range(5):
        batch_size = 2
        data = _cnn_data(batch_size=batch_size)
        quantitize.num_calls = 0
        logits = fix_forward(cand_net, data[0])
        print("fix_forward quantize calls: ", quantitize.num_calls)
    
        quantitize.num_calls = 0
        FixedConv.forward.num_calls = 0
        FixedLinear.forward.num_calls = 0
        FixedBatchNorm2d.forward.num_calls = 0
    
        logits_fixed = cand_net_fixed.forward(data[0])
        print("old fix_forward quantize calls: ", quantitize.num_calls)
        print("old fix_forward, num modules: conv, linear, bn",
              FixedConv.forward.num_calls, FixedLinear.forward.num_calls,
              FixedBatchNorm2d.forward.num_calls)
        assert (logits == logits_fixed).all()
    
        loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
        loss_fixed = nn.CrossEntropyLoss()(logits_fixed, data[1].cuda())
        assert (loss == loss_fixed).all()
        loss.backward()
        loss_fixed.backward()
        optimizer.step()
        optimizer_fixed.step()
