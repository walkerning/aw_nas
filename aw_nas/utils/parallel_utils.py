"""
Patch replicate function, to ignore those parameters/buffers that are not copied.
Bring a 1.4x performance increase for super_net weights_manager using discrete rollout.
"""
#pylint: disable-all

from itertools import chain

import torch
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index
from torch.nn.parallel import DataParallel as _DataParallel
from torch.nn.parallel.replicate import (
    _replicatable_module,
    _broadcast_coalesced_reshape,
    _is_script_module, _init_script_module,
    _is_script_method, _copy_scriptmodule_methods
)

__all__ = ["replicate", "DataParallel", "data_parallel"]

def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    devices = list(map(lambda x: _get_device_index(x, True), devices))
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg = []
    buffers_not_rg = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

    modules = list(network.modules())
    module_copies = [[] for _ in devices]
    module_indices = {}
    scriptmodule_skip_attr = {"_parameters", "_buffers", "_modules", "forward", "_c"}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            if _is_script_module(module):
                # we have to initialize ScriptModule properly so that
                # it works with pybind11
                replica = _init_script_module()

                attribute_names = set(entry[0] for entry in module._c._get_attributes())

                keys = set(module.__dict__.keys()) - scriptmodule_skip_attr - attribute_names
                for key in keys:
                    if not _is_script_method(module.__dict__[key]):
                        replica.__dict__[key] = module.__dict__[key]
                for name, the_type, value in module._c._get_attributes():
                    if name in module._buffers.keys():
                        continue
                    replica._c._register_attribute(name, the_type, value)
            else:
                replica = module.__new__(type(module))
                replica.__dict__ = module.__dict__.copy()
                replica._parameters = replica._parameters.copy()
                replica._buffers = replica._buffers.copy()
                replica._modules = replica._modules.copy()

            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = module_copies[j][module_idx]
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices.get(param, None)
                if param_idx is None:
                    continue
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = param_copies[j][param_idx]
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg.get(buf, None)
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg.get(buf, None)
                if buffer_idx is None:
                    continue
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = buffer_copies[j][buffer_idx]

    for j in range(num_replicas):
        _copy_scriptmodule_methods(modules, module_copies[j], module_indices)

    replicas = [module_copies[j][0] for j in range(num_replicas)]
    for model_replica in replicas:
        for _, submodule in model_replica.named_modules():
            if hasattr(submodule, "on_replicate") and callable(submodule.on_replicate):
                submodule.on_replicate()
    return replicas

## --- data parallel utils ---
def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None,
                  non_scatter_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output
            Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device("cuda:{}".format(device_ids[0]))

    for tensor in chain(module.parameters(), module.buffers()):
        if tensor.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, tensor.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

class DataParallel(_DataParallel):
    def replicate(self, module, device_ids):
        replicas = replicate(module, device_ids, not torch.is_grad_enabled())
        return replicas
