import torch
import torch.nn as nn
import torch.nn.functional as F

import types
import copy

from foresight.pruners.p_utils import *
from foresight.pruners import measures
from foresight.pruners.predictive import *


def get_measures_arrays(
    net_orig, device, inputs, targets, measure_names=None, loss_fn=F.cross_entropy
):
    if measure_names is None:
        measure_names = measures.available_measures

    net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    # move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu()
    torch.cuda.empty_cache()

    done, ds = False, 1
    measure_values = {}

    while not done:
        try:
            for measure_name in measure_names:
                if measure_name not in measure_values:
                    val = measures.calc_measure(
                        measure_name,
                        net_orig,
                        device,
                        inputs,
                        targets,
                        loss_fn=loss_fn,
                        split_data=ds,
                    )
                    measure_values[measure_name] = val

            done = True
        except RuntimeError as e:
            if "out of memory" in str(e):
                done = False
                if ds == inputs.shape[0] // 2:
                    raise ValueError(
                        f"Can't split data anymore, but still unable to run. Something is wrong"
                    )
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f"Caught CUDA OOM, retrying with data split into {ds} parts")
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return measure_values


def get_measures(
    net_orig,  # neural network
    device,  # GPU/CPU device used
    inputs,
    targets,
    loss_fn=F.cross_entropy,  # loss function to use within the zero-cost metrics
    measure_names=None,  # an array of measure names to compute, if left blank, all measures are computed by default
    measures_arr=None,
):  # [not used] if the measures are already computed but need to be summarized, pass them here

    # Given a neural net
    # and some information about the input data (dataloader)
    # and loss function (loss_fn)
    # this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.0
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    if measures_arr is None:
        measures_arr = get_measures_arrays(
            net_orig,
            device,
            inputs,
            targets,
            loss_fn=loss_fn,
            measure_names=measure_names,
        )

    measures = {}
    for k, v in measures_arr.items():
        if ("jacob_cov" in k) or ("relu" in k):
            measures[k] = v
        else:
            measures[k] = sum_arr(v)

    return measures
