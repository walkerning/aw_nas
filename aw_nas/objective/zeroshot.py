# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from aw_nas import utils
from aw_nas.objective.base import BaseObjective

try:
    from aw_nas.objective.zerocost import get_measures
except ImportError as e:
    utils.getLogger("objective.zeroshot").warn(
        ("Cannot import `get_measures` from `aw_nas.objective.zerocost`, zeroshot objectives cannot work normally! {}\n"
         "\tSee examples/research/surgery/README.md for more information").format(e))


def parameters_saliency(model, batch_inputs, saliency_fn):
    saliency = 0.0
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
            saliency += saliency_fn(module.weight, module.weight.grad)

    return saliency


def grad_norm(params, grads):
    return grads.abs().sum().item()


def snip(params, grads):
    return (params * grads).abs().sum().item()


def grasp(params, grads):
    # Hassian Matrix need to be calculated
    raise NotImplementedError()


def synflow(params, grads):
    # TODO: NOT sure whether it is right implementation
    return (params * grads).sum().item()


def activation_saliency(model, batch_inputs, saliency_fn):
    saliency = []

    def _hooker(self, module, inputs, outputs):
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            saliency += [saliency_fn(module, inputs, outputs)]

    model.zero_grad()
    l = model(batch_inputs)
    l.backward(torch.ones_like(l))
    del l
    # TODO: how to extract the gradient of activation layers


def fisher(module, inputs, activations):
    raise NotImplementedError(
        "To be solved how to extract the gradient of activation layer"
    )


def jacob_covariance(model, batch_inputs):
    # model.zero_grad()

    # batch_inputs.requires_grad_(True)

    # l = model(batch_inputs)
    # l.backward(torch.ones_like(l))
    jacob = batch_inputs.grad.detach().cpu().numpy()
    # del l
    return jacob


def correlation(jacob, eps=1e-5):
    corrs = np.corrcoef(jacob.reshape(jacob.shape[0], -1))
    v, _ = np.linalg.eig(corrs)
    v += eps
    return np.sum(np.log(v) + 1.0 / v)


def jacob_score(model, batch_inputs):
    jacob = jacob_covariance(model, batch_inputs)
    try:
        return -correlation(jacob)
    except:
        return 0.0


def calc_scores(
    model, batch_inputs, perfs=["snip", "synflow", "grad_norm", "jacob", "vote"]
):
    saliency_fns = {"snip": snip, "synflow": synflow, "grad_norm": grad_norm}

    # get the gradients (currently, only parameters)
    model.zero_grad()
    batch_inputs.requires_grad_(True)
    loss = model(batch_inputs)
    loss.backward(torch.ones_like(loss))

    saliency_fns = {
        k: partial(parameters_saliency, saliency_fn=v) for k, v in saliency_fns.items()
    }
    saliency_fns["jacob"] = jacob_score

    scores = {k: saliency_fns[k](model, batch_inputs) for k in perfs if k != "vote"}

    if "vote" in perfs:
        assert set(("snip", "synflow", "jacob")).issubset(set(perfs))
        scores["vote"] = np.max([scores[k] for k in ["snip", "synflow", "jacob"]])
    del loss
    return scores


class ZeroShot(BaseObjective):
    NAME = "zero-shot"

    def __init__(
        self, search_space, perf_names, aggregate_as_list=False, schedule_cfg=None
    ):
        super(ZeroShot, self).__init__(search_space, schedule_cfg=schedule_cfg)

        self.aggregate_as_list = aggregate_as_list
        self._perf_names = perf_names

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return self._perf_names

    def get_perfs(self, inputs, outputs, targets, cand_net):
        # scores = calc_scores(cand_net, inputs, self.perf_names())
        # return [scores[p] for p in self.perf_names()]
        measures = get_measures(cand_net, inputs.device, inputs, targets, measure_names=self.perf_names())
        return [measures[p] for p in self.perf_names()]

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def get_loss(
        self,
        inputs,
        outputs,
        targets,
        cand_net,
        add_controller_regularization=True,
        add_evaluator_regularization=True,
    ):

        return torch.tensor(0.0)

    def aggregate_fn(self, perf_name, is_training=True):
        if self.aggregate_as_list:
            return list
        else:
            return super().aggregate_fn(perf_name, is_training)
