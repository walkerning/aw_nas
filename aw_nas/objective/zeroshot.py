# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from aw_nas.objective.base import BaseObjective


def parameters_saliency(model, batch_inputs, saliency_fn):
    model.zero_grad()
    l = (batch_inputs)
    l.backward(torch.ones_like(l))

    saliency = 0.
    for _, p in model.named_modules():
        if isinstance(p, nn.Conv2d):
            saliency += saliency_fn(p.weight, p.weight.grad)
    return saliency


def grad_norm(params, grads):
    return grads.abs().sum()

def snip(params, grads):
    return (params * grads).abs().sum()

def grasp(params, grads):
    # Hassian Matrix need to be calculated
    raise NotImplementedError()

def synflow(params, grads):
    # TODO: NOT sure whether it is right implementation
    return (params * grads).sum()
    return params.sum()


def activation_saliency(model, batch_inputs, saliency_fn):
    saliency = []
    def _hooker(self, module, inputs, outputs):
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            saliency += [saliency_fn(module, inputs, outputs)]

    model.zero_grad()
    l = model(batch_inputs)
    l.backward(torch.ones_like(l))
    # TODO: how to extract the gradient of activation layers


def fisher(module, inputs, activations):
    raise NotImplementedError("To be solved how to extract the gradient of activation layer")

def jacob_covariance(model, batch_inputs):
    model.zero_grad()

    batch_inputs.requires_grad_(True)

    l = model(batch_inputs)
    l.backward(torch.ones_like(l))
    jacob = x.grad.detach()
    return jacob

def correlation(jacob, eps=1e-5):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    v += eps
    return np.sum(np.log(v) + 1. / v)

def jacob_score(model, batch_inputs):
    jacob = jacob_covariance(model, batch_inputs)
    return -correlation(jacob)



def calc_scores(model, batch_inputs, perfs=["snip", "synflow", "grad_norm", "jacob", "vote"]):
    saliency_fns = {
        "snip": snip,
        "synflow": synflow,
        "grad_norm": grad_norm
    }
    saliency_fns = {k: partial(parameters_saliency, saliency_fn=v) for k, v in saliency_fns.items()}
    saliency_fns["jacob"] = jacob_score

    scores = {k: saliency_fns[k](model, batch_inputs) for k in perfs if k != "vote"}

    if "vote" in perfs:
        assert set(("snip", "synflow", "jacob")).issubset(set(perfs))
        scores["vote"] = np.max([scores[k] for k in ["snip", "synflow", "jacob"]])
    
    return scores


class ZeroShot(BaseObjective):
    NAME = "zero-shot"

    def __init__(self, search_space, perf_names, schedule_cfg=None):
        super(ZeroShot, self).__init__(search_space, schedule_cfg=schedule_cfg)

        self._perf_names = perf_names

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return self._perf_names

    def get_perfs(self, inputs, outputs, targets, cand_net):
        scores = calc_scores(cand_net, inputs, self.perf_names())
        return [scores[p] for p in self.perf_names()]

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):

        return torch.tensor(0.)

