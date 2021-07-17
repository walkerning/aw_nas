
import numpy as np
import torch
import torch.nn as nn


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
    v, _ = np,.linalg.eig(corrs)
    v += eps
    return np.sum(np.log(v) + 1. / v)

def jacob_score(model, batch_inputs):
    jacob = jacob_covariance(model, batch_inputs)
    return -correlation(jacob)



def calc_scores(model, batch_inputs):
    saliency_fns = {
        "snip": snip,
        "synflow": synflow,
        "grad_norm": grad_norm
    }

    saliencies = {k: parameters_saliency(model, batch_inputs, v) for k, v in saliency_fns.items()}

    uncorrelation = jacob_score(model, batch_inputs)

    scores = {**saliencies}
    scores["jacob"] = uncorrelation

    scores["vote"] = np.max([scores[k] for k in ["snip", "synflow", "jacob"]])
    
    return scores

def make_vote(scores, poll=["snip", "synflow", "jacob"]):
    ranking = {}


    


