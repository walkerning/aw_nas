# -*- coding: utf-8 -*-
"""
Differentiable-relaxation based controllers
"""

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import utils
from aw_nas.common import DifferentiableRollout as DiffRollout
from aw_nas.controller.base import BaseController

class DiffController(BaseController, nn.Module):
    """
    Using the gumbel softmax reparametrization of categorical distribution.
    The sampled actions (ops/nodes) will be hard/soft vectors rather than discrete indexes.
    """
    NAME = "differentiable"

    SCHEDULABLE_ATTRS = [
        "gumbel_temperature",
        "entropy_coeff"
    ]

    def __init__(self, search_space, device,
                 gumbel_hard=False, gumbel_temperature=1.0,
                 entropy_coeff=0.01, max_grad_norm=None,
                 schedule_cfg=None):
        """
        Args:
            gumbel_hard (bool): If true, the soft relaxed vector calculated by gumbel softmax
                in the forward pass will be argmax to a one-hot vector. The gradients are straightly
                passed through argmax operation. This will cause discrepancy of the forward and
                backward pass, but allow the samples to be sparse.
            gumbel_temperature (float): The temperature of gumbel softmax. As the temperature gets
                smaller, when used with `gumbel_hard==True`, the discrepancy of the forward/backward
                pass gets smaller; When used with `gumbel_hard==False`, the samples become more
                sparse(smaller bias), but the variance of the gradient estimation using samples
                becoming larger.
        """
        super(DiffController, self).__init__(search_space, schedule_cfg=schedule_cfg)
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        _num_ops = len(self.search_space.shared_primitives)
        _num_init_nodes = self.search_space.num_init_nodes
        _num_edges = sum(_num_init_nodes+i for i in range(self.search_space.num_steps))
        self.cg_alphas = nn.ParameterList([nn.Parameter(torch.randn(_num_edges, _num_ops)) \
                                           for _ in range(self.search_space.num_cell_groups)])

        self.to(self.device)

    def set_mode(self, mode):
        if mode == "train":
            nn.Module.train(self)
        elif mode == "eval":
            nn.Module.eval(self)
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def forward(self, n=1): #pylint: disable=arguments-differ
        return self.sample(n=n)

    def sample(self, n=1):
        rollouts = []
        for _ in range(n):
            arch_list = []
            sampled_list = []
            logits_list = []
            for alpha in self.cg_alphas:
                sampled, _ = utils.gumbel_softmax(alpha, self.gumbel_temperature, hard=False)
                if self.gumbel_hard:
                    arch = utils.straight_through(sampled)
                else:
                    arch = sampled
                arch_list.append(arch)
                sampled_list.append(utils.get_numpy(sampled))
                logits_list.append(utils.get_numpy(alpha))
            rollouts.append(DiffRollout(arch_list, sampled_list, logits_list, self.search_space))
        return rollouts

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        """Load the parameters from disk."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        self.logger.info("Loaded controller network from %s", path)

    def _entropy_loss(self):
        probs = [F.softmax(alpha, dim=-1) for alpha in self.cg_alphas]
        return sum(-(torch.log(prob) * prob).sum() for prob in probs)

    def gradient(self, loss):
        self.zero_grad()
        _loss = loss
        if self.entropy_coeff:
            _loss += self._entropy_loss()
        _loss.backward()
        return utils.get_numpy(_loss), [(k, v.grad.clone()) for k, v in self.named_parameters()]

    def step_gradient(self, gradients, optimizer):
        self.zero_grad()
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # clip the gradients
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def step(self, rollouts, optimizer): # very memory inefficient
        self.zero_grad()
        losses = [r.get_perf() for r in rollouts]
        optimizer.step()
        [l.backward() for l in losses]
        return np.mean([l.detach().cpu().numpy() for l in losses])

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        num = len(rollouts)
        logits_list = [[utils.get_numpy(logits) for logits in r.logits] for r in rollouts]
        _ss = self.search_space
        if self.gumbel_hard:
            cg_logprobs = [0. for _ in range(_ss.num_cell_groups)]
        cg_entros = [0. for _ in range(_ss.num_cell_groups)]
        for rollout, logits in zip(rollouts, logits_list):
            for cg_idx, (vec, cg_logits) in enumerate(zip(rollout.arch, logits)):
                prob = utils.softmax(cg_logits)
                logprob = np.log(prob)
                if self.gumbel_hard:
                    inds = np.argmax(utils.get_numpy(vec), axis=-1)
                    cg_logprobs[cg_idx] += np.sum(logprob[range(len(inds)), inds])
                cg_entros[cg_idx] += -(prob * logprob).sum()

        # mean across rollouts
        if self.gumbel_hard:
            cg_logprobs = [s / num for s in cg_logprobs]
            total_logprob = sum(cg_logprobs)
            cg_logprobs_str = ",".join(["{:.2f}".format(n) for n in cg_logprobs])

        cg_entros = [s / num for s in cg_entros]
        total_entro = sum(cg_entros)
        cg_entro_str = ",".join(["{:.2f}".format(n) for n in cg_entros])

        if log:
            # maybe log the summary
            self.logger.info("%s%d rollouts: %s ENTROPY: %2f (%s)",
                             log_prefix, num,
                             "-LOG_PROB: %.2f (%s) ;"%(-total_logprob, cg_logprobs_str) \
                             if self.gumbel_hard else "",
                             total_entro, cg_entro_str)
            if step is not None and not self.writer.is_none():
                if self.gumbel_hard:
                    self.writer.add_scalar("log_prob", total_logprob, step)
                self.writer.add_scalar("entropy", total_entro, step)

        stats = [(n + " ENTRO", entro) for n, entro in zip(_ss.cell_group_names, cg_entros)]
        if self.gumbel_hard:
            stats += [(n + " LOGPROB", logprob) for n, logprob in \
                      zip(_ss.cell_group_names, cg_logprobs)]
        return OrderedDict(stats)

    def rollout_type(self):
        return "differentiable"
