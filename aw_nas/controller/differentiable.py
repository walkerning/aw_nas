# -*- coding: utf-8 -*-
"""
Differentiable-relaxation based controllers
"""

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aw_nas import utils, assert_rollout_type
from aw_nas.common import DifferentiableRollout as DiffRollout
from aw_nas.controller.base import BaseController
from aw_nas.rollout.base import DartsArch


class DiffController(BaseController, nn.Module):
    """
    Using the gumbel softmax reparametrization of categorical distribution.
    The sampled actions (ops/nodes) will be hard/soft vectors rather than discrete indexes.
    """
    NAME = "differentiable"

    SCHEDULABLE_ATTRS = [
        "gumbel_temperature",
        "entropy_coeff",
        "force_uniform"
    ]

    def __init__(self, search_space, device, rollout_type="differentiable",
                 use_prob=False, gumbel_hard=False, gumbel_temperature=1.0,
                 use_edge_normalization=False,
                 entropy_coeff=0.01, max_grad_norm=None, force_uniform=False,
                 inspect_hessian_every=-1,  # inspect Hessian eigenvalues every 1 epoch. Disabled when <= 0
                 schedule_cfg=None):
        """
        Args:
            use_prob (bool): If true, use the probability directly instead of relaxed sampling.
                If false, use gumbel sampling. Default: false.
            gumbel_hard (bool): If true, the soft relaxed vector calculated by gumbel softmax
                in the forward pass will be argmax to a one-hot vector. The gradients are straightly
                passed through argmax operation. This will cause discrepancy of the forward and
                backward pass, but allow the samples to be sparse. Also applied to `use_prob==True`.
            gumbel_temperature (float): The temperature of gumbel softmax. As the temperature gets
                smaller, when used with `gumbel_hard==True`, the discrepancy of the forward/backward
                pass gets smaller; When used with `gumbel_hard==False`, the samples become more
                sparse(smaller bias), but the variance of the gradient estimation using samples
                becoming larger. Also applied to `use_prob==True`
        """
        super(DiffController, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.use_prob = use_prob
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature

        # edge normalization
        self.use_edge_normalization = use_edge_normalization

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        self.inspect_hessian_every = inspect_hessian_every
        self.inspect_hessian = False

        _num_init_nodes = self.search_space.num_init_nodes
        _num_edges_list = [
            sum(
                _num_init_nodes + i
                for i in range(self.search_space.get_num_steps(i_cg))
            )
            for i_cg in range(self.search_space.num_cell_groups)
        ]

        self.cg_alphas = nn.ParameterList([
            nn.Parameter(
                1e-3 * torch.randn(
                    _num_edges, len(self.search_space.cell_shared_primitives[i_cg])
                )
            )  # shape: [num_edges, num_ops]
            for i_cg, _num_edges in enumerate(_num_edges_list)
        ])

        if self.use_edge_normalization:
            self.cg_betas = nn.ParameterList([
                nn.Parameter(1e-3 * torch.randn(_num_edges))  # shape: [num_edges]
                for _num_edges in _num_edges_list
            ])
        else:
            self.cg_betas = None

        # meta learning related
        self.params_clone = None
        self.buffers_clone = None
        self.grad_clone = None
        self.grad_count = 0

        self.to(self.device)

    def on_epoch_start(self, epoch):
        super(DiffController, self).on_epoch_start(epoch)
        if self.inspect_hessian_every >= 0 and epoch % self.inspect_hessian_every == 0:
            self.inspect_hessian = True

    def set_mode(self, mode):
        super(DiffController, self).set_mode(mode)
        if mode == "train":
            nn.Module.train(self)
        elif mode == "eval":
            nn.Module.eval(self)
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, n=1):  # pylint: disable=arguments-differ
        return self.sample(n=n)

    def sample(self, n=1, batch_size=1, support_set=None):
        rollouts = []
        for _ in range(n):
            # op_weights.shape: [num_edges, [batch_size,] num_ops]
            # edge_norms.shape: [num_edges] do not have batch_size.
            op_weights_list = []
            edge_norms_list = []
            sampled_list = []
            logits_list = []

            for alphas in self.cg_alphas:
                if self.force_uniform:  # cg_alpha parameters will not be in the graph
                    # NOTE: `force_uniform` config does not affects edge_norms (betas),
                    # if one wants a force_uniform search, keep `use_edge_normalization=False`
                    alphas = torch.zeros_like(alphas)

                if batch_size > 1:
                    expanded_alpha = alphas.reshape([alphas.shape[0], 1, alphas.shape[1]]) \
                        .repeat([1, batch_size, 1]) \
                        .reshape([-1, alphas.shape[-1]])
                else:
                    expanded_alpha = alphas

                if self.use_prob:
                    # probability as sample
                    sampled = F.softmax(expanded_alpha / self.gumbel_temperature, dim=-1)
                else:
                    # gumbel sampling
                    sampled, _ = utils.gumbel_softmax(expanded_alpha, self.gumbel_temperature,
                                                      hard=False)

                if self.gumbel_hard:
                    op_weights = utils.straight_through(sampled)
                else:
                    op_weights = sampled

                if batch_size > 1:
                    sampled = sampled.reshape([-1, batch_size, op_weights.shape[-1]])
                    op_weights = op_weights.reshape([-1, batch_size, op_weights.shape[-1]])

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampled))
                logits_list.append(utils.get_numpy(alphas))

            if self.use_edge_normalization:
                for i_cg, betas in enumerate(self.cg_betas):
                    # eg: for 2 init_nodes and 3 steps, this is [2, 3, 4]
                    num_inputs_on_nodes = np.arange(self.search_space.get_num_steps(i_cg)) \
                                          + self.search_space.num_init_nodes
                    edge_norms = []
                    for i_node, num_inputs_on_node in enumerate(num_inputs_on_nodes):
                        # eg: for node_0, it has edge_{0, 1} as inputs, there for start=0, end=2
                        start = num_inputs_on_nodes[i_node - 1] if i_node > 0 else 0
                        end = start + num_inputs_on_node

                        edge_norms.append(F.softmax(betas[start:end], dim=0))

                    edge_norms_list.append(torch.cat(edge_norms))

                arch_list = [
                    DartsArch(op_weights=op_weights, edge_norms=edge_norms)
                    for op_weights, edge_norms in zip(op_weights_list, edge_norms_list)
                ]
            else:
                arch_list = [
                    DartsArch(op_weights=op_weights, edge_norms=None)
                    for op_weights in op_weights_list
                ]

            rollouts.append(DiffRollout(arch_list, sampled_list, logits_list, self.search_space))
        return rollouts

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        """Load the parameters from disk."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    def base_update_parameters(self):
        return self.parameters()

    def meta_clone(self, include_buffers=False):
        """
        Clone parameters and optionally buffers
        Initialize zero grads for parameters
        """
        if include_buffers:
            self.buffers_clone = {k: v.data.clone()
                                  for k, v in self.named_buffers()}
        self.params_clone = {k: v.data.clone()
                             for k, v in self.named_parameters()}
        self.grad_clone = {k: torch.zeros_like(v.data)
                           for k, v in self.named_parameters()}
        self.grad_count = 0

    def meta_gradient(self, method, base_lr=None):
        """
        Accumulate gradients for each episode
        """
        if method == "fo_maml":
            self.grad_clone = {k: self.grad_clone[k] + v.grad.clone()
                               for k, v in self.named_parameters()}
        elif method == "reptile":
            assert base_lr is not None
            self.grad_clone = {k: self.grad_clone[k] +\
                               (self.params_clone[k] - v.data.clone()) / base_lr
                               for k, v in self.named_parameters()}
        else:
            raise Exception(f"Unrecognized meta method {method}")
        self.grad_count += 1

    def meta_recover(self, include_buffers=False):
        """
        Recover cloned parameters and optionally buffers
        """
        if include_buffers:
            for k, v in self.named_buffers():
                v.data.copy_(self.buffers_clone[k])
        for k, v in self.named_parameters():
            v.data.copy_(self.params_clone[k])

    def meta_update(self, optimizer):
        for k, v in self.named_parameters():
            v.grad.copy_(self.grad_clone[k] / self.grad_count)
        self.step_current_gradient(optimizer)

    def _entropy_loss(self):
        # if self.entropy_coeff > 0:
        # In the early stage of differentiable search, entropy should be encouraged
        # to avoid the search get stuck in degenerated solutions (usually many skips, no convs).
        # While in the later stage, entropy should be regularized to make the
        # op dist more close to one-hot, thus the discrepancy before & after derive
        # will be smaller.
        if self.entropy_coeff is not None:
            probs = [F.softmax(alpha, dim=-1) for alpha in self.cg_alphas]
            return self.entropy_coeff * sum(-(torch.log(prob) * prob).sum() for prob in probs)
        return 0.

    def gradient(self, loss, return_grads=True, zero_grads=True):
        if zero_grads:
            self.zero_grad()

        if self.inspect_hessian:
            for name, param in self.named_parameters():
                max_eig = utils.torch_utils.max_eig_of_hessian(loss, param)
                self.logger.info(
                    "Max eigenvalue of Hessian of %s: %f", name, max_eig
                )

            self.inspect_hessian = False

        _loss = loss + self._entropy_loss()
        _loss.backward()
        if return_grads:
            return utils.get_numpy(_loss), [(k, v.grad.clone()) for k, v in self.named_parameters()]
        return utils.get_numpy(_loss)

    def step_current_gradient(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

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

    def step(self, rollouts, optimizer, perf_name):  # very memory inefficient
        self.zero_grad()
        losses = [r.get_perf(perf_name) for r in rollouts]
        [l.backward() for l in losses]
        optimizer.step()
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
                    inds = np.argmax(utils.get_numpy(vec.op_weights), axis=-1)
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
                             "-LOG_PROB: %.2f (%s) ;" % (-total_logprob, cg_logprobs_str) \
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

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("differentiable")]
