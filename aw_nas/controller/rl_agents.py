# -*- coding: utf-8 -*-
"""
Implementations of various RL agents to update controller.
"""
import abc

import numpy as np
import torch

from aw_nas import Component
from aw_nas import utils

class BaseRLAgent(Component):
    REGISTRY = "rl_agent"

    def __init__(self, controller, schedule_cfg=None):
        super(BaseRLAgent, self).__init__(schedule_cfg)

        self.controller = controller

    @abc.abstractmethod
    def step(self, rollouts, optimizer, perf_name):
        """Update controller using rollouts."""

    @abc.abstractmethod
    def save(self, path):
        """Save the agent state to disk."""

    @abc.abstractmethod
    def load(self, path):
        """Load the agent state from disk."""

class PGAgent(BaseRLAgent):
    NAME = "pg"

    def __init__(self, controller, alpha=0.999, gamma=1.,
                 entropy_coeff=0.01, max_grad_norm=None, batch_update=True):
        """
        Args:
            controller (aw_nas.RLController)
            alpha (float): The moving average decay of the rewards baseline.
            gamma (float): Discount ratio of rewards.
            entropy_coeff (float): The coeffient of the entropy encouraging loss term.
            max_grad_norm (float): Clip the gradient of controller parameters.
            batch_update (bool): If true, batch the update.
        """
        super(PGAgent, self).__init__(controller)
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_update = batch_update
        assert batch_update == True

        self.baseline = None

    def _step(self, log_probs, entropies, returns, optimizer, retain_graph=False):
        if self.baseline is None:
            self.baseline = np.mean(returns, 0).copy()
        advantages = returns - self.baseline
        advantages = torch.from_numpy(advantages)\
                          .to(dtype=log_probs.dtype, device=log_probs.device)
        self.baseline = self.alpha * self.baseline + (1.0 - self.alpha) * np.mean(returns, 0)

        loss = -log_probs * advantages - self.entropy_coeff * entropies
        loss = loss.sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                           self.max_grad_norm)
        optimizer.step()
        return loss.item()

    def step(self, rollouts, optimizer, perf_name):
        if self.batch_update:
            log_probs = torch.stack([torch.cat(r.info["log_probs"]) for r in rollouts])
            entropies = torch.stack([torch.cat(r.info["entropies"]) for r in rollouts])
            returns = np.array([utils.compute_returns(r.get_perf(perf_name), self.gamma,
                                                      log_probs.shape[-1]) for r in rollouts])
            loss = self._step(log_probs, entropies, returns, optimizer)
        else:
            losses = []
            for i, rollout in enumerate(rollouts):
                log_probs = torch.cat(rollout.info["log_probs"]).unsqueeze(0)
                entropies = torch.cat(rollout.info["entropies"]).unsqueeze(0)
                returns = utils.compute_returns(rollout.get_perf(perf_name),
                                                self.gamma, len(log_probs))[None, :]
                loss = self._step(log_probs, entropies, returns, optimizer,
                                  retain_graph=i < len(rollouts)-1)
                losses.append(loss)
            loss = np.mean(losses)

        return loss

    def save(self, path):
        np.save(path, self.baseline)

    def load(self, path):
        # For 2,3 compatability
        try:
            FileNotFoundError
        except NameError:
            FileNotFoundError = IOError #pylint: disable=redefined-builtin
        try:
            self.baseline = np.load(path)
        except FileNotFoundError:
            self.baseline = np.load(path + ".npy")

    def on_epoch_end(self, epoch):
        if not self.writer.is_none() and self.baseline is not None:
            # maybe write tensorboard info
            self.writer.add_scalar("last_baseline", self.baseline[-1], epoch)

class PPOAgent(BaseRLAgent):
    NAME = "ppo"

    def __init__(self, controller, clip_param=0.2, alpha=0.999, gamma=1.,
                 entropy_coeff=0.01, max_grad_norm=None):
        """
        Args:
            controller (aw_nas.RLController)
            clip_param (float): The clip range of probability ratio in PPO.
            alpha (float): The moving average decay of the rewards baseline.
            gamma (float): Discount ratio of rewards.
            entropy_coeff (float): The coeffient of the entropy encouraging loss term.
            max_grad_norm (float): Clip the gradient of controller parameters.
        """
        super(PPOAgent, self).__init__(controller)
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.baseline = None

    def _step(self, log_probs, entropies, returns, optimizer):
        if self.baseline is None:
            self.baseline = np.mean(returns, 0).copy()
        advantages = returns - self.baseline
        advantages = torch.from_numpy(advantages)\
                          .to(dtype=log_probs.dtype, device=log_probs.device)
        self.baseline = self.alpha * self.baseline + (1.0 - self.alpha) * np.mean(returns, 0)

        ratio = torch.exp(log_probs - log_probs.clone().detach()) # pnew / pold
        surr1 = ratio * advantages # surrogate from conservative policy iteration
        # PPO's pessimistic surrogate (L^CLIP)
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_param,
                            max=1.0 + self.clip_param) * advantages
        loss = - torch.min(surr1, surr2)

        # TODO: value critic loss
        loss = loss - self.entropy_coeff * entropies
        loss = loss.sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
                                           self.max_grad_norm)
        optimizer.step()
        return loss.item()

    def step(self, rollouts, optimizer, perf_name):
        log_probs = torch.stack([torch.cat(r.info["log_probs"]) for r in rollouts])
        entropies = torch.stack([torch.cat(r.info["entropies"]) for r in rollouts])
        returns = np.array([utils.compute_returns(r.get_perf(perf_name), self.gamma,
                                                  log_probs.shape[-1]) for r in rollouts])
        loss = self._step(log_probs, entropies, returns, optimizer)
        return loss

    def save(self, path):
        np.save(path, self.baseline)

    def load(self, path):
        # For 2,3 compatability
        try:
            FileNotFoundError
        except NameError:
            FileNotFoundError = IOError #pylint: disable=redefined-builtin
        try:
            self.baseline = np.load(path)
        except FileNotFoundError:
            self.baseline = np.load(path + ".npy")

    def on_epoch_end(self, epoch):
        if not self.writer.is_none() and self.baseline is not None:
            # maybe write tensorboard info
            self.writer.add_scalar("last_baseline", self.baseline[-1], epoch)
