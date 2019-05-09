# -*- coding: utf-8 -*-
"""Base class definition of WeightsManager, CandidateNet."""

import abc
import contextlib

import six
import torch
from torch import nn

from aw_nas import Component

class BaseWeightsManager(Component):
    REGISTRY = "weights_manager"

    def __init__(self, search_space, device, schedule_cfg=None):
        super(BaseWeightsManager, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device

    @abc.abstractmethod
    def assemble_candidate(self, rollout):
        """Assemble a candidate net using rollout.
        """
        # eg. return CandidateNet(rollout)

    @abc.abstractmethod
    def step(self, gradients, optimizer):
        """Update the weights manager state using gradients."""

    @abc.abstractmethod
    def save(self, path):
        """Save the state of the weights_manager to `path` on disk."""

    @abc.abstractmethod
    def load(self, path):
        """Load the state of the weights_manager from `path` on disk."""


class CandidateNet(nn.Module):
    @contextlib.contextmanager
    def begin_virtual(self):
        """Enter a virtual context, in which all the state changes will be reset when exiting.

        For those evaluators that need to conduct virtual update as surrogate steps.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, *args, **kwargs): #pylint: disable=arguments-differ
        pass

    @abc.abstractmethod
    def get_device(self):
        """
        Get the device of the candidate net.
        """
        pass

    def train_queue(self, queue, optimizer, criterion=nn.CrossEntropyLoss(), steps=1):
        for _ in range(steps):
            data = next(queue)
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            _, targets = data
            outputs = self.forward_data(*data)
            loss = criterion(outputs, targets)
            self.zero_grad()
            loss.backward()
            optimizer.step()

    def forward_data(self, inputs, targets=None):
        """Forward the candidate net on the data.
        Args:
        Returns:
            output of the last layer.
        """
        inputs = inputs.to(self.get_device())
        return self(inputs)

    def forward_queue(self, queue, steps=1):
        outputs = []
        for _ in range(steps):
            data = next(queue)
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            outputs.append(self.forward_data(*data))
        return torch.cat(outputs, dim=0)

    def gradient(self, data, criterion=nn.CrossEntropyLoss(),
                 parameters=None, eval_criterions=None):
        """Get the gradient with respect to the candidate net parameters.

        Args:
            parameters (optional): if specificied, can be a dict of param_name: param,
            or a list of parameter name.
        Returns:
            grads (dict of name: grad tensor)
        """
        active_parameters = dict(self.named_parameters())
        if parameters is not None:
            _parameters = dict(parameters)
            _addi = set(_parameters.keys()).difference(active_parameters)
            assert not _addi,\
                "Cannot get gradient of parameters that are not active in this candidate net: {}"\
                    .format(", ".join(_addi))
        else:
            _parameters = active_parameters

        data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
        _, targets = data
        outputs = self.forward_data(*data)
        loss = criterion(outputs, targets)
        self.zero_grad()
        loss.backward()

        grads = [(k, v.grad.clone()) for k, v in six.iteritems(_parameters)\
                 if v.grad is not None]

        if eval_criterions:
            eval_res = [c(outputs, targets) for c in eval_criterions]
            return grads, eval_res
        return grads

    def eval_queue(self, queue, criterions, steps=1):
        average_ans = None
        count = 0
        with torch.no_grad():
            for _ in range(steps):
                data = next(queue)
                data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
                outputs = self.forward_data(data[0])
                ans = [c(outputs, data[1]) for c in criterions]
                if average_ans is None:
                    average_ans = ans
                else:
                    average_ans = [s + x for s, x in zip(average_ans, ans)]
                count += 1
        return [s / count for s in average_ans]

    def eval_data(self, data, criterions):
        """
        Returns:
           results (list of results return by criterions)
        """
        with torch.no_grad():
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            outputs = self.forward_data(data[0])
            return [c(outputs, data[1]) for c in criterions]
