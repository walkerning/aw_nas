import abc

import torch
from torch import nn

from aw_nas.rollout import BaseRollout
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet

__all__ = ["BaseNeck", "BaseNeckCandidateNet"]


class BaseNeck(BaseWeightsManager, nn.Module):
    NAME = "neck"

    def __init__(
        self, search_space, device, rollout_type, gpus=tuple(), schedule_cfg=None
    ):
        super(BaseNeck, self).__init__(search_space, device, rollout_type, schedule_cfg)
        nn.Module.__init__(self)

        self.gpus = gpus

    def forward(self, features, rollout=None):
        raise NotImplementedError()

    def finalize(self, rollout):
        raise NotImplementedError()

    def get_feature_channel_num(self):
        raise NotImplementedError()

    def assemble_candidate(self, rollout):
        return BaseNeckCandidateNet(self, rollout, gpus=self.gpus)

    @classmethod
    def supported_rollout_types(cls):
        return list(BaseRollout.all_classes_().keys()) + [None]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def save(self, path):
        pass

    def load(self, path):
        pass

    def step(self, gradients, optimizer):
        self.zero_grad()
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        optimizer.step()

    def step_current_gradients(self, optimizer):
        optimizer.step()

    def set_device(self, device):
        self.device = device
        self.to(device)


class BaseNeckCandidateNet(CandidateNet):
    """
    The candidate net for detection neck weights manager.
    """

    def __init__(self, super_net, rollout, gpus=tuple()):
        super(BaseNeckCandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.multiprocess = super_net.multiprocess
        self.search_space = super_net.search_space

        self._flops_calculated = False
        self.total_flops = 0
        self.rollout = rollout

    def get_device(self):
        return self._device

    def forward(self, inputs, single=False):  # pylint: disable=arguments-differ
        if self.multiprocess:
            return self.super_net.parallel_model.forward(inputs, self.rollout)
        else:
            return self.super_net.forward(inputs, self.rollout)
