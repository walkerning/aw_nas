"""
Supernet for differentiable rollouts.
"""

import contextlib

import torch

from aw_nas import assert_rollout_type
from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp

__all__ = ["DiffSubCandidateNet", "DiffSuperNet"]

class DiffSubCandidateNet(CandidateNet):
    def __init__(self, super_net, rollout, virtual_parameter_only=True):
        super(DiffSubCandidateNet, self).__init__()
        self.super_net = super_net
        self._device = super_net.device
        self.arch = rollout.arch
        self.virtual_parameter_only = virtual_parameter_only

    def get_device(self):
        return self._device

    @contextlib.contextmanager
    def begin_virtual(self):
        w_clone = {k: v.clone() for k, v in self.named_parameters()}
        if not self.virtual_parameter_only:
            buffer_clone = {k: v.clone() for k, v in self.named_buffers()}

        yield

        for n, v in self.named_parameters():
            v.data.copy_(w_clone[n])
        del w_clone

        if not self.virtual_parameter_only:
            for n, v in self.named_buffers():
                v.data.copy_(buffer_clone[n])
            del buffer_clone

    def forward(self, inputs, detach_arch=True): #pylint: disable=arguments-differ
        arch = [a.detach() for a in self.arch] if detach_arch else self.arch
        return self.super_net.forward(inputs, arch, detach_arch=detach_arch)

    def named_parameters(self, *args, **kwargs): #pylint: disable=arguments-differ
        return self.super_net.named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs): #pylint: disable=arguments-differ
        return self.super_net.named_buffers(*args, **kwargs)

    def eval_data(self, data, criterions, mode="eval", **kwargs): #pylint: disable=arguments-differ
        """
        Override eval_data, to enable gradient.

        Returns:
           results (list of results return by criterions)
        """
        self._set_mode(mode)

        outputs = self.forward_data(data[0], **kwargs)
        return [c(outputs, data[1]) for c in criterions]


class DiffSuperNet(SharedNet):
    NAME = "diff_supernet"

    def __init__(self, search_space, device, rollout_type="differentiable",
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 candidate_virtual_parameter_only=True):
        super(DiffSuperNet, self).__init__(search_space, device, rollout_type,
                                           cell_cls=DiffSharedCell, op_cls=DiffSharedOp,
                                           num_classes=num_classes, init_channels=init_channels,
                                           stem_multiplier=stem_multiplier,
                                           max_grad_norm=max_grad_norm, dropout_rate=dropout_rate)

        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return DiffSubCandidateNet(self, rollout,
                                   virtual_parameter_only=self.candidate_virtual_parameter_only)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("differentiable")]


class DiffSharedCell(SharedCell):
    def num_out_channel(self):
        return self.num_channels * self.search_space.num_steps

    def forward(self, inputs, arch, detach_arch=True): #pylint: disable=arguments-differ
        assert self._num_init == len(inputs)
        states = [op(inputs) for op, inputs in zip(self.preprocess_ops, inputs)]
        offset = 0

        for _ in range(self._steps):
            act_lst = [self.edges[offset+from_](state, arch[offset+from_],
                                                detach_arch=detach_arch) \
                       for from_, state in enumerate(states)]
            new_state = sum(act_lst)
            offset += len(states)
            states.append(new_state)

        return torch.cat(states[-self.search_space.num_steps:], dim=1)


class DiffSharedOp(SharedOp):
    def forward(self, x, weights, detach_arch=True): #pylint: disable=arguments-differ
        out_act = 0.
        for w, op in zip(weights, self.p_ops):
            if detach_arch and w.item() == 0:
                continue
            act = op(x).detach_() if w.item() == 0 else op(x)
            out_act += w * act
        return out_act
