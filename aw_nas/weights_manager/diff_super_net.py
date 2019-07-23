"""
Supernet for differentiable rollouts.
"""

import contextlib

import torch

from aw_nas import assert_rollout_type, utils
from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp
from aw_nas.utils import data_parallel

__all__ = ["DiffSubCandidateNet", "DiffSuperNet"]

class DiffSubCandidateNet(CandidateNet):
    def __init__(self, super_net, rollout, gpus=tuple(), virtual_parameter_only=True,
                 eval_no_grad=True):
        super(DiffSubCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = super_net.device
        self.gpus = gpus
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
        if not self.gpus or len(self.gpus) == 1:
            return self.super_net.forward(inputs, arch, detach_arch=detach_arch)
        if arch[0].ndimension() == 2:
            arch = [a.repeat([len(self.gpus), 1]) for a in arch]
        else:
            # Ugly fix for rollout_size > 1
            # call scatter here and stack...
            # split along dimension 1,
            # then concatenate along dimension 0 for `data_parallel` to scatter it again
            num_split = len(self.gpus)
            rollout_batch_size = arch[0].shape[1]
            assert rollout_batch_size % num_split == 0
            split_size = rollout_batch_size // num_split
            arch = [torch.cat(torch.split(a, split_size, dim=1), dim=0) for a in arch]
        return data_parallel(self.super_net, (inputs, arch), self.gpus,
                             module_kwargs={"detach_arch": detach_arch})

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
        return utils.flatten_list([c(data[0], outputs, data[1]) for c in criterions])


class DiffSuperNet(SharedNet):
    NAME = "diff_supernet"

    def __init__(self, search_space, device, rollout_type="differentiable",
                 gpus=tuple(),
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 cell_use_preprocess=True, cell_group_kwargs=None,
                 candidate_virtual_parameter_only=False,
                 candidate_eval_no_grad=True):
        super(DiffSuperNet, self).__init__(
            search_space, device, rollout_type,
            cell_cls=DiffSharedCell, op_cls=DiffSharedOp,
            gpus=gpus,
            num_classes=num_classes, init_channels=init_channels,
            stem_multiplier=stem_multiplier,
            max_grad_norm=max_grad_norm, dropout_rate=dropout_rate,
            use_stem=use_stem, stem_stride=stem_stride, stem_affine=stem_affine,
            cell_use_preprocess=cell_use_preprocess,
            cell_group_kwargs=cell_group_kwargs)

        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only
        self.candidate_eval_no_grad = candidate_eval_no_grad

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return DiffSubCandidateNet(self, rollout, gpus=self.gpus,
                                   virtual_parameter_only=self.candidate_virtual_parameter_only,
                                   eval_no_grad=self.candidate_eval_no_grad)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("differentiable")]


class DiffSharedCell(SharedCell):
    def num_out_channel(self):
        return self.num_out_channels * self.search_space.num_steps

    def forward(self, inputs, arch, detach_arch=True): #pylint: disable=arguments-differ
        assert self._num_init == len(inputs)
        states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]
        offset = 0

        for i_step in range(self._steps):
            to_ = i_step + self._num_init
            act_lst = [self.edges[from_][to_](state, arch[offset+from_],
                                              detach_arch=detach_arch) \
                       for from_, state in enumerate(states)]
            new_state = sum(act_lst)
            offset += len(states)
            states.append(new_state)
        return torch.cat(states[-self.search_space.num_steps:], dim=1)


class DiffSharedOp(SharedOp):
    def forward(self, x, weights, detach_arch=True): #pylint: disable=arguments-differ
        if weights.ndimension() == 2:
            # weights: (batch_size, num_op)
            if not weights.shape[0] == x.shape[0]:
                # every `x.shape[0] % weights.shape[0]` data use the same sampled arch weights
                assert x.shape[0] % weights.shape[0] == 0
                weights = weights.repeat(x.shape[0] // weights.shape[0], 1)
            return sum([weights[:, i].reshape(-1, 1, 1, 1) * op(x)
                        for i, op in enumerate(self.p_ops)])

        out_act = 0.
        # weights: (num_op)
        for w, op in zip(weights, self.p_ops):
            if detach_arch and w.item() == 0:
                continue
            act = op(x).detach_() if w.item() == 0 else op(x)
            out_act += w * act
        return out_act
