"""
Supernet for differentiable rollouts.
"""

import contextlib

import torch
from torch.nn import functional as F

from aw_nas import assert_rollout_type, utils
from aw_nas.rollout.base import DartsArch, DifferentiableRollout, BaseRollout
from aw_nas.utils import data_parallel, use_params
from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp

__all__ = ["DiffSubCandidateNet", "DiffSuperNet"]

class DiffSubCandidateNet(CandidateNet):
    def __init__(self, super_net, rollout: DifferentiableRollout, gpus=tuple(),
                 virtual_parameter_only=True, eval_no_grad=True):
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
        if detach_arch:
            arch = [
                DartsArch(
                    op_weights=op_weights.detach(),
                    edge_norms=edge_norms.detach() if edge_norms is not None else None
                ) for op_weights, edge_norms in self.arch
            ]
        else:
            arch = self.arch

        if not self.gpus or len(self.gpus) == 1:
            return self.super_net.forward(inputs, arch, detach_arch=detach_arch)

        if arch[0].op_weights.ndimension() == 2:
            arch = [
                DartsArch(
                    op_weights=a.op_weights.repeat(len(self.gpus), 1),
                    edge_norms=(a.edge_norms.repeat(len(self.gpus)) \
                     if a.edge_norms is not None else None))
                for a in arch
            ]
        else:
            # Ugly fix for rollout_size > 1
            # call scatter here and stack...
            # split along dimension 1,
            # then concatenate along dimension 0 for `data_parallel` to scatter it again
            num_split = len(self.gpus)
            rollout_batch_size = arch[0].op_weights.shape[1]
            assert rollout_batch_size % num_split == 0
            split_size = rollout_batch_size // num_split
            # arch = [torch.cat(torch.split(a, split_size, dim=1), dim=0) for a in arch]
            # Note: edge_norms (1-dim) do not support batch_size, just repeat
            arch = [DartsArch(
                op_weights=torch.cat(torch.split(a.op_weights, split_size, dim=1), dim=0),
                edge_norms=(a.edge_norms.repeat(len(self.gpus)) \
                            if a.edge_norms is not None else None))
                    for a in arch]
        return data_parallel(self.super_net, (inputs, arch), self.gpus,
                             module_kwargs={"detach_arch": detach_arch})

    def _forward_with_params(self, inputs, params, **kwargs): #pylint: disable=arguments-differ
        with use_params(self.super_net, params):
            return self.forward(inputs, **kwargs)

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
                 preprocess_op_type=None,
                 cell_use_preprocess=True,
                 cell_use_shortcut=False,
                 cell_shortcut_op_type="skip_connect",
                 cell_group_kwargs=None,
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
            preprocess_op_type=preprocess_op_type,
            cell_use_preprocess=cell_use_preprocess,
            cell_group_kwargs=cell_group_kwargs,
            cell_use_shortcut=cell_use_shortcut,
            cell_shortcut_op_type=cell_shortcut_op_type)

        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only
        self.candidate_eval_no_grad = candidate_eval_no_grad

    # ---- APIs ----
    def extract_features(self, inputs, rollout_or_arch, **kwargs):
        if isinstance(rollout_or_arch, BaseRollout):
            # from extract_features (wrapper wm)
            arch = rollout_or_arch.arch
        else:
            # from candidate net
            arch = rollout_or_arch
        return super().extract_features(inputs, arch, **kwargs)

    def assemble_candidate(self, rollout):
        return DiffSubCandidateNet(self, rollout, gpus=self.gpus,
                                   virtual_parameter_only=self.candidate_virtual_parameter_only,
                                   eval_no_grad=self.candidate_eval_no_grad)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("differentiable")]


class DiffSharedCell(SharedCell):
    def num_out_channel(self):
        return self.num_out_channels * self._steps

    def forward(self, inputs, arch, detach_arch=True):  # pylint: disable=arguments-differ
        assert self._num_init == len(inputs)

        states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]
        offset = 0

        # in parallel forward, after scatter, a namedtuple will be come a normal tuple
        arch = DartsArch(*arch)
        use_edge_normalization = arch.edge_norms is not None

        for i_step in range(self._steps):
            to_ = i_step + self._num_init
            if use_edge_normalization:
                act_lst = [
                    arch.edge_norms[offset + from_] *  # edge norm factor scalar on this edge
                    self.edges[from_][to_](
                        state,
                        arch.op_weights[offset + from_],  # op weights vector on this edge
                        detach_arch=detach_arch
                    )
                    for from_, state in enumerate(states)
                ]
            else:
                act_lst = [
                    self.edges[from_][to_](
                        state, arch.op_weights[offset + from_], detach_arch=detach_arch
                    )
                    for from_, state in enumerate(states)
                ]
            new_state = sum(act_lst)
            offset += len(states)
            states.append(new_state)

        out = torch.cat(states[-self._steps:], dim=1)
        if self.use_shortcut and self.layer_index != 0:
            out = out + self.shortcut_reduction_op(inputs[-1])
        return out


class DiffSharedOp(SharedOp):
    def forward(self, x, weights, detach_arch=True):  # pylint: disable=arguments-differ
        if weights.ndimension() == 2:
            # weights: (batch_size, num_op)
            if not weights.shape[0] == x.shape[0]:
                # every `x.shape[0] % weights.shape[0]` data use the same sampled arch weights
                assert x.shape[0] % weights.shape[0] == 0
                weights = weights.repeat(x.shape[0] // weights.shape[0], 1)
            return sum(
                [
                    weights[:, i].reshape(-1, 1, 1, 1) * op(x)
                    for i, op in enumerate(self.p_ops)
                ]
            )

        out_act: torch.Tensor = 0.0
        # weights: (num_op)
        if self.partial_channel_proportion is None:
            for w, op in zip(weights, self.p_ops):
                if detach_arch and w.item() == 0:
                    continue
                act = op(x).detach_() if w.item() == 0 else op(x)
                out_act += w * act
        else:
            op_channels = x.shape[1] // self.partial_channel_proportion
            x_1 = x[:, :op_channels, :, :]  # these channels goes through op
            x_2 = x[:, op_channels:, :, :]  # these channels skips op

            # apply pooling if the ops have stride=2
            if self.stride == 2:
                x_2 = F.max_pool2d(x_2, 2, 2)

            for w, op in zip(weights, self.p_ops):
                # if detach_arch and w.item() == 0:
                #     continue  # not really sure about this
                act = op(x_1)

                # if w.item() == 0:
                #     act.detach_()  # not really sure about this either
                out_act += w * act

            out_act = torch.cat((out_act, x_2), dim=1)

            # PC-DARTS implements a deterministic channel_shuffle() (not what they said in the paper)
            # ref: https://github.com/yuhuixu1993/PC-DARTS/blob/b74702f86c70e330ce0db35762cfade9df026bb7/model_search.py#L9
            out_act = self._channel_shuffle(out_act, self.partial_channel_proportion)

            # this is the random channel shuffle
            # channel_perm = torch.randperm(out_act.shape[1])
            # out_act = out_act[:, channel_perm, :, :]

        return out_act

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int):
        """channel shuffle for PC-DARTS"""
        n, c, h, w = x.shape

        x = x.view(n, groups, -1, h, w).transpose(1, 2).contiguous()

        x = x.view(n, c, h, w).contiguous()

        return x
