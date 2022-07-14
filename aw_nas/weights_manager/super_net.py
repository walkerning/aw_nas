# -*- coding: utf-8 -*-
"""
Shared weights super net.
"""

from __future__ import print_function

import itertools
from collections import OrderedDict
import contextlib
import six

from torch import nn

from aw_nas.common import assert_rollout_type, group_and_sort_by_to_node, BaseRollout
from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.shared import SharedNet, SharedCell, SharedOp
from aw_nas.utils import data_parallel, use_params
from aw_nas.utils.parallel_utils import _check_support_candidate_member_mask

__all__ = ["SubCandidateNet", "SuperNet"]


class SubCandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """

    def __init__(self, super_net, rollout, member_mask, gpus=tuple(), cache_named_members=False,
                 virtual_parameter_only=True, eval_no_grad=True):
        super(SubCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.search_space = super_net.search_space
        self.member_mask = member_mask
        self.cache_named_members = cache_named_members
        self.virtual_parameter_only = virtual_parameter_only
        self._cached_np = None
        self._cached_nb = None

        self._flops_calculated = False
        self.total_flops = 0

        genotype_list = rollout.genotype_list()
        self.genotypes = [g[1] for g in genotype_list]
        self.genotypes_grouped = list(zip(
            [group_and_sort_by_to_node(conns)
             for conns in self.genotypes[:self.search_space.num_cell_groups]],
            self.genotypes[self.search_space.num_cell_groups:]))
        # self.genotypes_grouped = [group_and_sort_by_to_node(g[1]) for g in \
        #                           if "concat" not in g[0]]

    @contextlib.contextmanager
    def begin_virtual(self):
        """
        On entering, store the current states (parameters/buffers) of the network
        On exiting, restore the stored states.
        Needed for surrogate steps of each candidate network,
        as different SubCandidateNet share the same set of SuperNet weights.
        """

        w_clone = {k: v.clone() for k, v in self.super_net.named_parameters()}
        if not self.virtual_parameter_only:
            buffer_clone = {k: v.clone() for k, v in self.super_net.named_buffers()}

        yield

        for n, v in self.super_net.named_parameters():
            v.data.copy_(w_clone[n])
        del w_clone

        if not self.virtual_parameter_only:
            for n, v in self.super_net.named_buffers():
                v.data.copy_(buffer_clone[n])
            del buffer_clone

    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net.forward(inputs, self.genotypes_grouped)

    def forward(self, inputs, single=False): #pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs={"single": True})

    def _forward_with_params(self, inputs, params, **kwargs): #pylint: disable=arguments-differ
        with use_params(self.super_net, params):
            return self.forward(inputs, **kwargs)

    def forward_one_step(self, context, inputs=None):
        """
        Forward one step.
        Data parallism is not supported for now.
        """
        return self.super_net.forward_one_step(context, inputs, self.genotypes_grouped)

    def plot_arch(self):
        return self.super_net.search_space.plot_arch(list(zip(
            self.cell_group_names + [n + "_concat" for n in self.cell_group_names],
            self.genotypes)))

    def named_parameters(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                # use cached members
                if self._cached_np is None:
                    self._cached_np = []
                    for n, v in self.active_named_members(member="parameters", prefix=""):
                        self._cached_np.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_np:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(member="parameters", prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_parameters(prefix=prefix):
                yield n, v

    def named_buffers(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                if self._cached_nb is None:
                    self._cached_nb = []
                    for n, v in self.active_named_members(member="buffers", prefix=""):
                        self._cached_nb.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_nb:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(member="buffers", prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_buffers(prefix=prefix):
                yield n, v

    def active_named_members(self, member, prefix="", recurse=True, check_visited=False):
        """
        Get the generator of name-member pairs active
        in this candidate network. Always recursive.
        """
        # memo, there are potential weight sharing, e.g. when `tie_weight` is True in rnn_super_net,
        # encoder/decoder share weights. If there is no memo, `sub_named_members` will return
        # 'decoder.weight' and 'encoder.weight', both refering to the same parameter, whereasooo
        # `named_parameters` (with memo) will only return 'encoder.weight'. For possible future
        # weight sharing, use memo to keep the consistency with the builtin `named_parameters`.
        memo = set()
        for n, v in self.super_net.sub_named_members(self.genotypes,
                                                     prefix=prefix,
                                                     member=member,
                                                     check_visited=check_visited):
            if v in memo:
                continue
            memo.add(v)
            yield n, v

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        member_lst = []
        for n, v in itertools.chain(self.active_named_members(member="parameters", prefix=""),
                                    self.active_named_members(member="buffers", prefix="")):
            member_lst.append((n, v))
        state_dict = OrderedDict(member_lst)
        return state_dict

    def forward_one_step_callback(self, inputs, callback):
        """
        For fault injection.
        """
        # forward stem
        _, context = self.forward_one_step(context=None, inputs=inputs)
        callback(context.last_state, context)

        # forward the cells
        for i_layer in range(0, self.search_space.num_layers):
            num_steps = self.search_space.get_layer_num_steps(i_layer) + \
                        self.search_space.num_init_nodes + 1
            for _ in range(num_steps):
                while True: # call `forward_one_step` until this step ends
                    _, context = self.forward_one_step(context)
                    callback(context.last_state, context)
                    if context.is_end_of_cell or context.is_end_of_step:
                        break
            # end of cell (every cell has the same number of num_steps)
        # final forward
        _, context = self.forward_one_step(context)
        callback(context.last_state, context)
        return context.last_state

class SuperNet(SharedNet):
    """
    A cell-based super network
    """
    NAME = "supernet"

    def __init__(self, search_space, device, rollout_type="discrete",
                 gpus=tuple(),
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 preprocess_op_type=None,
                 cell_use_preprocess=True, cell_group_kwargs=None,
                 cell_use_shortcut=False,
                 cell_shortcut_op_type="skip_connect",
                 bn_affine=False,
                 candidate_member_mask=True, candidate_cache_named_members=False,
                 candidate_virtual_parameter_only=False, candidate_eval_no_grad=True):
        """
        Args:
            candidate_member_mask (bool): If true, the candidate network's `named_parameters`
                or `named_buffers` method will only return parameters/buffers that is active,
                `begin_virtual` just need to store/restore these active variables.
                This should be more efficient.
            candidate_cache_named_members (bool): If true, the candidate network's
                named parameters/buffers will be cached on the first calculation.
                It should not cause any logical, however, due to my benchmark, this bring no
                performance increase. So default disable it.
            candidate_virtual_parameter_only (bool): If true, the candidate network's
                `begin_virtual` will only store/restore parameters, not buffers (e.g. running
                mean/running std in BN layer).
        """
        _check_support_candidate_member_mask(gpus, candidate_member_mask, self.NAME)

        super(SuperNet, self).__init__(search_space, device, rollout_type,
                                       cell_cls=DiscreteSharedCell, op_cls=DiscreteSharedOp,
                                       gpus=gpus,
                                       num_classes=num_classes, init_channels=init_channels,
                                       stem_multiplier=stem_multiplier,
                                       max_grad_norm=max_grad_norm, dropout_rate=dropout_rate,
                                       use_stem=use_stem, stem_stride=stem_stride,
                                       stem_affine=stem_affine,
                                       preprocess_op_type=preprocess_op_type,
                                       cell_use_preprocess=cell_use_preprocess,
                                       cell_group_kwargs=cell_group_kwargs,
                                       cell_use_shortcut=cell_use_shortcut,
                                       cell_shortcut_op_type=cell_shortcut_op_type,
                                       bn_affine=bn_affine)

        # candidate net with/without parameter mask
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_virtual_parameter_only = candidate_virtual_parameter_only
        self.candidate_eval_no_grad = candidate_eval_no_grad
        self.set_hook()
        self._flops_calculated = False
        self.total_flops = 0

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def set_hook(self):
        for name, module in self.named_modules():
            if "auxiliary" in name:
                continue
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    inputs[0].size(2) * inputs[0].size(3) / \
                                    (module.stride[0] * module.stride[1] * module.groups)
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def sub_named_members(self, genotypes,
                          prefix="", member="parameters", check_visited=False):
        conns, concat_nodes = genotypes[:self.search_space.num_cell_groups], \
                              genotypes[self.search_space.num_cell_groups:]

        prefix = prefix + ("." if prefix else "")

        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "cells":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix+mod_name):
                yield n, v

        if check_visited:
            # only a subset of modules under `self.cells` will be forwarded
            # from the last output, parse the dependency backward
            visited = set()
            cell_idxes = [len(self.cells)-1]
            depend_nodes_lst = [{edge[1] for edge in genotype}.intersection(range(self._num_init))\
                                for genotype in conns]
            while cell_idxes:
                cell_idx = cell_idxes.pop()
                visited.update([cell_idx])
                # cg_idx is the cell group of the cell i
                cg_idx = self._cell_layout[cell_idx]
                depend_nodes = depend_nodes_lst[cg_idx]
                depend_cell_idxes = [cell_idx - self._num_init + node_idx
                                     for node_idx in depend_nodes]
                depend_cell_idxes = [i for i in depend_cell_idxes if i >= 0 and i not in visited]
                cell_idxes += depend_cell_idxes
        else:
            visited = list(range(self._num_layers))

        for cell_idx in sorted(visited):
            cell = self.cells[cell_idx]
            genotype = conns[self._cell_layout[cell_idx]]
            for n, v in cell.sub_named_members((genotype,
                                                concat_nodes[self._cell_layout[cell_idx]]),
                                               prefix=prefix + "cells.{}".format(cell_idx),
                                               member=member,
                                               check_visited=check_visited):
                yield n, v

    # ---- APIs ----
    def extract_features(self, inputs, rollout_or_genotypes, **kwargs):
        if isinstance(rollout_or_genotypes, BaseRollout):
            # from extract_features
            genotype_list = rollout_or_genotypes.genotype_list()
            genotypes = [g[1] for g in genotype_list]
            genotypes_grouped = list(zip(
                [group_and_sort_by_to_node(conns)
                 for conns in genotypes[:self.search_space.num_cell_groups]],
                genotypes[self.search_space.num_cell_groups:]))
        else:
            # from candidate net
            genotypes_grouped = rollout_or_genotypes
        return super().extract_features(inputs, genotypes_grouped, **kwargs)

    def assemble_candidate(self, rollout, **kwargs):
        return SubCandidateNet(self, rollout,
                               gpus=self.gpus,
                               member_mask=self.candidate_member_mask,
                               cache_named_members=self.candidate_cache_named_members,
                               virtual_parameter_only=self.candidate_virtual_parameter_only,
                               eval_no_grad=self.candidate_eval_no_grad)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("discrete")]

class DiscreteSharedCell(SharedCell):
    def num_out_channel(self):
        return self.num_out_channels * self._out_multipler

    def forward(self, inputs, genotype_grouped): #pylint: disable=arguments-differ
        conns_grouped, concat_nodes = genotype_grouped
        assert self._num_init == len(inputs)
        if self.use_preprocess:
            states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]
        else:
            states = [s for s in inputs]

        for to_, connections in conns_grouped:
            state_to_ = 0.
            for op_type, from_, _ in connections:
                out = self.edges[from_][to_](states[from_], op_type)
                state_to_ = state_to_ + out
            states.append(state_to_)

        out = self.concat_op([states[ind] for ind in concat_nodes])
        if self.use_shortcut and self.layer_index != 0:
            out = out + self.shortcut_reduction_op(inputs[-1])
        return out

    def forward_one_step(self, context, genotype_grouped):
        to_ = cur_step = context.next_step_index[1]
        if cur_step < self._num_init: # `self._num_init` preprocess steps
            ind = len(context.previous_cells) - (self._num_init - cur_step)
            ind = max(ind, 0)
            # state = self.preprocess_ops[cur_step](context.previous_cells[ind])
            # context.current_cell.append(state)
            # context.last_conv_module = self.preprocess_ops[cur_step].get_last_conv_module()
            current_op = context.next_op_index[1]
            state, context = self.preprocess_ops[cur_step].forward_one_step(
                context=context,
                inputs=context.previous_cells[ind] if current_op == 0 else None)
            if context.next_op_index[1] == 0: # this preprocess op finish, append to `current_cell`
                assert len(context.previous_op) == 1
                context.current_cell.append(context.previous_op[0])
                context.previous_op = []
                context.last_conv_module = self.preprocess_ops[cur_step].get_last_conv_module()
        elif cur_step < self._num_init + self._steps: # the following steps
            conns_grouped = genotype_grouped[0]
            conns = conns_grouped[cur_step - self._num_init][1]
            op_ind, current_op = context.next_op_index
            if op_ind == len(conns):
                # all connections added to context.previous_ops, sum them up
                state = sum([st for st in context.previous_op])
                context.current_cell.append(state)
                context.previous_op = []
            else:
                op_type, from_, _ = conns[op_ind]
                state, context = self.edges[from_][to_].forward_one_step(
                    context=context,
                    inputs=context.current_cell[from_] if current_op == 0 else None,
                    op_type=op_type)
        else: # final concat
            concat_nodes = genotype_grouped[1]
            state = self.concat_op([context.current_cell[ind] for ind in concat_nodes])
            context.current_cell = []
            context.previous_cells.append(state)
        return state, context

    def sub_named_members(self, genotype,
                          prefix="", member="parameters", check_visited=False):
        conns, _ = genotype

        prefix = prefix + ("." if prefix else "")
        all_from = {edge[1] for edge in conns}
        for i, pre_op in enumerate(self.preprocess_ops):
            if not check_visited or i in all_from:
                for n, v in getattr(pre_op, "named_" + member)\
                    (prefix=prefix+"preprocess_ops."+str(i)):
                    yield n, v

        for op_type, from_, to_ in conns:
            edge_share_op = self.edges[from_][to_]
            for n, v in edge_share_op.sub_named_members(
                    op_type,
                    prefix=prefix + "edge_mod.f_{}_t_{}".format(from_, to_),
                    member=member):
                yield n, v

class DiscreteSharedOp(SharedOp):
    def forward(self, x, op_type): #pylint: disable=arguments-differ
        index = self.primitives.index(op_type)
        return self.p_ops[index](x)

    def forward_one_step(self, context, inputs, op_type): #pylint: disable=arguments-differ
        index = self.primitives.index(op_type)
        return self.p_ops[index].forward_one_step(context=context, inputs=inputs)

    def sub_named_members(self, op_type,
                          prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        index = self.primitives.index(op_type)
        for n, v in getattr(self.p_ops[index], "named_" + member)(prefix="{}p_ops.{}"\
                                                .format(prefix, index)):
            yield n, v
