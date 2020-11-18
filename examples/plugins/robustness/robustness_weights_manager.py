"""
Robustness weights manager and candidate net.
"""
# pylint: disable=invalid-name,missing-docstring

import re
import weakref
from collections import defaultdict, OrderedDict
import itertools

import six
import numpy as np
import torch
import torch.nn as nn

from aw_nas import ops
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas import utils
from aw_nas.utils import data_parallel, flatten_list
from aw_nas.utils.common_utils import nullcontext
from aw_nas.utils.torch_utils import _to_device


class _Cache(OrderedDict):
    def __init__(self, *args, **kwargs):
        self.buffer_size = kwargs.pop("buffer_size", 3)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size()

    def _check_size(self):
        if self.buffer_size is not None:
            while len(self) > self.buffer_size:
                self.popitem(last=False)


class RobSharedOp(nn.Module):
    """
    The operation on an edge, consisting of multiple primitives.
    Primitive 1: Zero
    Primitive 2: Identity
    Primitive 3: sep_conv_3x3
    Primitive 4: ResSepConv
    """

    def __init__(self, C, C_out, stride, primitives):
        super(RobSharedOp, self).__init__()
        self.stride = stride
        self.primitives = primitives
        self.p_ops = nn.ModuleList()

        # Load the candidate operations and save in self.p_ops
        for primitive in self.primitives:
            op = ops.get_op(primitive)(C, C_out, stride, False)
            self.p_ops.append(op)

    def forward(self, x, op_type):
        return self.p_ops[op_type](x)

    def sub_named_members(self, op_type, prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        for n, v in getattr(self.p_ops[op_type], "named_" + member)(
            prefix="{}p_ops.{}".format(prefix, op_type)
        ):
            yield n, v


class RobSharedCell(nn.Module):
    def __init__(
        self,
        op_cls,
        search_space,
        num_input_channels,
        num_out_channels,
        stride,
        prev_strides,
    ):
        super(RobSharedCell, self).__init__()

        self.search_space = search_space
        self.stride = stride
        self.num_input_channels = num_input_channels
        self.num_out_channels = num_out_channels
        self.prev_strides = prev_strides
        self._primitives = self.search_space.primitives
        self._num_nodes = self.search_space._num_nodes
        self.num_init_nodes = self.search_space.num_init_nodes
        self.preprocess_ops = nn.ModuleList()
        prev_strides = prev_strides[-self.num_init_nodes :]
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = list(reversed(prev_strides[: len(num_input_channels)]))
        for prev_c, prev_s in zip(num_input_channels, prev_strides):
            preprocess = ops.get_op("skip_connect_2")(
                C=prev_c, C_out=num_out_channels, stride=prev_s, affine=True
            )
            self.preprocess_ops.append(preprocess)

        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module()
        self.is_reduce = stride != 1

        # We save all the opertions on edges in an upper triangular matrix
        for from_ in range(self._num_nodes):
            for to_ in range(max(self.num_init_nodes, from_ + 1), self._num_nodes):
                self.edges[from_][to_] = op_cls(
                    self.num_out_channels,
                    self.num_out_channels,
                    stride=self.stride if from_ < self.num_init_nodes else 1,
                    primitives=self._primitives,
                )
                self.edge_mod.add_module(
                    "f_{}_t_{}".format(from_, to_), self.edges[from_][to_]
                )
                self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

    def on_replicate(self):
        # Although this edges is easy to understand, when paralleized,
        # the reference relationship between `self.edge` and modules under `self.edge_mod`
        # will not get updated automatically.
        # So, after each replicate, we should initialize a new edges dict
        # and update the reference manually.
        self.edges = defaultdict(dict)
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_ = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)] = edge_mod

    def forward(self, inputs, genotype):
        states = [op(_input) for op, _input in zip(self.preprocess_ops, inputs)]
        batch_size, _, height, width = states[0].shape
        o_height, o_width = height // self.stride, width // self.stride
        for to_ in range(self.num_init_nodes, self._num_nodes):
            state_to_ = torch.zeros(
                [batch_size, self.num_out_channels, o_height, o_width],
                device=states[0].device,
            )
            for from_ in range(to_):
                out = self.edges[from_][to_](states[from_], int(genotype[to_][from_]))
                state_to_ = state_to_ + out
            states.append(state_to_)
        return torch.cat(states[self.num_init_nodes :], dim=1)

    def sub_named_members(
        self, genotype, prefix="", member="parameters", check_visited=False
    ):
        prefix = prefix + ("." if prefix else "")
        # not check visited
        for i, pre_op in enumerate(self.preprocess_ops):
            for n, v in getattr(pre_op, "named_" + member)(
                prefix=prefix + "preprocess_ops." + str(i)
            ):
                yield n, v

        for from_ in range(self._num_nodes):
            for to_ in range(max(self.num_init_nodes, from_ + 1), self._num_nodes):
                edge_share_op = self.edges[from_][to_]
                for n, v in edge_share_op.sub_named_members(
                    int(genotype[to_][from_]),
                    prefix=prefix + "edge_mod.f_{}_t_{}".format(from_, to_),
                    member=member,
                ):
                    yield n, v

    def num_out_channel(self):
        return self.search_space.num_steps * self.num_out_channels


class RobSharedNet(BaseWeightsManager, nn.Module):
    NAME = "dense_rob_wm"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="dense_rob",
        gpus=tuple(),
        num_classes=10,
        init_channels=36,
        stem_multiplier=1,
        max_grad_norm=5.0,
        drop_rate=0.2,
        drop_out_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        candidate_eval_no_grad=False,  # need grad in eval to craft adv examples
            calib_bn_batch=0,
            calib_bn_num=0
    ):
        super(RobSharedNet, self).__init__(search_space, device, rollout_type)

        nn.Module.__init__(self)

        cell_cls = RobSharedCell
        op_cls = RobSharedOp
        # optionally data parallelism in SharedNet
        self.gpus = gpus

        self.search_space = search_space
        self.num_classes = num_classes
        self.device = device
        self.drop_rate = drop_rate
        self.drop_out_rate = drop_out_rate
        self.init_channels = init_channels

        # channels of stem conv / init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem

        # training
        self.max_grad_norm = max_grad_norm

        # search space configs
        self._ops_choices = self.search_space.primitives
        self._num_layers = self.search_space.num_layers

        self._num_init = self.search_space.num_init_nodes
        self._num_layers = self.search_space.num_layers
        self._cell_layout = self.search_space.cell_layout

        self.calib_bn_batch = calib_bn_batch
        self.calib_bn_num = calib_bn_num
        if self.calib_bn_num > 0 and self.calib_bn_batch > 0:
            self.logger.warn("`calib_bn_num` and `calib_bn_batch` set simultaneously, "
                             "will use `calib_bn_num` only")

        ## initialize sub modules
        if not self.use_stem:
            c_stem = 3
            init_strides = [1] * self._num_init
        elif isinstance(self.use_stem, (list, tuple)):
            self.stems = []
            c_stem = self.stem_multiplier * self.init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stems.append(
                    ops.get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )
            self.stems = nn.ModuleList(self.stems)
            init_strides = [stem_stride] * self._num_init
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )
            init_strides = [1] * self._num_init

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = [c_stem] * self._num_init
        strides = [
            2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)
        ]

        for i_layer, stride in enumerate(strides):
            if stride > 1:
                num_channels *= stride
            num_out_channels = num_channels
            cell = cell_cls(
                op_cls,
                self.search_space,
                num_input_channels=prev_num_channels,
                num_out_channels=num_out_channels,
                stride=stride,
                prev_strides=init_strides + strides[:i_layer],
            )

            self.cells.append(cell)
            prev_num_channel = cell.num_out_channel()
            prev_num_channels.append(prev_num_channel)
            prev_num_channels = prev_num_channels[1:]

        self.lastact = nn.Identity()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.drop_rate and self.drop_rate > 0:
            self.dropout = nn.Dropout(p=self.drop_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(prev_num_channels[-1], self.num_classes)
        self.to(self.device)

        self.candidate_eval_no_grad = candidate_eval_no_grad
        self.assembled = 0
        self.candidate_map = weakref.WeakValueDictionary()

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
                self.total_flops += (
                    2
                    * inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * outputs.size(2)
                    * outputs.size(3)
                    / module.groups
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def sub_named_members(
        self, genotype, prefix="", member="parameters", check_visited=False
    ):
        prefix = prefix + ("." if prefix else "")
        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "cells":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix + mod_name):
                yield n, v
        for cell_idx, cell in enumerate(self.cells):
            for n, v in cell.sub_named_members(
                genotype[cell_idx],
                prefix=prefix + "cells.{}".format(cell_idx),
                member=member,
                check_visited=check_visited,
            ):
                yield n, v

    def __setstate__(self, state):
        super(RobSharedNet, self).__setstate__(state)
        self.candidate_map = weakref.WeakValueDictionary()

    def __getstate__(self):
        state = super(RobSharedNet, self).__getstate__()
        del state["candidate_map"]
        return state

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        cand_net = RobCandidateNet(
            self,
            rollout,
            gpus=self.gpus,
            eval_no_grad=self.candidate_eval_no_grad,
            calib_bn_batch=self.calib_bn_batch,
            calib_bn_num=self.calib_bn_num
        )
        # keep record of candidate networks that are not GCed, and call their `clear_cache`
        self.candidate_map[self.assembled] = cand_net
        self.assembled += 1
        return cand_net

    @classmethod
    def supported_rollout_types(cls):
        return ["dense_rob"]

    def _is_reduce(self, layer_idx):
        return (
            self.search_space.cell_layout[layer_idx]
            in self.search_space.reduce_cell_groups
        )

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, inputs, genotype, **kwargs):  # pylint: disable=arguments-differ
        if not self.use_stem:
            states = inputs
        elif isinstance(self.use_stem, (list, tuple)):
            stemed = inputs
            for stem in self.stems:
                stemed = stem(stemed)
            states = stemed
        else:
            stemed = self.stem(inputs)
            states = stemed
        states = [states] * self._num_init

        for cg_idx, cell in zip(self._cell_layout, self.cells):
            o_states = cell(states, genotype[cg_idx], **kwargs)
            states.append(o_states)
            states = states[1:]

        out = self.lastact(states[-1])
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self._flops_calculated = True

        return logits

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()
        for cand_net in self.candidate_map.values():
            cand_net.clear_cache()

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()
        for cand_net in self.candidate_map.values():
            cand_net.clear_cache()

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def save(self, path):
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    @classmethod
    def supported_data_types(cls):
        return ["image"]


class RobCandidateNet(CandidateNet):
    def __init__(
        self,
        super_net,
        rollout,
        gpus=tuple(),
        cache_named_members=False,
        eval_no_grad=True,
        calib_bn_batch=0,
            calib_bn_num=0,
    ):
        super(RobCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.search_space = super_net.search_space
        self.cache_named_members = cache_named_members

        self._flops_calculated = False
        self.total_flops = 0

        self.genotype_arch = rollout.arch
        self.genotype = rollout.genotype

        self.cached_advs = _Cache([], buffer_size=3)

        self.calib_bn_batch = calib_bn_batch
        self.calib_bn_num = calib_bn_num

    def clear_cache(self):
        self.cached_advs.clear()

    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net.forward(inputs, self.genotype_arch)

    def forward(self, inputs, single=False):  # pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs={"single": True})

    def named_parameters(
        self, prefix="", recurse=True
    ):  # pylint: disable=arguments-differ
        for n, v in self.super_net.named_parameters(prefix=prefix):
            yield n, v

    def named_buffers(
        self, prefix="", recurse=True
    ):  # pylint: disable=arguments-differ
        for n, v in self.super_net.named_buffers(prefix=prefix):
            yield n, v

    def active_named_members(
        self, member, prefix="", recurse=True, check_visited=False
    ):
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
        for n, v in self.super_net.sub_named_members(
            self.genotype_arch,
            prefix=prefix,
            member=member,
            check_visited=check_visited,
        ):
            if v in memo:
                continue
            memo.add(v)
            yield n, v

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        member_lst = []
        for n, v in itertools.chain(
            self.active_named_members(member="parameters", prefix=""),
            self.active_named_members(member="buffers", prefix=""),
        ):
            member_lst.append((n, v))
        state_dict = OrderedDict(member_lst)
        return state_dict

    def calib_bn(self, calib_data):

        for name, m in self.super_net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True
                m.running_mean.fill_(0)
                m.running_var.fill_(1)
                m.num_batches_tracked.fill_(0)
                m.momentum = None
                m.train()

        # forward once to get running mean and running var
        with torch.no_grad():
            for data in calib_data:
                inputs = data[0].to(self.get_device())
                self._forward(inputs)

        for name, m in self.super_net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.training = False
                m.track_running_stats = True

            self.super_net.drop_rate = 0.0
            self.super_net.drop_out_rate = 0.0

    def train_queue(self,
                    queue,
                    optimizer,
                    criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                    eval_criterions=None,
                    steps=1,
                    aggregate_fns=None,
                    **kwargs):
        assert steps > 0

        self._set_mode("train")

        aggr_ans = []
        for _ in range(steps):
            data = next(queue)
            data = _to_device(data, self.get_device())
            _, targets = data
            outputs = self.forward_data(*data, **kwargs)
            loss = criterion(data[0], outputs, targets)
            if eval_criterions:
                ans = utils.flatten_list(
                    [c(data[0], outputs, targets) for c in eval_criterions])
                aggr_ans.append(ans)
            self.zero_grad()
            loss.backward()
            optimizer.step()
            self.clear_cache()

        if eval_criterions:
            aggr_ans = np.asarray(aggr_ans).transpose()
            if aggregate_fns is None:
                # by default, aggregate batch rewards with MEAN
                aggregate_fns = [lambda perfs: np.mean(perfs) if len(perfs) > 0 else 0.]\
                                * len(aggr_ans)
            return [
                aggr_fn(ans) for aggr_fn, ans in zip(aggregate_fns, aggr_ans)
            ]
        return []

    def eval_queue(self,
                   queue,
                   criterions,
                   steps=1,
                   mode="eval",
                   aggregate_fns=None,
                   **kwargs):
        # BN running statistics calibration
        if self.calib_bn_num > 0:
            # check `calib_bn_num` first
            calib_num = 0
            calib_data = []
            calib_batch = 0
            while calib_num < self.calib_bn_num:
                if calib_batch == steps:
                    utils.getLogger("robustness plugin.{}".format(self.__class__.__name__)).warn(
                        "steps (%d) reached, true calib bn num (%d)", calib_num, steps)
                    break
                calib_data.append(next(queue))
                calib_num += len(calib_data[-1][1])
                calib_batch += 1
            self.calib_bn(calib_data)
        elif self.calib_bn_batch > 0:
            if self.calib_bn_batch > steps:
                utils.getLogger("robustness plugin.{}".format(self.__class__.__name__)).warn(
                    "eval steps (%d) < `calib_bn_batch` (%d). Only use %d batches.",
                    steps, self.calib_bn_steps, steps)
                calib_bn_batch = steps
            else:
                calib_bn_batch = self.calib_bn_batch
            # check `calib_bn_batch` then
            calib_data = [next(queue) for _ in range(calib_bn_batch)]
            self.calib_bn(calib_data)
        else:
            calib_data = []

        self._set_mode("eval") # Use eval mode after BN calibration

        aggr_ans = []
        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for i in range(steps):
                if i < len(calib_data):# self.calib_bn_batch:
                    data = calib_data[i]
                else:
                    data = next(queue)
                data = _to_device(data, self.get_device())
                outputs = self.forward_data(data[0], **kwargs)
                ans = utils.flatten_list(
                    [c(data[0], outputs, data[1]) for c in criterions])
                aggr_ans.append(ans)
                del outputs
                print("\reva step {}/{} ".format(i, steps), end="", flush=True)

        aggr_ans = np.asarray(aggr_ans).transpose()

        if aggregate_fns is None:
            # by default, aggregate batch rewards with MEAN
            aggregate_fns = [lambda perfs: np.mean(perfs) if len(perfs) > 0 else 0.]\
                            * len(aggr_ans)
        return [aggr_fn(ans) for aggr_fn, ans in zip(aggregate_fns, aggr_ans)]
