import abc
import contextlib
import functools
import math

import numpy as np
import torch
import torch.nn as nn
from aw_nas import germ
from aw_nas.utils import make_divisible, nullcontext
from aw_nas.utils.common_utils import (_get_channel_mask, _get_feature_mask,
                                       get_sub_kernel)
from aw_nas.utils.exception import InvalidUseException, expect

divisor_fn = functools.partial(make_divisible, divisor=8)


def gcd(*args):
    return functools.reduce(math.gcd, args)


class MaskHandler(object):
    REGISTRED = set()

    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        """
        registry mask attributes in module

        Args:
            module: pytorch module to registry mask buffer.
                !NOTICE that the mask handler needs to be deleted when you want
                to release the memory of the module, or it would cause memory
                leak.

            name: the name registried on module, duplicates are forbidden.

            values: the default attributes of MaskHandler includes max, mask
                and hash, which need to be assigned before __init__ is called.

            extra_attrs: other attributes.
                format: {
                    name: attr_format,
                    ...
                }

        """

        self.ctx = ctx
        self.module = module
        self.name = name
        self._attrs = set()

        # duplicates check
        # hash_code = hash(module)
        if (id(module), name) in MaskHandler.REGISTRED:
            raise ValueError(
                f"name `{name}` has already been registried on " f"module {module}."
            )

        MaskHandler.REGISTRED.add((id(module), name))

        # --- attr ---
        self.registry("max", "max_{}")

        self.choices = choices
        if isinstance(choices, germ.BaseChoices):
            self.max = choices.range()[1]
        elif isinstance(choices, germ.BaseDecision):
            raise InvalidUseException("Other decisions not suppported!")
        else:
            self.max = choices

        if extra_attrs is not None:
            assert isinstance(
                extra_attrs, dict
            ), "extra_attrs: {attr_name: attr_format, ...}"
            for k, v in self.extra_attrs.items():
                self.registry(self, k, v)

    def registry(self, attr_name, attr_format, exists_ok=False):
        if attr_name in self._attrs and not exists_ok:
            raise ValueError(f"{attr_name} is already registried.")

        # assert isinstance(attr_val, (list, tuple)) and len(attr_val) == 2, \
        #    "Assert that attr_val is (attr_format, attr_value)"
        setattr(self, f"{attr_name}_attr", attr_format.format(self.name))
        self._attrs.add(attr_name)

        # setattr(self, attr_name, attr_val[1])

    def __getattr__(self, attr):
        if "_attrs" in self.__dict__:
            if attr in self.__dict__["_attrs"]:
                if self.__dict__[f"{attr}_attr"] in self.module._parameters.keys():
                    return self.module._parameters[self.__dict__[f"{attr}_attr"]]
                elif self.__dict__[f"{attr}_attr"] in self.module._buffers.keys():
                    return self.module._buffers[self.__dict__[f"{attr}_attr"]]
                return getattr(self.module, self.__dict__[f"{attr}_attr"])
        return super().__getattribute__(attr)

    def __setattr__(self, attr, val):
        if attr in self.__dict__.get("_attrs", {}):
            if isinstance(val, torch.nn.Parameter):
                self.module.register_parameter(self.__dict__[f"{attr}_attr"], val)
            elif isinstance(val, torch.Tensor):
                self.module.register_buffer(self.__dict__[f"{attr}_attr"], val)
            else:
                self.module.__setattr__(self.__dict__[f"{attr}_attr"], val)
        super().__setattr__(attr, val)

    def is_none(self):
        return not isinstance(self.choices, germ.BaseDecision)

    def __del__(self):
        """
        update MaskHandler.REGISTRED if del
        """
        # MaskHandler.REGISTRED.add((id(module), name))
        MaskHandler.REGISTRED = set(filter(lambda x: x[0] != id(self.module), MaskHandler.REGISTRED))
        # super().__del__()
        if hasattr(super(), "__del__"):
            super().__del__()

    @abc.abstractmethod
    def apply(self, module, choice, ctx=None, detach=False):
        pass


class KernelMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)

        if isinstance(choices, germ.Choices):
            kernel_sizes = sorted(self.choices.choices)
            self.kernel_sizes = kernel_sizes
            for smaller, larger in reversed(
                list(zip(kernel_sizes[:-1], kernel_sizes[1:]))
            ):
                if self.max >= larger:
                    kernel_transform_matrix = torch.nn.Parameter(torch.eye(
                        smaller * smaller,
                    ))

                    attr = "linear_{}to{}".format(larger, smaller)
                    self.registry(attr, ("{}_%s" % attr))

                    # requires grad
                    setattr(self, attr, kernel_transform_matrix)
        else:
            self.kernel_sizes = [choices]
        # check for conv layer
        assert isinstance(module, germ.SearchableConv), \
            "KernelMaskHandler can only support searchable conv"

    @contextlib.contextmanager
    def apply(self, module, choice, ctx=None, detach=False):
        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return

            assert (
                choice in self.kernel_sizes
            ), f"except kernel size in {self.kernel_sizes}, got {choice} instead."

            ori_weight = module.weight
            ori_padding = module.padding
            ori_kernel_size = module.kernel_size

            module.padding = (choice // 2,) * 2
            module.kernel_size = (choice,) * 2
            new_weight = self._transform_kernel(ori_weight, choice)
            if detach:
                new_weight = nn.Parameter(new_weight)
            module._parameters["weight"] = new_weight

            yield
            if detach:
                return

            module._parameters["weight"] = ori_weight
            module.padding = ori_padding
            module.kernel_size = ori_kernel_size

    def _transform_kernel(self, origin_filter, kernel_size):
        if origin_filter.shape[-1] == kernel_size:
            return origin_filter
        # return get_sub_kernel(origin_filter, kernel_size)
        cur_filter = origin_filter
        expect(
            cur_filter.shape[-1] > kernel_size,
            "The kernel size must be less than origin kernel size {}, got {} instead.".format(
                origin_filter.shape[-1], kernel_size
            ),
            ValueError,
        )
        for smaller, larger in reversed(
            list(zip(self.kernel_sizes[:-1], self.kernel_sizes[1:]))
        ):
            if cur_filter.shape[-1] < larger:
                continue
            if kernel_size >= larger:
                break
            sub_filter = get_sub_kernel(cur_filter, smaller).view(
                cur_filter.shape[0] * cur_filter.shape[1], -1
            )
            # sub_filter = sub_filter.view(-1, sub_filter.shape[-1])
            transform_matrix = self.__getattr__("linear_{}to{}".format(larger, smaller))
            sub_filter = torch.mm(sub_filter, transform_matrix.to(sub_filter.device))
            # sub_filter = sub_filter.view(
            #     cur_filter.shape[0], cul_filter.shape[1], smaller ** 2
            # )
            sub_filter = sub_filter.view(
                cur_filter.shape[0], cur_filter.shape[1], smaller, smaller
            )
            cur_filter = sub_filter
        return cur_filter


class ChannelMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)
        if isinstance(module, germ.SearchableConv):
            assert module.g_choices == 1 or \
                module.g_choices == module.ci_choices == module.co_choices, \
                "ChannelMaskHandler support regular conv or depth-wise conv"
        elif not isinstance(module, germ.SearchableBN):
            raise NotImplementedError(
                "ChannelMaskHandler support bn and conv"
            )

    @contextlib.contextmanager
    def apply(self, module, choice, axis=1, ctx=None, detach=False):
        """
        Args:
            module: bound module
            choice: choice value
            axis: 0 means applying mask on out channels, 1 means applying mask on in channels.
        """
        assert axis in (0, 1)

        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return

            assert hasattr(
                self.ctx, "rollout"
            ), "context should have rollout attribute."
            mask_idx = self.ctx.rollout.masks.get(self.choices.decision_id)
            if mask_idx is None:
                if isinstance(module, nn.BatchNorm2d):
                    if module.weight is None:
                        raise ValueError(
                            "ChannelMaskHandler does not support infering mask affine-less "
                            "BatchNorm2d before Conv2d."
                        )
                    mask_idx = sorted(
                        module.weight.data.argsort()[-choice:].detach().numpy()
                    )
                    self.ctx.rollout.masks[self.choices.decision_id] = mask_idx
                elif isinstance(module, nn.Conv2d) and module.groups > 1 and axis == 1:
                    # no need to calculate mask idx
                    pass
                else:
                    mask_idx = _get_channel_mask(module.weight.data, choice, axis)
                    self.ctx.rollout.masks[self.choices.decision_id] = mask_idx
            if mask_idx is not None:
                assert len(mask_idx) == choice

            if isinstance(module, nn.Conv2d):
                if module.groups > 1 and axis == 1:
                    # the input shape of depthwise is always 1
                    # only change groups number
                    # weight is sliced when axis == 0
                    ori_in_channels = module.in_channels
                    module.in_channels = choice

                    yield
                    if detach:
                        return

                    module.in_channels = ori_in_channels
                else:
                    # regular conv
                    # or slice the output dimension of depthwise conv
                    ori_weight = module.weight
                    ori_bias = module.bias

                    new_weight = ori_weight.index_select(axis, mask_idx.to(ori_weight.device))
                    if axis == 1:
                        new_bias = ori_bias
                    else:
                        new_bias = ori_bias.index_select(axis, mask_idx.to(ori_bias.device)) if ori_bias is not None else None
                    if detach:
                        new_weight = nn.Parameter(new_weight)
                        if new_bias is not None:
                            new_bias = nn.Parameter(new_bias)

                    if axis == 0:
                        module.out_channels = len(mask_idx)
                    elif axis == 1:
                        module.in_channels = len(mask_idx)

                    module._parameters["weight"] = new_weight
                    module._parameters["bias"] = new_bias

                    yield
                    if detach:
                        return

                    module._parameters["weight"] = ori_weight
                    module._parameters["bias"] = ori_bias

                    if axis == 0:
                        module.out_channels = ori_weight.shape[0]
                    elif axis == 1:
                        module.in_channels = ori_weight.shape[1]

            elif isinstance(module, nn.BatchNorm2d):
                ori = dict()
                ori_num_features = module.num_features
                module.num_features = choice

                if module.affine:
                    keys = ["running_mean", "running_var", "weight", "bias"]
                else:
                    keys = ["running_mean", "running_var"]

                for k in keys:
                    ori[k] = getattr(module, k)
                    if ori[k] is None:
                        continue
                    new_attr = ori[k][mask_idx]
                    if k in module._parameters:
                        if detach:
                            new_attr = nn.Parameter(new_attr)
                        module._parameters[k] = new_attr
                    else:
                        module._buffers[k] = new_attr

                yield
                if detach:
                    return

                module.num_features = ori_num_features
                for k in keys:
                    if k in module._parameters:
                        module._parameters[k] = ori[k]
                    else:
                        module._buffers[k] = ori[k]
            else:
                raise ValueError(
                    "ChannelMaskHandler only support nn.Conv2d and "
                    "nn.BatchNorm2d now."
                )


class StrideMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)
        # check for conv layer
        assert isinstance(module, germ.SearchableConv), \
            "StrideMaskHandler can only support searchable conv"

    @contextlib.contextmanager
    def apply(self, module, choice, ctx=None, detach=False):
        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return
            if isinstance(choice, (tuple, list)):
                pass
            else:
                stride = (int(choice),) * 2

            ori_stride = module.stride
            module.stride = stride

            yield
            if detach:
                return

            module.stride = ori_stride

def check_depth_wise(module):
    if module.g_choices == module.ci_choices == module.co_choices:
        depth_wise_flag = True
    else:
        depth_wise_flag = False
    return depth_wise_flag

class GroupMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)
        # check for conv layer
        assert isinstance(module, germ.SearchableConv), \
            "GroupMaskHandler can only support searchable conv"
        self.depth_wise_flag = check_depth_wise(module)

    @contextlib.contextmanager
    def apply(self, module, choice, ctx=None, detach=False):
        """
        Args:
            module: bound module
            choice: choice value
        """
        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return

            if self.depth_wise_flag:
                ori_groups = module.groups
                module.groups = choice

                yield
                if detach:
                    return

                module.groups = ori_groups
                return

            assert (
                choice % module.groups == 0
            ), f"choice must be divisible by module.groups, got {choice} and {module.groups} instead."

            ori_groups = module.groups
            module.groups = int(choice)
            ori_weight = module.weight

            # num_inshape = 1 when choice == in_channels
            num_inshape = module.in_channels // choice
            num_outshape = module.out_channels // choice
            sub_groups_per_group = choice // ori_groups
            out_index = list()
             # get the input-channel indices in one biggest group
            for offset in range(sub_groups_per_group):
                for _ in range(num_outshape):
                    out_index.append([i + offset * num_inshape for i in range(num_inshape)])
            # repeat the input-channel indices for `ori_groups` times
            out_index = out_index * ori_groups

            assert len(out_index) == module.out_channels
            out_index = torch.tensor(out_index, dtype=torch.long).to(ori_weight.device)
            out_index = out_index.unsqueeze(-1).unsqueeze(-1)
            # repeat along kernel size height/width
            out_index = out_index.repeat(
                1, 1, ori_weight.shape[-2], ori_weight.shape[-1]
            )
            new_weight = ori_weight.gather(1, out_index)
            assert new_weight.shape[0] == module.out_channels
            assert new_weight.shape[1] == num_inshape
            if detach:
                new_weight = nn.Parameter(new_weight)

            module._parameters["weight"] = new_weight
            yield
            if detach:
                return

            module._parameters["weight"] = ori_weight
            module.groups = ori_groups

# set and find
def _set_common_index(groups_num, step, select_num):
    # check for number or list
    if isinstance(select_num, (int, float)):
        select_num = range(select_num)
    m_idx = list()
    for k in range(groups_num):
        for j in select_num:
            m_idx.append(j + k * step)
    return torch.LongTensor(m_idx)

class OrdinalChannelMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)
        assert isinstance(module, (germ.SearchableConv, germ.SearchableBN)), \
            "OrdinalChannelMaskHandler support for only searchable conv and bn"
        if isinstance(module, germ.SearchableConv):
            self.depth_wise_flag = check_depth_wise(module)
        else:
            self.depth_wise_flag = False

    @contextlib.contextmanager
    def apply(self, module, choice, axis=1, ctx=None, detach=False):
        """
        Args:
            module: bound module
            choice: choice value
            axis: 0 means applying mask on out channels, 1 means applying mask on in channels.
        """
        assert axis in (0, 1)

        if ctx is None:
            ctx = nullcontext()

        with ctx:
            if self.is_none():
                yield
                return

            assert hasattr(
                self.ctx, "rollout"
            ), "context should have rollout attribute."

            # check for depth wise flag
            if isinstance(module, germ.SearchableConv) and self.depth_wise_flag:
                # for depth wise in channels
                if axis == 1:
                    ori_in_channels = module.in_channels
                    module.in_channels = choice

                    yield
                    if detach:
                        return

                    module.in_channels = ori_in_channels
                else:
                    ori_weight = module.weight
                    ori_bias = module.bias
                    ori_out_channels = module.out_channels
                    module.out_channels = choice
                    assert module.out_channels == module.groups, \
                        "output channels should same with group in depth-wise conv"

                    _mask_idx = _set_common_index(module.groups, 1, 1)
                    _mask_idx = _mask_idx.to(ori_weight.device)
                    new_weight = ori_weight.index_select(axis, _mask_idx)
                    new_bias = ori_bias.index_select(axis, _mask_idx) if ori_bias is not None else None
                    if detach:
                        new_weight = nn.Parameter(new_weight)
                        if new_bias is not None:
                            new_bias = nn.Parameter(new_bias)
                    module._parameters["weight"] = new_weight
                    module._parameters["bias"] = new_bias

                    yield
                    if detach:
                        return

                    module.out_channels = ori_out_channels
                    module._parameters["weight"] = ori_weight
                    module._parameters["bias"] = ori_bias
                return

            mask_idx = self.ctx.rollout.masks.get(self.choices.decision_id)
            if mask_idx is None:
                if isinstance(module, nn.BatchNorm2d):
                    raise ValueError(
                        "OrdinalChannelMaskHandler is not support infering mask by "
                        "BatchNorm2d yet."
                    )
                # for in channels
                elif isinstance(module, nn.Conv2d) and axis == 1:
                    # no need to calculate mask idx
                    pass
                else:
                    mask_idx = _set_common_index(
                        module.groups, module.out_channels // module.groups, int(choice // module.groups)
                    )
                    self.ctx.rollout.masks[self.choices.decision_id] = mask_idx
            if mask_idx is not None:
                assert len(mask_idx) == choice

            if isinstance(module, nn.Conv2d):
                assert (
                    module.in_channels % module.groups == 0 and \
                    module.out_channels % module.groups == 0 and \
                    choice % module.groups == 0
                ), f"choice must divisible by module.groups, got {choice} and {module.groups} instead."


                if axis == 1:
                    # input channel handler
                    ori_in = module.in_channels
                    ori_weight = module.weight
                    module.in_channels = choice

                    _mask_idx = torch.LongTensor(list(range(choice // module.groups)))
                    _mask_idx = _mask_idx.to(ori_weight.device)
                    new_weight = ori_weight.index_select(axis, _mask_idx)
                    if detach:
                        new_weight = nn.Parameter(new_weight)
                    module._parameters["weight"] = new_weight

                    yield
                    if detach:
                        return

                    module.in_channels = ori_in
                    module._parameters["weight"] = ori_weight
                else:
                    # output channel handler
                    ori_weight = module.weight
                    ori_bias = module.bias
                    ori_out = module.out_channels
                    module.out_channels = choice

                    _mask_idx = mask_idx.to(ori_weight.device)
                    new_weight = ori_weight.index_select(axis, _mask_idx)
                    new_bias = ori_bias.index_select(axis, _mask_idx) if ori_bias is not None else None
                    if detach:
                        new_weight = nn.Parameter(new_weight)
                        if new_bias is not None:
                            new_bias = nn.Parameter(new_bias)
                    module._parameters["weight"] = new_weight
                    module._parameters["bias"] = new_bias

                    yield
                    if detach:
                        return

                    module.out_channels = ori_out
                    module._parameters["weight"] = ori_weight
                    module._parameters["bias"] = ori_bias
            elif isinstance(module, nn.BatchNorm2d):
                ori = dict()
                ori_num_features = module.num_features
                module.num_features = choice

                for k in ["running_mean", "running_var", "weight", "bias"]:
                    ori[k] = getattr(module, k)
                    if ori[k] is None:
                        continue
                    new_attr = ori[k].index_select(0, mask_idx.to(ori[k].device))
                    if k in module._parameters:
                        if detach:
                            new_attr = nn.Parameter(new_attr)
                        module._parameters[k] = new_attr
                    else:
                        module._buffers[k] = new_attr

                yield
                if detach:
                    return

                module.num_features = ori_num_features
                for k in ["running_mean", "running_var", "weight", "bias"]:
                    if k in module._parameters:
                        module._parameters[k] = ori[k]
                    else:
                        module._buffers[k] = ori[k]
            else:
                raise ValueError(
                    "OrdinalChannelMaskHandler only support nn.Conv2d and "
                    "nn.BatchNorm2d now."
                )

class FeatureMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)

    @contextlib.contextmanager
    def apply(self, module, choice, axis=1, ctx=None, detach=False):
        """
        Args:
            module: bound module
            choice: choice value
            axis: 0 means applying mask on out features, 1 means applying mask on in features.
        """
        assert axis in (0, 1)

        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return

            assert hasattr(
                self.ctx, "rollout"
            ), "context should have rollout attribute."
            mask_idx = self.ctx.rollout.masks.get(self.choices.decision_id)
            if mask_idx is None:
                # mask_idx = _get_feature_mask(module.weight.data, choice, axis)
                mask_idx = _set_common_index(choice, 1, 1)
                self.ctx.rollout.masks[self.choices.decision_id] = mask_idx

            if isinstance(module, nn.Linear):
                ori_weight = module.weight
                ori_bias = module.bias

                if axis == 0:
                    ori_features = module.out_features
                    module.out_features = choice
                elif axis == 1:
                    ori_features = module.in_features
                    module.in_features = choice

                # regular linear
                _mask_idx = mask_idx.to(ori_weight.device)
                new_weight = ori_weight.index_select(axis, _mask_idx)
                if ori_bias is not None:
                    if axis == 0:
                        new_bias = ori_bias.index_select(axis, _mask_idx)
                    else:
                        new_bias = ori_bias
                else:
                    new_bias = None
                if detach:
                    new_weight = nn.Parameter(new_weight)
                    if new_bias is not None:
                        new_bias = nn.Parameter(new_bias)
                module._parameters["weight"] = new_weight
                module._parameters["bias"] = new_bias

                yield
                if detach:
                    return

                module._parameters["weight"] = ori_weight
                module._parameters["bias"] = ori_bias
                if axis == 0:
                    module.out_features = ori_features
                elif axis == 1:
                    module.in_features = ori_features
            else:
                raise ValueError(
                    "FeatureMaskHandler only support nn.Linear now."
                )
