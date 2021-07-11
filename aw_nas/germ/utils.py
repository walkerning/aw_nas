import abc
import contextlib

import torch
import torch.nn as nn

from aw_nas.utils.exception import expect
from aw_nas import germ
from aw_nas.utils import nullcontext
from aw_nas.utils.common_utils import get_sub_kernel, _get_channel_mask


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

            name: the name registried on module, dupilicates are forbidden.

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
        if isinstance(choices, germ.Choices):
            self.max = choices.range()[1]
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
                return getattr(self.module, self.__dict__[f"{attr}_attr"])
        return super().__getattribute__(attr)

    def __setattr__(self, attr, val):
        if attr in self.__dict__.get("_attrs", {}):
            if isinstance(val, (torch.Tensor, torch.nn.Parameter)):
                self.module.register_buffer(self.__dict__[f"{attr}_attr"], val)
            else:
                self.module.__setattr__(self.__dict__[f"{attr}_attr"], val)
        super().__setattr__(attr, val)

    def is_none(self):
        return not isinstance(self.choices, germ.Choices)

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
                    kernel_transform_matrix = nn.Linear(
                        smaller * smaller, smaller * smaller, bias=False
                    )
                    torch.nn.init.eye_(kernel_transform_matrix.weight.data)

                    attr = "linear_{}to{}".format(larger, smaller)
                    self.registry(attr, ("{}_%s" % attr))

                    # requires grad
                    setattr(self, attr, kernel_transform_matrix)
        else:
            self.kernel_sizes = [choices]

    @contextlib.contextmanager
    def apply(self, conv, choice, ctx=None, detach=False):
        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return

            assert (
                choice in self.kernel_sizes
            ), f"except kernel size in {self.kernel_sizes}, got {choice} instead."

            ori_weight = conv.weight
            new_weight = self._transform_kernel(ori_weight, choice)
            if detach:
                new_weight = nn.Parameter(new_weight)
            conv._parameters["weight"] = new_weight
            conv.padding = (choice // 2,) * 2
            conv.kernel_size = (choice,) * 2
            yield

            if detach:
                return

            conv._parameters["weight"] = ori_weight
            conv.padding = (ori_weight.shape[-1] // 2,) * 2
            conv.kernel_size = (ori_weight.shape[-1],) * 2

    def _transform_kernel(self, origin_filter, kernel_size):
        if origin_filter.shape[-1] == kernel_size:
            return origin_filter
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
            sub_filter = get_sub_kernel(origin_filter, smaller).view(
                cur_filter.shape[0], cur_filter.shape[1], -1
            )
            sub_filter = sub_filter.view(-1, sub_filter.shape[-1])
            sub_filter = self.__getattr__("linear_{}to{}".format(larger, smaller))(
                sub_filter
            )
            sub_filter = sub_filter.view(
                origin_filter.shape[0], origin_filter.shape[1], smaller ** 2
            )
            sub_filter = sub_filter.view(
                origin_filter.shape[0], origin_filter.shape[1], smaller, smaller
            )
            cur_filter = sub_filter
        return cur_filter


class ChannelMaskHandler(MaskHandler):
    def __init__(self, ctx, module, name, choices, extra_attrs=None):
        super().__init__(ctx, module, name, choices, extra_attrs)

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
                    raise ValueError(
                        "BN-based channel selection is not supported yet, please apply BN "
                        "mask after convolution."
                    )

                elif isinstance(module, nn.Conv2d) and module.groups > 1 and axis == 1:
                    # no need to calculate mask idx
                    pass
                else:
                    mask_idx = _get_channel_mask(module.weight.data, choice, axis)
                    self.ctx.rollout.masks[self.choices.decision_id] = mask_idx

            if isinstance(module, nn.Conv2d):
                ori_weight = module.weight
                ori_bias = module.bias

                if module.groups > 1 and axis == 1:
                    # the input shape of depthwise is always 1
                    # only change groups number
                    # weight is sliced when axis == 0
                    ori_groups = module.groups
                    module.groups = int(choice)
                    module.in_channels = module.groups
                    module.out_channels = module.groups

                    yield
                    if detach:
                        return

                    module.groups = ori_groups
                    module.in_channels = ori_groups
                    module.out_channels = ori_groups

                else:
                    # regular conv
                    # or slice the output dimension of depthwise conv
                    ori_weight = module.weight
                    ori_bias = module.bias

                    new_weight = ori_weight.index_select(axis, mask_idx)
                    new_bias = ori_bias[mask_idx] if ori_bias is not None else None
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
                ori = {}
                module.num_features = len(mask_idx)
                for k in ["running_mean", "running_var", "weight", "bias"]:
                    ori[k] = getattr(module, k)
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

                module.num_features = len(ori["weight"])
                for k in ["running_mean", "running_var", "weight", "bias"]:
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

    @contextlib.contextmanager
    def apply(self, module, stride, ctx=None, detach=False):
        if ctx is None:
            ctx = nullcontext()
        with ctx:
            if self.is_none():
                yield
                return
            if isinstance(stride, (tuple, list)):
                pass
            else:
                stride = (int(stride),) * 2
            ori_stride = module.stride
            module.stride = stride

            yield
            if detach:
                return

            module.stride = ori_stride
