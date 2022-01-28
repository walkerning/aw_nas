# pylint: disable=arguments-differ
from collections import abc as collection_abcs
from collections import OrderedDict
import random

from torch import nn
from torch.nn import functional as F

from aw_nas import ops, germ
from aw_nas.utils import nullcontext
from .utils import *
from aw_nas.utils.common_utils import _get_channel_mask, _get_feature_mask


class GermOpOnEdge(germ.SearchableBlock):
    """
    A node with operation on edge
    """

    NAME = "op_on_edge"

    def __init__(
        self,
        ctx,
        num_input_nodes,
        # basic info
        from_nodes,
        op_list,
        aggregation="sum",
        # optional decisions
        from_nodes_choices=None,
        edgeops_choices_dict=None,
        aggregation_op_choice=None,
        allow_repeat_choice=False,
        **op_kwargs
    ):
        super().__init__(ctx)

        self.from_nodes = from_nodes
        self.num_input_nodes = num_input_nodes
        self.op_list = op_list
        self.aggregation = aggregation
        self.allow_repeat_choice = allow_repeat_choice
        self.op_kwargs = op_kwargs

        # select
        if from_nodes_choices is not None:
            self.from_nodes_choices = from_nodes_choices
        else:
            self.from_nodes_choices = germ.Choices(
                from_nodes, size=num_input_nodes, replace=allow_repeat_choice
            )
        self.select = GermSelect(ctx, self.from_nodes_choices)

        # edge ops
        if edgeops_choices_dict is None:
            self.edge_ops = nn.ModuleDict(
                {
                    str(from_node): GermMixedOp(ctx, op_list, **op_kwargs)
                    for from_node in from_nodes
                }
            )
            self.edgeops_choices_dict = {
                from_node: op.op_choice for from_node, op in self.edge_ops.items()
            }
        else:
            self.edge_ops = nn.ModuleDict(
                {
                    from_node: GermMixedOp(ctx, op_list, op_choices, **op_kwargs)
                    for from_node, op_choices in edgeops_choices_dict.items()
                }
            )
            self.edgeops_choices_dict = edgeops_choices_dict

        # aggregation
        if isinstance(aggregation, (list, tuple)):
            aggregation_op = GermMixedOp(
                ctx,
                [ops.get_concat_op(op_name) for op_name in aggregation],
                op_choice=aggregation_op_choice,
            )
            self.aggregation_op = aggregation_op
            self.aggregation_op_choice = self.aggregation_op.op_choice
            # TODO: elementwise searchable with concat, need to align the dimension using 1x1
        else:
            self.aggregation_op = ops.get_concat_op(aggregation)
            self.aggregation_op_choice = None

    def forward_rollout(self, rollout, inputs):
        from_nodes, features_list = self.select(inputs)
        features_list = [
            self.edge_ops[str(from_node)](features)
            for from_node, features in zip(from_nodes, features_list)
        ]
        return self.aggregation_op(features_list)

    def finalize_rollout(self, rollout):
        # in-place finalize
        super().finalize_rollout(rollout)

        # drop unused edges
        final_from_nodes = self.select.from_nodes
        self.edge_ops = nn.ModuleDict(
            {
                str(from_node): self.edge_ops[str(from_node)]
                for from_node in final_from_nodes
            }
        )
        return self

    def finalize_rollout_outplace(self, rollout):
        # Outplace finalize could be clearer
        select = self.select.finalize_rollout(rollout)
        edge_ops = nn.ModuleDict(
            {
                str(from_node): self.edge_ops[str(from_node)].finalize_rollout(rollout)
                for from_node in select.from_nodes
            }
        )
        if isinstance(self.aggregation_op, GermMixedOp):
            aggr = self.aggregation_op.finalize_rollout(rollout)
        else:
            aggr = self.aggregation
        return FinalOpOnEdge(select, edge_ops, aggr)


class FinalOpOnEdge(nn.Module):
    def __init__(self, select, edge_ops, aggr):
        super().__init__()
        self.select = select
        self.edge_ops = edge_ops
        self.aggregation_op = aggr

    def forward(self, inputs):
        from_nodes, features_list = self.select(inputs)
        features_list = [
            self.edge_ops[str(from_node)](features)
            for from_node, features in zip(from_nodes, features_list)
        ]
        return self.aggregation_op(features_list)


class GermOpOnNode(germ.SearchableBlock):
    """
    A node with operation on node
    """

    NAME = "op_on_node"

    def __init__(
        self,
        ctx,
        num_input_nodes,
        # basic info
        from_nodes,
        op_list,
        aggregation="sum",
        # optional decisions
        from_nodes_choices=None,
        op_choice=None,
        aggregation_op_choice=None,
        allow_repeat_choice=False,
        **op_kwargs
    ):
        super().__init__(ctx)

        self.from_nodes = from_nodes
        self.num_input_nodes = num_input_nodes
        self.op_list = op_list
        self.aggregation = aggregation
        self.allow_repeat_choice = allow_repeat_choice
        self.op_kwargs = op_kwargs

        # select
        if from_nodes_choices is not None:
            self.from_nodes_choices = from_nodes_choices
        else:
            self.from_nodes_choices = germ.Choices(
                from_nodes, size=num_input_nodes, replace=allow_repeat_choice
            )
        self.select = GermSelect(ctx, self.from_nodes_choices)

        # aggregation
        if isinstance(aggregation, (list, tuple)):
            self.aggregation_op = GermMixedOp(
                ctx,
                [ops.get_concat_op(op_name) for op_name in aggregation],
                op_choice=aggregation_op_choice,
            )
            self.aggregation_op_choice = self.aggregation_op.op_choice
            # TODO: elementwise searchable with concat, need to align the dimension using 1x1
        else:
            self.aggregation_op = ops.get_concat_op(aggregation)
            self.aggregation_op_choice = None

        # op
        self.op = GermMixedOp(ctx, op_list, op_choice=op_choice, **op_kwargs)
        self.op_choice = self.op.op_choice

    def forward_rollout(self, rollout, inputs):
        _, features = self.select(inputs)
        aggregate = self.aggregation_op(features)
        return self.op(aggregate)


class GermSelect(germ.SearchableBlock):
    NAME = "germ_select"

    def __init__(self, ctx, from_nodes):
        super().__init__(ctx)
        self.from_nodes = from_nodes
        if isinstance(from_nodes, germ.BaseDecision):
            self.num_from_nodes = from_nodes.num_choices
        else:
            self.num_from_nodes = len(from_nodes)

    def forward_rollout(self, rollout, inputs):
        from_nodes = self._get_decision(self.from_nodes, rollout)
        return from_nodes, [inputs[node_ind] for node_ind in from_nodes]

    def finalize_rollout_outplace(self, rollout):
        return self.finalize_rollout(rollout)

    def finalize_rollout(self, rollout):
        from_nodes = self._get_decision(self.from_nodes, rollout)
        if not isinstance(from_nodes, collection_abcs.Iterable):
            from_nodes = [from_nodes]
        return GermSelect(ctx=None, from_nodes=from_nodes)

    def __repr__(self):
        return "GermSelect({})".format(self.from_nodes)

class GermMixedOp(germ.SearchableBlock):
    NAME = "mixed_op"

    def __init__(self, ctx, op_list, op_choice=None, **op_kwargs):
        super().__init__(ctx)

        self.op_list = op_list
        if op_choice is None:
            # initialize new choice
            self.op_choice = germ.Choices(
                choices=list(range(len(self.op_list))), size=1
            )
        else:
            # directly use the passed-in `op_choice`
            self.op_choice = op_choice
        self.op_kwargs = op_kwargs
        # initialize ops, each item in `op_list` can be a nn.Module subclass or a string,
        # if it is a string, the class would be got by calling `aw_nas.ops.get_op`
        self.p_ops = nn.ModuleList()
        for op_cls in self.op_list:
            if isinstance(op_cls, nn.Module):
                # already an nn.Module
                # e.g., shared modules, aggregation ops...
                op = op_cls
            else:
                if not isinstance(op_cls, type):
                    op_cls = ops.get_op(op_cls)
                else:
                    assert issubclass(op_cls, nn.Module)
                op = op_cls(**self.op_kwargs)
            self.p_ops.append(op)

    def forward(self, inputs):
        op_index = self._get_decision(self.op_choice, self.ctx.rollout)
        return self.p_ops[op_index](inputs)

    def finalize_rollout_outplace(self, rollout):
        return self.finalize_rollout(rollout)

    def finalize_rollout(self, rollout):
        op_index = self._get_decision(self.op_choice, rollout)
        op = self.p_ops[op_index]
        if isinstance(op, germ.SearchableBlock):
            op = op.finalize_rollout(rollout)
        return op

class AnyKernelConv(nn.Conv2d):
    def __init__(self, padding_flag, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.padding_flag = bool(random.randint(0, 1))
        assert padding_flag in [0, 1]
        # self.register_buffer("padding_flag", torch.tensor(padding_flag))
        self.padding_flag = padding_flag

    def forward(self, inputs):
        outputs = super().forward(inputs)
        # check for even kernel
        if self.kernel_size[0] % 2 == 0 and \
            self.padding[0] >= self.kernel_size[0] // 2:
            # fake output
            assert len(outputs.shape) == 4, \
                "The outputs should have 4 dim"
            # if self.padding_flag.item() == 1:
            if self.padding_flag == 1:
                outputs = outputs[:, :, :-1, :-1]
            else:
                outputs = outputs[:, :, 1:, 1:]
        return outputs

class SearchableConv(germ.SearchableBlock, AnyKernelConv):
    NAME = "conv"

    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        groups=1,
        force_use_ordinal_channel_handler=True,
        **kwargs
    ):
        super().__init__(ctx)

        self.ci_choices = in_channels
        self.co_choices = out_channels
        self.k_choices = kernel_size
        self.s_choices = stride
        self.g_choices = groups
        self.force_use_ordinal_channel_handler = force_use_ordinal_channel_handler

        if groups == 1 or out_channels == in_channels == groups:
            # regular conv or depthwise conv
            # in case for breaking other code
            if isinstance(groups, germ.BaseDecision):
                groups = groups.range()[1]

            self.g_handler = GroupMaskHandler(ctx, self, "groups", self.g_choices)
            if not self.force_use_ordinal_channel_handler:
                self.ci_handler = ChannelMaskHandler(ctx, self, "in_channels", in_channels)
                self.co_handler = ChannelMaskHandler(ctx, self, "out_channels", out_channels)
            else:
                self.ci_handler = OrdinalChannelMaskHandler(ctx, self, "in_channels", in_channels)
                self.co_handler = OrdinalChannelMaskHandler(ctx, self, "out_channels", out_channels)
            self.k_handler = KernelMaskHandler(ctx, self, "kernel_size", kernel_size)
            self.s_handler = StrideMaskHandler(ctx, self, "stride", stride)

        else:
            # group share weight conv
            if isinstance(groups, germ.BaseDecision):
                groups = gcd(*groups.choices)

            self.g_handler = GroupMaskHandler(ctx, self, "groups", self.g_choices)
            self.ci_handler = OrdinalChannelMaskHandler(ctx, self, "in_channels", in_channels)
            self.co_handler = OrdinalChannelMaskHandler(ctx, self, "out_channels", out_channels)
            self.k_handler = KernelMaskHandler(ctx, self, "kernel_size", kernel_size)
            self.s_handler = StrideMaskHandler(ctx, self, "stride", stride)

        _modules = self._modules
        _parameters = self._parameters
        _buffers = self._buffers
        padding_flag = random.randint(0, 1)
        AnyKernelConv.__init__(
            self,
            padding_flag=padding_flag,
            in_channels=self.ci_handler.max,
            out_channels=self.co_handler.max,
            kernel_size=self.k_handler.max,
            stride=self.s_handler.max,
            padding=self.k_handler.max // 2,
            bias=bias,
            groups=groups,
            **kwargs
        )
        self._modules.update(_modules)
        self._parameters.update(_parameters)
        self._buffers.update(_buffers)

    def rollout_context(self, rollout=None, detach=False):
        if rollout is None:
            return nullcontext()
        r_g = self._get_decision(self.g_handler.choices, rollout)
        # stride
        r_s = self._get_decision(self.s_handler.choices, rollout)
        # kernel size
        r_k_s = self._get_decision(self.k_handler.choices, rollout)
        # out channels
        r_o_c = self._get_decision(self.co_handler.choices, rollout)
        r_i_c = self._get_decision(self.ci_handler.choices, rollout)

        ctx = self.g_handler.apply(self, r_g, ctx=None, detach=detach)
        ctx = self.ci_handler.apply(self, r_i_c, axis=1, ctx=ctx, detach=detach)
        ctx = self.co_handler.apply(self, r_o_c, axis=0, ctx=ctx, detach=detach)
        ctx = self.k_handler.apply(self, r_k_s, ctx=ctx, detach=detach)
        ctx = self.s_handler.apply(self, r_s, ctx=ctx, detach=detach)
        return ctx

    def forward(self, inputs):
        with self.rollout_context(self.ctx.rollout):
            out = super().forward(inputs)
        return out

    def finalize_rollout(self, rollout):
        with self.rollout_context(rollout, detach=True):
            conv = AnyKernelConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias is not None,
                groups=self.groups,
                dilation=self.dilation,
            )
            conv.weight.data.copy_(self.weight.data)
            if self.bias is not None:
                conv.bias.data.copy_(self.bias.data)
        return conv


class SearchableBN(germ.SearchableBlock, nn.BatchNorm2d):
    NAME = "bn"

    def __init__(self, ctx, channels, force_use_ordinal_channel_handler=True, **kwargs):
        super().__init__(ctx)
        self.c_choices = channels
        self.force_use_ordinal_channel_handler = force_use_ordinal_channel_handler
        if self.force_use_ordinal_channel_handler:
            self.handler = OrdinalChannelMaskHandler(ctx, self, "channel", channels)
        else:
            self.handler = ChannelMaskHandler(ctx, self, "channel", channels)
        nn.BatchNorm2d.__init__(self, self.handler.max, **kwargs)

    def rollout_context(self, rollout, detach=False):
        co = self._get_decision(self.handler.choices, rollout)
        return self.handler.apply(self, co, detach=detach)

    def forward(self, inputs):
        with self.rollout_context(self.ctx.rollout):
            out = super().forward(inputs)
        return out

    def finalize_rollout(self, rollout):
        with self.rollout_context(rollout, True):
            bn = nn.BatchNorm2d(
                self.num_features,
                self.eps,
                self.momentum,
                self.affine,
                self.track_running_stats,
            )
        bn.weight.data.copy_(self.weight.data)
        bn.bias.data.copy_(self.bias.data)
        return bn


class SearchableConvBNBlock(germ.SearchableBlock):
    NAME = "conv_bn"

    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        force_use_ordinal_channel_handler=False,
        conv_cfg={},
        norm_cfg={},
    ):
        super().__init__(ctx)

        self.conv = SearchableConv(
            ctx,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=False,
            groups=groups,
            force_use_ordinal_channel_handler=force_use_ordinal_channel_handler,
            **conv_cfg
        )
        self.bn = SearchableBN(
            ctx, out_channels,
            force_use_ordinal_channel_handler=force_use_ordinal_channel_handler,
            **norm_cfg)

    def forward(self, inputs):
        out = self.bn(self.conv(inputs))
        return out

class SearchableFC(germ.SearchableBlock, nn.Linear):
    NAME = "fc"

    def __init__(
        self,
        ctx,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__(ctx)
        self.fi_choices = in_features
        self.fo_choices = out_features
        self.fi_handler = FeatureMaskHandler(ctx, self, "in_features", in_features)
        self.fo_handler = FeatureMaskHandler(ctx, self, "out_features", out_features)
        _modules = self._modules
        _parameters = self._parameters
        _buffers = self._buffers
        nn.Linear.__init__(
            self,
            in_features=self.fi_handler.max,
            out_features=self.fo_handler.max,
            bias=bias
        )
        self._modules.update(_modules)
        self._parameters.update(_parameters)
        self._buffers.update(_buffers)

    def rollout_context(self, rollout=None, detach=False):
        if rollout is None:
            return nullcontext()
        # in features and out features
        r_i_f = self._get_decision(self.fi_handler.choices, rollout)
        r_o_f = self._get_decision(self.fo_handler.choices, rollout)
        # apply
        ctx = self.fi_handler.apply(self, r_i_f, axis=1, detach=detach)
        ctx = self.fo_handler.apply(self, r_o_f, axis=0, ctx=ctx, detach=detach)
        return ctx
    def forward(self, inputs):
        with self.rollout_context(self.ctx.rollout):
            out = super().forward(inputs)
        return out
    def finalize_rollout(self, rollout):
        with self.rollout_context(self.ctx.rollout):
            fc = nn.Linear(self.in_features, self.out_features)
            fc.weight.data.copy_(self.weight.data)
            if self.bias is not None:
                fc.bias.data.copy_(self.bias.data)
        return fc


class SearchableSepConv(germ.SearchableBlock):
    NAME = "sep_conv"

    def __init__(
        self, ctx, in_channels, out_channels, kernel_size, stride=1, activation="relu"
    ):
        super().__init__(ctx)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        depth_wise = SearchableConvBNBlock(
            ctx, in_channels, in_channels, kernel_size, stride, groups=in_channels
        )
        point_linear = SearchableConvBNBlock(ctx, in_channels, out_channels, 1)

        self.conv = nn.Sequential(depth_wise, ops.get_op(activation)(), point_linear)

    def forward(self, inputs):
        return self.conv(inputs)


class SearchableMBV2Block(germ.SearchableBlock):
    NAME = "mobilenet_v2"

    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        exp_ratio,
        kernel_size,
        stride=1,
        activation="relu",
        **kwargs
    ):
        super().__init__(ctx)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.inner_channels = (exp_ratio * in_channels).apply(divisor_fn)
        self.stride = stride

        self.inv_bottleneck = SearchableConvBNBlock(
            ctx, in_channels, self.inner_channels, 1
        )
        self.depth_wise = SearchableConvBNBlock(
            ctx,
            self.inner_channels,
            self.inner_channels,
            kernel_size,
            stride,
            groups=self.inner_channels,
        )
        self.point_linear = SearchableConvBNBlock(
            ctx, self.inner_channels, out_channels, 1
        )

        self.act1 = ops.get_op(activation)()
        self.act2 = ops.get_op(activation)()

    def forward(self, inputs):
        out = self.inv_bottleneck(inputs)
        out = self.act1(out)
        out = self.depth_wise(out)
        out = self.act2(out)
        out = self.point_linear(out)
        if inputs.shape[-1] == out.shape[-1]  and inputs.shape[1] == out.shape[1]:
            #and self.in_channels == self.out_channels:
            out += inputs
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class SearchableShuffleV2Block(germ.SearchableBlock):
    NAME = "shufflenet_v2"

    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        **kwargs
    ):
        super().__init__(ctx)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        assert isinstance(stride, int)

        self.inner_channels = self.out_channels * 0.5

        if isinstance(self.inner_channels, float):
            self.inner_channels = int(self.inner_channels)

        bottleneck = SearchableConvBNBlock(
            ctx,
            in_channels if stride > 1 else self.inner_channels,
            self.inner_channels,
            1,
        )
        depth_wise = SearchableConvBNBlock(
            ctx,
            self.inner_channels,
            self.inner_channels,
            kernel_size,
            stride,
            groups=self.inner_channels,
        )
        point_linear = SearchableConvBNBlock(
            ctx, self.inner_channels, self.inner_channels, 1
        )

        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(
                SearchableConvBNBlock(
                    ctx,
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    groups=in_channels,
                ),
                SearchableConvBNBlock(ctx, in_channels, self.inner_channels, 1),
                ops.get_op(activation)(),
            )

        self.branch = nn.Sequential(
            bottleneck,
            ops.get_op(activation)(),
            depth_wise,
            point_linear,
            ops.get_op(activation)(),
        )

    def forward(self, inputs):
        if self.stride == 1:
            x1, x2 = inputs.chunk(2, dim=1)
            out = torch.cat([x1, self.branch(x2)], dim=1)
        else:
            out = torch.cat([self.shortcut(inputs), self.branch(inputs)], dim=1)
        return channel_shuffle(out, 2)


class SearchableTucker(germ.SearchableBlock):
    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        sqz_ratio_1,
        sqz_ratio_2,
        kernel_size,
        stride=1,
        activation="relu",
    ):
        super().__init__(ctx)
        self.in_channels = in_channels
        self.squeeze_channels = (sqz_ratio_1 * in_channels).apply(divisor_fn)
        self.expand_channels = (sqz_ratio_2 * out_channels).apply(divisor_fn)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

        bottleneck = SearchableConvBNBlock(ctx, in_channels, self.squeeze_channels, 1)
        regular = SearchableConvBNBlock(
            ctx, self.squeeze_channels, self.expand_channels, kernel_size, stride
        )
        point_linear = SearchableConvBNBlock(ctx, self.expand_channels, out_channels, 1)

        self.tucker = nn.Sequential(
            bottleneck,
            ops.get_op(activation)(),
            regular,
            ops.get_op(activation)(),
            point_linear,
        )

    def forward(self, inputs):
        out = self.tucker(inputs)
        if self.stride == 1 and inputs.shape[1] == out.shape[1]:
            #self.in_channels == self.out_channels:
            out += inputs
        return out


class SearchableFusedConv(germ.SearchableBlock):
    def __init__(
        self,
        ctx,
        in_channels,
        out_channels,
        exp_ratio,
        kernel_size,
        stride=1,
        activation="relu",
    ):
        super().__init__(ctx)
        self.in_channels = in_channels
        self.inner_channels = (exp_ratio * in_channels).apply(divisor_fn)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

        bottleneck = SearchableConvBNBlock(ctx, in_channels, self.inner_channels, kernel_size,
                stride=stride)
        point_linear = SearchableConvBNBlock(ctx, self.inner_channels, out_channels, 1)

        self.fused_conv = nn.Sequential(
            bottleneck, ops.get_op(activation)(), point_linear
        )

    def forward(self, inputs):
        out = self.fused_conv(inputs)
        if self.stride == 1 and out.shape[1] == inputs.shape[1]:
            #self.in_channels == self.out_channels:
            out += inputs
        return out


class RepConv(germ.SearchableBlock):
    def __init__(self, ctx, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__(ctx)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups

        assert isinstance(stride, int)

        self.conv = SearchableConvBNBlock(
            ctx, in_channels, out_channels, kernel_size, stride=stride, groups=groups
        )
        self.branch = SearchableConvBNBlock(
            ctx, in_channels, out_channels, 1, stride=stride
        )
        self.bn = SearchableBN(ctx, in_channels)

    def forward(self, inputs):
        out = self.conv(inputs)
        out += self.branch(inputs)
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + self.bn(inputs)
        return out

    def reparameter(self):
        """
        call this function after calling finalize_rollout
        """
        conv_w = self.conv.weight.data
        branch_w = self.branch.weight.data
        branch_w = F.pad(branch_w, (1, 1, 1, 1))
        weight = conv_w + branch_w
        if self.stride == 1 and self.in_channels == self.out_channels:
            w = torch.zeros_like(weight)
            w[:, :, 1, 1] = 1
            weight += w
        conv = nn.Conv2d(
            self.conv.in_channels,
            self.conv.out_channels,
            self.conv.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=False,
        )

        conv.weight.data.copy_(weight)
        return nn.Sequential(conv, self.bn)
