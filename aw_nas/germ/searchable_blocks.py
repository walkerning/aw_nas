#pylint: disable=arguments-differ
from collections import OrderedDict
from collections import abc as collection_abcs

from torch import nn

from aw_nas import ops, germ


class GermOpOnEdge(germ.SearchableBlock):
    """
    A node with operation on edge
    """
    NAME = "op_on_edge"

    def __init__(self, ctx, num_input_nodes,
                 # basic info
                 from_nodes,
                 op_list,
                 aggregation="sum",

                 # optional decisions
                 from_nodes_choices=None,
                 edgeops_choices_dict=None,
                 aggregation_op_choice=None,

                 allow_repeat_choice=False,
                 **op_kwargs):
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
            self.from_nodes_choices = germ.Choices(from_nodes, size=num_input_nodes,
                                                   replace=allow_repeat_choice)
        self.select = GermSelect(ctx, self.from_nodes_choices)

        # edge ops
        if edgeops_choices_dict is None:
            self.edge_ops = nn.ModuleDict({
                str(from_node): GermMixedOp(ctx, op_list, **op_kwargs)
                for from_node in from_nodes})
            self.edgeops_choices_dict = {
                from_node: op.op_choice for from_node, op in self.edge_ops.items()
            }
        else:
            self.edge_ops = nn.ModuleDict({
                from_node: GermMixedOp(ctx, op_list, op_choices, **op_kwargs)
                for from_node, op_choices in edgeops_choices_dict.items()})
            self.edgeops_choices_dict = edgeops_choices_dict

        # aggregation
        if isinstance(aggregation, (list, tuple)):
            aggregation_op = GermMixedOp(ctx, [ops.get_concat_op(op_name)
                                               for op_name in aggregation],
                                         op_choice=aggregation_op_choice)
            self.aggregation_op = aggregation_op
            self.aggregation_op_choice = self.aggregation_op.op_choice
            # TODO: elementwise searchable with concat, need to align the dimension using 1x1
        else:
            self.aggregation_op = ops.get_concat_op(aggregation)
            self.aggregation_op_choice = None

    def forward_rollout(self, rollout, inputs):
        from_nodes, features_list = self.select(inputs)
        features_list = [self.edge_ops[str(from_node)](features)
                         for from_node, features in zip(from_nodes, features_list)]
        return self.aggregation_op(features_list)

    def finalize_rollout(self, rollout):
        # in-place finalize
        super().finalize_rollout(rollout)

        # drop unused edges
        final_from_nodes = self.select.from_nodes
        self.edge_ops = nn.ModuleDict({
            str(from_node): self.edge_ops[str(from_node)] for from_node in final_from_nodes
        })
        return self

    def finalize_rollout_outplace(self, rollout):
        # Outplace finalize could be clearer
        select = self.select.finalize_rollout(rollout)
        edge_ops = nn.ModuleDict(
            {str(from_node): self.edge_ops[str(from_node)].finalize_rollout(rollout)
             for from_node in select.from_nodes})
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
        features_list = [self.edge_ops[str(from_node)](features)
                         for from_node, features in zip(from_nodes, features_list)]
        return self.aggregation_op(features_list)


class GermOpOnNode(germ.SearchableBlock):
    """
    A node with operation on node
    """
    NAME = "op_on_node"

    def __init__(self, ctx, num_input_nodes,
                 # basic info
                 from_nodes,
                 op_list,
                 aggregation="sum",

                 # optional decisions
                 from_nodes_choices=None,
                 op_choice=None,
                 aggregation_op_choice=None,

                 allow_repeat_choice=False,
                 **op_kwargs):
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
            self.from_nodes_choices = germ.Choices(from_nodes, size=num_input_nodes,
                                                   replace=allow_repeat_choice)
        self.select = GermSelect(ctx, self.from_nodes_choices)

        # aggregation
        if isinstance(aggregation, (list, tuple)):
            self.aggregation_op = GermMixedOp(
                ctx, [ops.get_concat_op(op_name)
                      for op_name in aggregation],
                op_choice=aggregation_op_choice)
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
        if isinstance(from_nodes, germ.Choices):
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
            # initalize new choice
            self.op_choice = germ.Choices(choices=list(range(len(self.op_list))), size=1)
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

    def forward_rollout(self, rollout, inputs):
        op_index = self._get_decision(self.op_choice, rollout)
        return self.p_ops[op_index](inputs)

    def finalize_rollout_outplace(self, rollout):
        return self.finalize_rollout(rollout)

    def finalize_rollout(self, rollout):
        op_index = self._get_decision(self.op_choice, rollout)
        return self.p_ops[op_index]
