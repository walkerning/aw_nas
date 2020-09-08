"""
Search space for robustness NAS
"""
# pylint: disable=missing-docstring

import os
import collections
from collections import namedtuple
import random

import numpy as np
import torch.nn as nn

from aw_nas.ops import SepConv, Identity, FactorizedReduce, register_primitive
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.utils.exception import expect, ConfigException


class DenseRobRollout(BaseRollout):
    NAME = "dense_rob"
    supported_components = [("trainer", "simple"), ("evaluator", "mepa")]

    def __init__(self, arch, search_space, candidate_net=None):
        super(DenseRobRollout, self).__init__()

        self.arch = arch
        self.search_space = search_space
        self.candidate_net = candidate_net
        self.perf = collections.OrderedDict()
        self._genotype = None  # calc when need

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

    def plot_arch(self, filename, label="", edge_labels=None, plot_format="pdf"):
        fnames = []
        for i_cell, cell_arch in enumerate(self.arch):
            fname = self.search_space.plot_cell(
                cell_arch,
                os.path.join(filename, "{}".format(i_cell)),
                label="{}cell {}".format(
                    "{} - ".format(label) if label else "", i_cell
                ),
                edge_labels=edge_labels[i_cell] if edge_labels else None,
                plot_format=plot_format,
            )
            fnames.append(("cell{}".format(i_cell), fname))
        return fnames


class DenseRobSearchSpace(SearchSpace):
    """
    Macro-arch: how many cell groups, how to arrange them, how cells connect with each other -> cnn
    Micro-arch: -> nb201
    """

    NAME = "dense_rob"

    def __init__(
        self,
        ## macro arch
        num_cell_groups=2,
        num_init_nodes=1,
        cell_layout=None,
        reduce_cell_groups=None,
        ## micro arch
        num_steps=4,
        # concat for output
        concat_op="concat",
        concat_nodes=None,
        loose_end=False,
        primitives=("none", "skip_connect", "sep_conv_3x3", "ResSepConv"),
    ):
        super(DenseRobSearchSpace, self).__init__()

        ## micro-arch
        self.primitives = primitives
        self.num_op_choices = len(primitives)
        self.num_steps = num_steps
        self.num_init_nodes = num_init_nodes
        self._num_nodes = self.num_steps + self.num_init_nodes
        _num_input_conns = self.num_init_nodes * (self.num_init_nodes - 1) // 2
        self.idx = np.tril_indices(self._num_nodes, k=-1)
        self.idx = (self.idx[0][_num_input_conns:], self.idx[1][_num_input_conns:])
        self.num_possible_edges = np.sum(
            self.num_init_nodes + np.arange(self.num_steps)
        )
        assert len(self.idx[0]) == self.num_possible_edges

        self.concat_op = concat_op
        self.concat_nodes = concat_nodes
        self.loose_end = loose_end
        if loose_end:
            expect(
                self.concat_nodes is None,
                "When `loose_end` is given, `concat_nodes` will be automatically determined, "
                "should not be explicitly specified.",
                ConfigException,
            )

        ## overall structure/meta-arch: how to arrange cells
        # number of cell groups, different cell groups has different structure
        self.num_cell_groups = num_cell_groups

        # cell layout
        expect(
            cell_layout is not None,
            "`cell_layout` need to be explicitly specified",
            ConfigException,
        )
        expect(
            np.max(cell_layout) == self.num_cell_groups - 1,
            "Max of elements of `cell_layout` should equal `num_cell_groups-1`",
        )
        self.cell_layout = cell_layout
        self.num_layers = len(cell_layout)

        self.reduce_cell_groups = reduce_cell_groups
        expect(
            reduce_cell_groups is not None,
            "`reduce_cell_groups` need to be explicitly specified",
            ConfigException,
        )

        self.cell_group_names = [
            "{}_{}".format("reduce" if i in self.reduce_cell_groups else "normal", i)
            for i in range(self.num_cell_groups)
        ]

        self.genotype_type_name = "DenseRobGenotype"
        self.genotype_type = namedtuple(self.genotype_type_name, self.cell_group_names)

        # init nodes(meta arch: connect from the last `num_init_nodes` cell's output)
        self.num_init_nodes = num_init_nodes

    def _random_sample_cell_arch(self):
        arch = np.zeros((self._num_nodes, self._num_nodes))
        arch[self.idx] = np.random.randint(
            low=0, high=self.num_op_choices, size=self.num_possible_edges
        )
        return arch

    def _random_sample_arch(self):
        return [self._random_sample_cell_arch() for _ in range(self.num_cell_groups)]

    def __getstate__(self):
        state = super(DenseRobSearchSpace, self).__getstate__().copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        super(DenseRobSearchSpace, self).__setstate__(state)
        self.genotype_type = namedtuple(self.genotype_type_name, self.cell_group_names)

    def plot_cell(
        self, matrix, filename, label="", edge_labels=None, plot_format="pdf"
    ):
        from graphviz import Digraph

        graph = Digraph(
            format=plot_format,
            body=['label="{l}"'.format(l=label), "labelloc=top", "labeljust=left"],
            edge_attr=dict(fontsize="20", fontname="times"),
            node_attr=dict(
                style="filled",
                shape="rect",
                align="center",
                fontsize="20",
                height="0.5",
                width="0.5",
                penwidth="2",
                fontname="times",
            ),
            engine="dot",
        )
        graph.body.extend(["rankdir=LR"])
        final_output_node = self._num_nodes
        init_node = self.num_init_nodes
        [graph.node(str(i), fillcolor="darkseagreen2") for i in range(0, init_node)]
        [
            graph.node(str(i), fillcolor="lightblue")
            for i in range(init_node, final_output_node)
        ]
        graph.node(str(final_output_node), fillcolor="palegoldenrod")

        for to_, from_ in zip(*self.idx):
            op_name = self.primitives[int(matrix[to_, from_])]
            if op_name == "none":
                continue
            graph.edge(str(from_), str(to_), label=op_name, fillcolor="gray")

        # final concat edges
        for node in range(init_node, final_output_node):
            graph.edge(str(node), str(final_output_node), fillolor="gray")

        graph.render(filename, view=False)
        return filename + ".{}".format(plot_format)

    # ---- APIs ----
    def random_sample(self):
        """Random sample an architecture rollout from search space"""
        return DenseRobRollout(self._random_sample_arch(), search_space=self)

    def genotype(self, arch):
        """Convert arch (controller representation) to genotype (semantic representation)"""
        cell_strs = []
        for cell_arch in arch:
            node_strs = ["init_node~{}".format(self.num_init_nodes)]
            for i_node in range(self.num_init_nodes, self._num_nodes):
                node_strs.append(
                    "|"
                    + "|".join(
                        [
                            "{}~{}".format(
                                self.primitives[int(cell_arch[i_node, i_input])],
                                i_input,
                            )
                            for i_input in range(0, i_node)
                        ]
                    )
                    + "|"
                )
            cell_str = "+".join(node_strs)
            cell_strs.append(cell_str)
        return self.genotype_type(**dict(zip(self.cell_group_names, cell_strs)))

    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""
        cell_strs = list(genotype._asdict().values())
        arch = []
        for cell_str in cell_strs:
            node_strs = cell_str.strip().split("+")
            _geno_num_i_nodes = int(node_strs[0].split("~")[1])
            expect(
                _geno_num_i_nodes == self.num_init_nodes,
                (
                    "Search space configuration (`num_init_nodes={}` "
                    "differs from the genotype specification {})"
                ).format(self.num_init_nodes, _geno_num_i_nodes),
            )
            all_conn_ops = [
                conn_str.split("~")[0]
                for node_str in node_strs[1:]
                for conn_str in node_str[1:-1].split("|")
            ]
            all_conn_op_inds = [
                self.primitives.index(conn_op) for conn_op in all_conn_ops
            ]
            cell_arch = np.zeros((self._num_nodes, self._num_nodes))
            cell_arch[self.idx] = all_conn_op_inds
            arch.append(cell_arch)
        return DenseRobRollout(arch, search_space=self)

    def genotype_to_arch(self, genotype):
        cell_strs = list(genotype._asdict().values())
        arch = []
        for cell_str in cell_strs:
            node_strs = cell_str.strip().split("+")
            _geno_num_i_nodes = int(node_strs[0].split("~")[1])
            expect(
                _geno_num_i_nodes == self.num_init_nodes,
                (
                    "Search space configuration (`num_init_nodes={}` "
                    "differs from the genotype specification {})"
                ).format(self.num_init_nodes, _geno_num_i_nodes),
            )
            all_conn_ops = [
                conn_str.split("~")[0]
                for node_str in node_strs[1:]
                for conn_str in node_str[1:-1].split("|")
            ]
            all_conn_op_inds = [
                self.primitives.index(conn_op) for conn_op in all_conn_ops
            ]
            cell_arch = np.zeros((self._num_nodes, self._num_nodes))
            cell_arch[self.idx] = all_conn_op_inds
            arch.append(cell_arch)
        return arch

    def plot_arch(self, genotypes, filename, label="", edge_labels=None, **kwargs):
        return self.rollout_from_genotype(genotypes).plot_arch(
            filename, label=label, edge_labels=edge_labels, **kwargs
        )

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    def mutate(self, parent):
        child = parent.arch.copy()
        change_cell_idx = random.randint(0, self.num_cell_groups - 1)
        change_node_idx = random.randint(self.num_init_nodes, self._num_nodes - 1)
        change_node_from = random.randint(0, change_node_idx - 1)
        old = child[change_cell_idx][change_node_idx][change_node_from]
        offset = random.randint(1, self.num_op_choices - 1)
        child[change_cell_idx][change_node_idx][change_node_from] = (
            old + offset
        ) % self.num_op_choices
        return DenseRobRollout(child, self)

    @classmethod
    def supported_rollout_types(cls):
        return ["dense_rob"]


class ResSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ResSepConv, self).__init__()
        self.conv = SepConv(C_in, C_out, kernel_size, stride, padding)
        self.res = Identity() if stride == 1 else FactorizedReduce(C_in, C_out, stride)

    def forward(self, x):
        return self.conv(x) + self.res(x)


register_primitive(
    "ResSepConv", lambda C, C_out, stride, affine: ResSepConv(C, C_out, 3, stride, 1)
)
