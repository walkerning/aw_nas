"""
NASBench-101 search space, rollout, controller, evaluator.
During the development,
referred https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
"""

import abc
import copy
import os
import re
import random
import collections
import itertools
import yaml
from typing import List

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import nasbench
from nasbench import api
from nasbench.lib import graph_util, config

from aw_nas import utils
from aw_nas.ops import get_op, Identity
from aw_nas.utils.exception import expect
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.controller.base import BaseController
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.rollout.compare import CompareRollout
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.utils import DenseGraphConvolution, DenseGraphFlow
from aw_nas.weights_manager.shared import SharedCell, SharedOp
from aw_nas.weights_manager.base import CandidateNet, BaseWeightsManager


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT_NODE = 6

VERTICES = 7
MAX_EDGES = 9
_nasbench_cfg = config.build_config()


def parent_combinations_old(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the current index,
        # because of the upper triangular adjacency matrix and because the index is also a
        # topological ordering in our case.
        return itertools.combinations(np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
                                      n_parents)  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]


def parent_combinations(node, num_parents):
    if node == 1 and num_parents == 1:
        return [(0,)]
    else:
        return list(itertools.combinations(list(range(int(node))), num_parents))


def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix,
                  5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def _literal_np_array(arr):
    if arr is None:
        return None
    return "np.array({})".format(np.array2string(arr, separator=",").replace("\n", " "))


class _ModelSpec(api.ModelSpec):
    def __repr__(self):
        return "_ModelSpec({}, {}; pruned_matrix={}, pruned_ops={})".format(
            _literal_np_array(self.original_matrix),
            self.original_ops,
            _literal_np_array(self.matrix),
            self.ops,
        )

    def hash_spec(self, *args, **kwargs):
        return super(_ModelSpec, self).hash_spec(_nasbench_cfg["available_ops"])


class NasBench101SearchSpace(SearchSpace):
    NAME = "nasbench-101"

    def __init__(
        self,
        multi_fidelity=False,
        load_nasbench=True,
        compare_reduced=True,
        compare_use_hash=False,
        validate_spec=True,
    ):
        super(NasBench101SearchSpace, self).__init__()

        self.ops_choices = ["conv1x1-bn-relu",
                            "conv3x3-bn-relu", "maxpool3x3", "none"]

        awnas_ops = [
            "conv_bn_relu_1x1",
            "conv_bn_relu_3x3",
            "max_pool_3x3",
            "none",
        ]

        self.op_mapping = {k: v for k, v in zip(self.ops_choices, awnas_ops)}

        self.ops_choice_to_idx = {
            choice: i for i, choice in enumerate(self.ops_choices)
        }

        # operations: "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"
        self.multi_fidelity = multi_fidelity
        self.load_nasbench = load_nasbench
        self.compare_reduced = compare_reduced
        self.compare_use_hash = compare_use_hash

        self.num_vertices = VERTICES
        self.max_edges = MAX_EDGES
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * \
            (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices)  # 3 + 1 (none)
        self.num_ops = self.num_vertices - 2  # 5
        self.idx = np.triu_indices(self.num_vertices, k=1)
        self.validate_spec = validate_spec

        if self.load_nasbench:
            self._init_nasbench()

    def __getstate__(self):
        state = super(NasBench101SearchSpace, self).__getstate__().copy()
        del state["nasbench"]
        return state

    def __setstate__(self, state):
        super(NasBench101SearchSpace, self).__setstate__(state)
        if self.load_nasbench:
            # slow, comment this if do not need to load nasbench API when pickle load from disk
            self._init_nasbench()

    def pad_archs(self, archs):
        return [self._pad_arch(arch) for arch in archs]

    def _pad_arch(self, arch):
        # padding for batchify training
        adj, ops = arch
        # all normalize the the reduced one
        spec = self.construct_modelspec(edges=None, matrix=adj, ops=ops)
        adj, ops = spec.matrix, self.op_to_idx(spec.ops)
        num_v = adj.shape[0]
        if num_v < VERTICES:
            padded_adj = np.concatenate(
                (adj[:-1], np.zeros((VERTICES - num_v, num_v), dtype=np.int8), adj[-1:])
            )
            padded_adj = np.concatenate(
                (
                    padded_adj[:, :-1],
                    np.zeros((VERTICES, VERTICES - num_v)),
                    padded_adj[:, -1:],
                ),
                axis=1,
            )
            padded_ops = ops + [3] * (7 - num_v)
            adj, ops = padded_adj, padded_ops
        return (adj, ops)

    def _random_sample_ori(self):
        while 1:
            matrix = np.random.choice(
                [0, 1], size=(self.num_vertices, self.num_vertices)
            )
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(
                self.ops_choices[:-1], size=(self.num_vertices)
            ).tolist()
            ops[0] = "input"
            ops[-1] = "output"
            spec = _ModelSpec(matrix=matrix, ops=ops)
            if self.validate_spec and not self.nasbench.is_valid(spec):
                continue
            return NasBench101Rollout(
                spec.original_matrix,
                ops=self.op_to_idx(spec.original_ops),
                search_space=self,
            )

    def _random_sample_me(self):
        while 1:
            splits = np.array(
                sorted(
                    [0]
                    + list(
                        np.random.randint(
                            0, self.max_edges + 1, size=self.num_possible_edges - 1
                        )
                    )
                    + [self.max_edges]
                )
            )
            edges = np.minimum(splits[1:] - splits[:-1], 1)
            matrix = self.edges_to_matrix(edges)
            ops = np.random.randint(0, self.num_op_choices, size=self.num_ops)
            rollout = NasBench101Rollout(matrix, ops, search_space=self)
            try:
                self.nasbench._check_spec(rollout.genotype)
            except api.OutOfDomainError:
                # ignore out-of-domain archs (disconnected)
                continue
            else:
                return rollout

    # optional API
    def genotype_from_str(self, genotype_str):
        return eval(genotype_str)
        return eval(re.search("(_ModelSpec\(.+);", genotype_str).group(1) + ")")

    # ---- APIs ----
    def random_sample(self):
        m, ops = self.sample(True)
        if len(ops) < len(m) - 2:
            ops.append("none")
        return NasBench101Rollout(m, [self.ops_choices.index(op) for op in ops], search_space=self)
        return self._random_sample_ori()

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        matrix, ops = arch
        return self.construct_modelspec(edges=None, matrix=matrix, ops=ops)

    def rollout_from_genotype(self, genotype):
        return NasBench101Rollout(
            genotype.original_matrix,
            ops=self.op_to_idx(genotype.original_ops),
            search_space=self,
        )

    def plot_arch(self, genotypes, filename, label, plot_format="pdf", **kwargs):
        graph = genotypes.visualize()
        graph.format = "pdf"
        graph.render(filename, view=False)
        return filename + ".{}".format(plot_format)

    def distance(self, arch1, arch2):
        pass

    # ---- helpers ----
    def _init_nasbench(self):
        # the arch -> performances dataset
        self.base_dir = os.path.join(
            utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-101"
        )
        if self.multi_fidelity:
            self.nasbench = api.NASBench(
                os.path.join(self.base_dir, "nasbench_full.tfrecord")
            )
        else:
            self.nasbench = api.NASBench(
                os.path.join(self.base_dir, "nasbench_only108.tfrecord")
            )

    def edges_to_matrix(self, edges):
        matrix = np.zeros(
            [self.num_vertices, self.num_vertices], dtype=np.int8)
        matrix[self.idx] = edges
        return matrix

    def op_to_idx(self, ops):
        return [
            self.ops_choice_to_idx[op] for op in ops if op not in {"input", "output"}
        ]

    def matrix_to_edges(self, matrix):
        return matrix[self.idx]

    def matrix_to_connection(self, matrix):
        edges = matrix[self.idx].astype(np.bool)
        node_connections = {}
        concat_nodes = []
        for from_, to_ in zip(self.idx[0][edges], self.idx[1][edges]):
            # index of nodes starts with 1 rather than 0
            if to_ < len(matrix) - 1:
                node_connections.setdefault(to_, []).append(from_)
            else:
                if from_ >= len(matrix) - 2:
                    continue
                concat_nodes.append(from_)
        return node_connections, concat_nodes

    def construct_modelspec(self, edges, matrix, ops):
        if matrix is None:
            assert edges is not None
            matrix = self.edges_to_matrix(edges)

        # expect(graph_util.num_edges(matrix) <= self.max_edges,
        #        "number of edges could not exceed {}".format(self.max_edges))

        labeling = [self.ops_choices[op_ind] for op_ind in ops]
        labeling = ["input"] + list(labeling) + ["output"]
        model_spec = _ModelSpec(matrix, labeling)
        return model_spec

    def random_sample_arch(self):
        # not uniform, and could be illegal,
        #   if there is not edge from the INPUT or no edge to the OUTPUT,
        # Just check and reject for now
        return self.random_sample().arch

    def batch_rollouts(self, batch_size, shuffle=True, max_num=None):
        len_ = ori_len_ = len(self.nasbench.fixed_statistics)
        if max_num is not None:
            len_ = min(max_num, len_)
        list_ = list(self.nasbench.fixed_statistics.values())
        indexes = np.arange(ori_len_)
        np.random.shuffle(indexes)
        ind = 0
        while ind < len_:
            end_ind = min(len_, ind + batch_size)
            yield [
                NasBench101Rollout(
                    list_[r_ind]["module_adjacency"],
                    self.op_to_idx(list_[r_ind]["module_operations"]),
                    search_space=self,
                )
                for r_ind in indexes[ind:end_ind]
            ]
            ind = end_ind

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]


class NasBench101OneShotSearchSpace(NasBench101SearchSpace):
    # NAME = "nasbench-101-1shot"

    def __init__(
        self,
        multi_fidelity=False,
        load_nasbench=True,
        compare_reduced=True,
        compare_use_hash=False,
        validate_spec=True,
        num_cell_groups=2,
        num_init_nodes=1,
        cell_layout=None,
        reduce_cell_groups=(1,),
        num_layers=8,
    ):
        super(NasBench101OneShotSearchSpace, self).__init__(
            multi_fidelity,
            load_nasbench,
            compare_reduced,
            compare_use_hash,
            validate_spec,
        )

        self.num_init_nodes = num_init_nodes
        self.num_cell_groups = num_cell_groups
        self.reduce_cell_groups = reduce_cell_groups
        self.num_layers = num_layers

        if cell_layout is not None:
            expect(
                len(cell_layout) == self.num_layers,
                "Length of `cell_layout` should equal `num_layers`",
            )
            expect(
                np.max(cell_layout) == self.num_cell_groups - 1,
                "Max of elements of `cell_layout` should equal `num_cell_groups-1`",
            )
            self.cell_layout = cell_layout
        elif self.num_cell_groups == 2:
            # by default: cell 0: normal cel, cell 1: reduce cell
            self.cell_layout = [0] * self.num_layers
            self.cell_layout[self.num_layers // 3] = 1
            self.cell_layout[(2 * self.num_layers) // 3] = 1
        else:
            raise ValueError

        self.loose_end = False
        self.num_steps = 4
        self.concat_op = "concat"
        self.concat_nodes = None
        self.cellwise_primitives = False
        self.shared_primitives = self.ops_choices

        self.num_parents = None

        if self.load_nasbench:
            self._init_nasbench()

    def _is_valid(self, matrix):
        assert self.num_parents is not None, \
            "Do no use nasbench-101-1shot directly, please use nasbench-101-1shot-1, "\
            "nasbench-101-1shot-2 or nasbench-101-1shot-3 search space instead."
        num_node = list(matrix.sum(0))
        if len(num_node) == VERTICES - 1:
            num_node.insert(-2, 0)
        return all([p == k for p, k in zip(self.num_parents, num_node)])

    @abc.abstractmethod
    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        """Based on given connectivity pattern create the corresponding adjacency matrix."""
        pass

    def sample(self, with_loose_ends, upscale=True):
        if with_loose_ends:
            adjacency_matrix_sample = self._sample_adjacency_matrix_with_loose_ends()
        else:
            adjacency_matrix_sample = self._sample_adjacency_matrix_without_loose_ends(
                adjacency_matrix=np.zeros(
                    [self.num_intermediate_nodes + 2, self.num_intermediate_nodes + 2]),
                node=self.num_intermediate_nodes + 1)
            assert self._check_validity_of_adjacency_matrix(
                adjacency_matrix_sample), 'Incorrect graph'

        if upscale and self.NAME[-1] in ["1", "2"]:
            adjacency_matrix_sample = upscale_to_nasbench_format(
                adjacency_matrix_sample)
        return adjacency_matrix_sample, random.choices(self.ops_choices[:-1], k=self.num_intermediate_nodes)

    def _sample_adjacency_matrix_with_loose_ends(self):
        parents_per_node = [random.sample(list(itertools.combinations(list(range(int(node))), num_parents)), 1) for
                            node, num_parents in self.num_parents_per_node.items()][2:]
        parents = {
            '0': [],
            '1': [0]
        }
        for node, node_parent in enumerate(parents_per_node, 2):
            parents[str(node)] = node_parent
        adjacency_matrix = self._create_adjacency_matrix_with_loose_ends(
            parents)
        return adjacency_matrix

    def _sample_adjacency_matrix_without_loose_ends(self, adjacency_matrix, node):
        req_num_parents = self.num_parents_per_node[str(node)]
        current_num_parents = np.sum(adjacency_matrix[:, node], dtype=np.int)
        num_parents_left = req_num_parents - current_num_parents
        sampled_parents = \
            random.sample(list(parent_combinations_old(
                adjacency_matrix, node, n_parents=num_parents_left)), 1)[0]
        for parent in sampled_parents:
            adjacency_matrix[parent, node] = 1
            adjacency_matrix = self._sample_adjacency_matrix_without_loose_ends(
                adjacency_matrix, parent)
        return adjacency_matrix

    @abc.abstractmethod
    def generate_adjacency_matrix_without_loose_ends(self, **kwargs):
        """Returns every adjacency matrix in the search space without loose ends."""
        pass

    def convert_config_to_nasbench_format(self, config):
        parents = {node: config["choice_block_{}_parents".format(node)] for node in
                   list(self.num_parents_per_node.keys())[1:]}
        parents['0'] = []
        adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
            parents)
        ops = [config["choice_block_{}_op".format(node)] for node in list(
            self.num_parents_per_node.keys())[1:-1]]
        return adjacency_matrix, ops

    def generate_search_space_without_loose_ends(self):
        # Create all possible connectivity patterns
        for iter, adjacency_matrix in enumerate(self.generate_adjacency_matrix_without_loose_ends()):
            print(iter)
            # Print graph
            # Evaluate every possible combination of node ops.
            n_repeats = int(np.sum(np.sum(adjacency_matrix, axis=1)[1:-1] > 0))
            for combination in itertools.product([CONV1X1, CONV3X3, MAXPOOL3X3], repeat=n_repeats):
                # Create node labels
                # Add some op as node 6 which isn't used, here conv1x1
                ops = [INPUT]
                combination = list(combination)
                for i in range(5):
                    if np.sum(adjacency_matrix, axis=1)[i + 1] > 0:
                        ops.append(combination.pop())
                    else:
                        ops.append(CONV1X1)
                assert len(combination) == 0, 'Something is wrong'
                ops.append(OUTPUT)

                # Create nested list from numpy matrix
                nasbench_adjacency_matrix = adjacency_matrix.astype(
                    np.int).tolist()

                # Assemble the model spec
                model_spec = api.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=nasbench_adjacency_matrix,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=ops)

                yield adjacency_matrix, ops, model_spec

    def _generate_adjacency_matrix(self, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            yield adjacency_matrix
        else:
            req_num_parents = self.num_parents_per_node[str(node)]
            current_num_parents = np.sum(
                adjacency_matrix[:, node], dtype=np.int)
            num_parents_left = req_num_parents - current_num_parents

            for parents in parent_combinations_old(adjacency_matrix, node, n_parents=num_parents_left):
                # Make copy of adjacency matrix so that when it returns to this stack
                # it can continue with the unmodified adjacency matrix
                adjacency_matrix_copy = copy.copy(adjacency_matrix)
                for parent in parents:
                    adjacency_matrix_copy[parent, node] = 1
                    for graph in self._generate_adjacency_matrix(adjacency_matrix=adjacency_matrix_copy, node=parent):
                        yield graph

    def _create_adjacency_matrix(self, parents, adjacency_matrix, node):
        if self._check_validity_of_adjacency_matrix(adjacency_matrix):
            # If graph from search space then yield.
            return adjacency_matrix
        else:
            for parent in parents[str(node)]:
                adjacency_matrix[parent, node] = 1
                if parent != 0:
                    adjacency_matrix = self._create_adjacency_matrix(parents=parents, adjacency_matrix=adjacency_matrix,
                                                                     node=parent)
            return adjacency_matrix

    def _create_adjacency_matrix_with_loose_ends(self, parents):
        # Create the adjacency_matrix on a per node basis
        adjacency_matrix = np.zeros([len(parents), len(parents)])
        for node, node_parents in parents.items():
            for parent in node_parents:
                adjacency_matrix[parent, int(node)] = 1
        return adjacency_matrix

    def _check_validity_of_adjacency_matrix(self, adjacency_matrix):
        """
        Checks whether a graph is a valid graph in the search space.
        1. Checks that the graph is non empty
        2. Checks that every node has the correct number of inputs
        3. Checks that if a node has outgoing edges then it should also have incoming edges
        4. Checks that input node is connected
        5. Checks that the graph has no more than 9 edges
        :param adjacency_matrix:
        :return:
        """
        # Check that the graph contains nodes
        num_intermediate_nodes = sum(
            np.array(np.sum(adjacency_matrix, axis=1) > 0, dtype=int)[1:-1])
        if num_intermediate_nodes == 0:
            return False

        # Check that every node has exactly the right number of inputs
        col_sums = np.sum(adjacency_matrix[:, :], axis=0)
        for col_idx, col_sum in enumerate(col_sums):
            # important FIX!
            if col_idx > 0:
                if col_sum != self.num_parents_per_node[str(col_idx)]:
                    return False

        # Check that if a node has outputs then it should also have incoming edges (apart from zero)
        col_sums = np.sum(np.sum(adjacency_matrix, axis=0) > 0)
        row_sums = np.sum(np.sum(adjacency_matrix, axis=1) > 0)
        if col_sums != row_sums:
            return False

        # Check that the input node is always connected. Otherwise the graph is disconnected.
        row_sum = np.sum(adjacency_matrix, axis=1)
        if row_sum[0] == 0:
            return False

        # Check that the graph returned has no more than 9 edges.
        num_edges = np.sum(adjacency_matrix.flatten())
        if num_edges > 9:
            return False

        return True

    def get_layer_num_steps(self, layer_index):
        return self.get_num_steps(self.cell_layout[layer_index])

    def get_num_steps(self, cell_index):
        return (
            self.num_steps
            if isinstance(self.num_steps, int)
            else self.num_steps[cell_index]
        )

    def _random_sample_ori(self):
        while 1:
            matrix = np.random.choice(
                [0, 1], size=(self.num_vertices, self.num_vertices)
            )
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(
                self.ops_choices[:-1], size=(self.num_vertices)
            ).tolist()
            ops[0] = "input"
            ops[-1] = "output"
            spec = _ModelSpec(matrix=matrix, ops=ops)
            if (
                self.validate_spec
                and not self.nasbench.is_valid(spec)
                and not self._is_valid(matrix)
            ):
                continue
            return NasBench101Rollout(
                spec.original_matrix,
                ops=self.op_to_idx(spec.original_ops),
                search_space=self,
            )


class NasBench101OneShot1SearchSpace(NasBench101OneShotSearchSpace):
    NAME = "nasbench-101-1shot-1"

    def __init__(
        self,
        multi_fidelity=False,
        load_nasbench=True,
        compare_reduced=True,
        compare_use_hash=False,
        validate_spec=True,
        num_cell_groups=2,
        num_init_nodes=1,
        cell_layout=None,
        reduce_cell_groups=(1,),
        num_layers=8,
    ):
        super(NasBench101OneShot1SearchSpace, self).__init__(
            multi_fidelity,
            load_nasbench,
            compare_reduced,
            compare_use_hash,
            validate_spec,
            num_cell_groups,
            num_init_nodes,
            cell_layout,
            reduce_cell_groups,
            num_layers,
        )

        self.num_parents = [0, 1, 2, 2, 2, 0, 2]
        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 2,
            '4': 2,
            '5': 2
        }
        self.num_intermediate_nodes = 4

        assert sum(self.num_parents) == 9, "The num of edges must equal to 9."

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        adjacency_matrix = self._create_adjacency_matrix(parents, adjacency_matrix=np.zeros([6, 6]),
                                                         node=OUTPUT_NODE - 1)
        # Create nasbench compatible adjacency matrix
        return upscale_to_nasbench_format(adjacency_matrix)

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return upscale_to_nasbench_format(self._create_adjacency_matrix_with_loose_ends(parents))

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([6, 6]),
                                                                node=OUTPUT_NODE - 1):
            yield upscale_to_nasbench_format(adjacency_matrix)

    def generate_with_loose_ends(self):
        for _, parent_node_3, parent_node_4, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': [0, 1],
                '3': parent_node_3,
                '4': parent_node_4,
                '5': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
                parents)
            yield adjacency_matrix


class NasBench101OneShot2SearchSpace(NasBench101OneShotSearchSpace):
    NAME = "nasbench-101-1shot-2"

    def __init__(
        self,
        multi_fidelity=False,
        load_nasbench=True,
        compare_reduced=True,
        compare_use_hash=False,
        validate_spec=True,
        num_cell_groups=2,
        num_init_nodes=1,
        cell_layout=None,
        reduce_cell_groups=(1,),
        num_layers=8,
    ):
        super(NasBench101OneShot2SearchSpace, self).__init__(
            multi_fidelity,
            load_nasbench,
            compare_reduced,
            compare_use_hash,
            validate_spec,
            num_cell_groups,
            num_init_nodes,
            cell_layout,
            reduce_cell_groups,
            num_layers,
        )

        self.num_parents = [0, 1, 1, 2, 2, 0, 3]
        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 1,
            '3': 2,
            '4': 2,
            '5': 3
        }
        self.num_intermediate_nodes = 4

        assert sum(self.num_parents) == 9, "The num of edges must equal to 9."

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        adjacency_matrix = self._create_adjacency_matrix(parents, adjacency_matrix=np.zeros([6, 6]),
                                                         node=OUTPUT_NODE - 1)
        # Create nasbench compatible adjacency matrix
        return upscale_to_nasbench_format(adjacency_matrix)

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return upscale_to_nasbench_format(self._create_adjacency_matrix_with_loose_ends(parents))

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([6, 6]),
                                                                node=OUTPUT_NODE - 1):
            yield upscale_to_nasbench_format(adjacency_matrix)

    def generate_with_loose_ends(self):
        for parent_node_2, parent_node_3, parent_node_4, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': parent_node_2,
                '3': parent_node_3,
                '4': parent_node_4,
                '5': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
                parents)
            yield adjacency_matrix


class NasBench101OneShot3SearchSpace(NasBench101OneShotSearchSpace):
    NAME = "nasbench-101-1shot-3"

    def __init__(
        self,
        multi_fidelity=False,
        load_nasbench=True,
        compare_reduced=True,
        compare_use_hash=False,
        validate_spec=True,
        num_cell_groups=2,
        num_init_nodes=1,
        cell_layout=None,
        reduce_cell_groups=(1,),
        num_layers=8,
    ):
        super(NasBench101OneShot3SearchSpace, self).__init__(
            multi_fidelity,
            load_nasbench,
            compare_reduced,
            compare_use_hash,
            validate_spec,
            num_cell_groups,
            num_init_nodes,
            cell_layout,
            reduce_cell_groups,
            num_layers,
        )

        self.num_parents = [0, 1, 1, 1, 2, 2, 2]

        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 1,
            '3': 1,
            '4': 2,
            '5': 2,
            '6': 2
        }

        self.num_intermediate_nodes = 5

        assert sum(self.num_parents) == 9, "The num of edges must equal to 9."

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        # Create nasbench compatible adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(
            parents, adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE)
        return adjacency_matrix

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return self._create_adjacency_matrix_with_loose_ends(parents)

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE):
            yield adjacency_matrix

    def generate_with_loose_ends(self):
        for parent_node_2, parent_node_3, parent_node_4, parent_node_5, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': parent_node_2,
                '3': parent_node_3,
                '4': parent_node_4,
                '5': parent_node_5,
                '6': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
                parents)
            yield adjacency_matrix


class NasBench101Rollout(BaseRollout):
    NAME = "nasbench-101"
    supported_components = [("evaluator", "mepa"), ("trainer", "simple")]

    def __init__(self, matrix, ops, search_space):
        self._arch = (matrix, ops)
        self.search_space = search_space
        self.perf = collections.OrderedDict()
        self._genotype = None

    @property
    def arch(self):
        return self._arch

    @property
    def pruned_arch(self):
        matrix, ops = self._arch
        index = [i for i, op in enumerate(ops) if op != 3]
        ops = [ops[i] for i in index]
        index = [0] + [i + 1 for i in index] + [6]
        matrix = matrix[index][:, index]
        return matrix, ops

    def set_candidate_net(self, c_net):
        raise Exception("Should not be called")

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(
            self.genotype, filename, label=label, edge_labels=edge_labels
        )

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.pruned_arch)
        return self._genotype

    def __repr__(self):
        return "NasBench101Rollout(matrix={arch}, perf={perf})".format(
            arch=self.arch, perf=self.perf
        )

    def __eq__(self, other):
        if self.search_space.compare_reduced:
            if self.search_space.compare_use_hash:
                # compare using hash, isomorphic archs would be equal
                return self.genotype.hash_spec() == other.genotype.hash_spec()
            else:
                # compared using reduced archs
                return (
                    np.array(self.genotype.matrix).tolist()
                    == np.array(other.genotype.matrix).tolist()
                ) and list(self.genotype.ops) == list(other.genotype.ops)

        # compared using original/non-reduced archs, might be wrong
        return (np.array(other.arch[0]).tolist(), list(other.arch[1])) == (
            np.array(self.arch[0]).tolist(),
            list(self.arch[1]),
        )


class NasBench101CompareController(BaseController):
    NAME = "nasbench-101-compare"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="compare",
        mode="eval",
        shuffle_indexes=True,
        schedule_cfg=None,
    ):
        super(NasBench101CompareController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

        self.shuffle_indexes = shuffle_indexes

        # get the infinite iterator of the model matrix and ops
        self.fixed_statistics = list(
            self.search_space.nasbench.fixed_statistics.values()
        )
        self.num_data = len(self.fixed_statistics)
        self.indexes = list(np.arange(self.num_data))
        self.comp_indexes = list(np.arange(self.num_data))
        self.cur_ind = 0
        self.cur_comp_ind = 1

    def sample(self, n=1, batch_size=None):
        assert batch_size is None
        rollouts = []
        n_r = 0
        while n_r < n:
            fixed_stat = self.fixed_statistics[self.indexes[self.cur_ind]]
            rollout_1 = NasBench101Rollout(
                fixed_stat["module_adjacency"],
                self.search_space.op_to_idx(fixed_stat["module_operations"]),
                search_space=self.search_space,
            )
            if self.comp_indexes[self.cur_comp_ind] != self.indexes[self.cur_ind]:
                fixed_stat_2 = self.fixed_statistics[
                    self.comp_indexes[self.cur_comp_ind]
                ]
                rollout_2 = NasBench101Rollout(
                    fixed_stat_2["module_adjacency"],
                    self.search_space.op_to_idx(
                        fixed_stat_2["module_operations"]),
                    search_space=self.search_space,
                )
                rollouts.append(
                    CompareRollout(rollout_1=rollout_1, rollout_2=rollout_2)
                )
                n_r += 1

            self.cur_comp_ind += 1
            if self.cur_comp_ind >= self.num_data:
                self.cur_comp_ind = 0
                if self.shuffle_indexes:
                    random.shuffle(self.comp_indexes)
                self.cur_ind += 1
                if self.cur_ind >= self.num_data:
                    self.logger.info("One epoch end")
                    self.cur_ind = 0
                    if self.shuffle_indexes:
                        random.shuffle(self.indexes)

        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["compare"]

    # ---- APIs that is not necessary ----
    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer, perf_name):
        return 0.0

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101Controller(BaseController):
    NAME = "nasbench-101"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-101",
        mode="eval",
        shuffle_indexes=True,
        avoid_repeat=False,
        schedule_cfg=None,
    ):
        super(NasBench101Controller, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

        self.shuffle_indexes = shuffle_indexes
        self.avoid_repeat = avoid_repeat

        # get the infinite iterator of the model matrix and ops
        self.fixed_statistics = list(
            self.search_space.nasbench.fixed_statistics.values()
        )
        self.num_data = len(self.fixed_statistics)
        self.indexes = list(np.arange(self.num_data))
        self.cur_ind = 0

        self.gt_rollouts = []
        self.gt_scores = []

    def sample(self, n, batch_size=None):
        rollouts = []
        if self.mode == "eval":
            # Return n archs seen(self.gt_rollouts) with best rewards
            # If number of the evaluated rollouts is smaller than n,
            # return random sampled rollouts
            #self.logger.info(
            #    "Return the best {} rollouts in the population".format(n))
            sampled_rollouts = []
            n_evaled = len(self.gt_rollouts)
            if n_evaled < n:
                sampled_rollouts = [
                    self.search_space.random_sample() for _ in range(n - n_evaled)
                ]
            best_inds = np.argpartition(self.gt_scores, -n)[-n:]
            return sampled_rollouts + [self.gt_rollouts[ind] for ind in best_inds]

        if self.avoid_repeat:
            # assert batch_size is None
            n_r = 0
            while n_r < n:
                fixed_stat = self.fixed_statistics[self.indexes[self.cur_ind]]
                rollouts.append(
                    NasBench101Rollout(
                        fixed_stat["module_adjacency"],
                        self.search_space.op_to_idx(
                            fixed_stat["module_operations"]),
                        search_space=self.search_space,
                    )
                )
                self.cur_ind += 1
                n_r += 1
                if self.cur_ind >= self.num_data:
                    self.logger.info("One epoch end")
                    self.cur_ind = 0
                    if self.shuffle_indexes:
                        random.shuffle(self.indexes)
        else:
            rollouts = [self.search_space.random_sample() for _ in range(n)]
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

    # ---- APIs that is not necessary ----
    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer, perf_name):
        num_rollouts_to_keep = 200
        self.gt_rollouts = self.gt_rollouts + rollouts
        self.gt_scores = self.gt_scores + \
            [r.get_perf(perf_name) for r in rollouts]
        if len(self.gt_rollouts) >= num_rollouts_to_keep:
            best_inds = np.argpartition(self.gt_scores, -num_rollouts_to_keep)[
                -num_rollouts_to_keep:
            ]
            self.gt_rollouts = [self.gt_rollouts[ind] for ind in best_inds]
            self.gt_scores = [self.gt_scores[ind] for ind in best_inds]
        return 0.0

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        return collections.OrderedDict()

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101FileSampleController(BaseController):
    NAME = "file-sampler"

    def __init__(
        self,
        search_space,
        device,
        archs_path,
        rollout_type="nasbench-101",
        mode="eval",
        shuffle_indexes=True,
        avoid_repeat=False,
        schedule_cfg=None,
    ):
        super(NasBench101FileSampleController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

        self.shuffle_indexes = shuffle_indexes
        self.avoid_repeat = avoid_repeat
        self.archs_path = archs_path

        with open(archs_path, "r") as fr:
            self.fixed_statistics = [eval(arch) for arch in yaml.load(fr)]

        self.num_data = len(self.fixed_statistics)
        self.indexes = list(np.arange(self.num_data))
        self.cur_ind = 0

        self.gt_rollouts = []
        self.gt_scores = []

    def sample(self, n, batch_size=None):
        rollouts = []
        genotypes = np.random.choice(self.fixed_statistics, n)
        rollouts = [
            NasBench101Rollout(
                g.matrix,
                self.search_space.op_to_idx(g.ops),
                search_space=self.search_space
            ) for g in genotypes
        ]
        if self.mode == "eval":
            # Return n archs seen(self.gt_rollouts) with best rewards
            # If number of the evaluated rollouts is smaller than n,
            # return random sampled rollouts
            sampled_rollouts = []
            n_evaled = len(self.gt_rollouts)
            if n_evaled < n:
                sampled_rollouts = [
                    self.search_space.random_sample() for _ in range(n - n_evaled)
                ]
            best_inds = np.argpartition(self.gt_scores, -n)[-n:]
            return sampled_rollouts + [self.gt_rollouts[ind] for ind in best_inds]

        if self.avoid_repeat:
            # assert batch_size is None
            n_r = 0
            while n_r < n:
                fixed_stat = self.fixed_statistics[self.indexes[self.cur_ind]]
                rollouts.append(
                    NasBench101Rollout(
                        fixed_stat["module_adjacency"],
                        self.search_space.op_to_idx(
                            fixed_stat["module_operations"]),
                        search_space=self.search_space,
                    )
                )
                self.cur_ind += 1
                n_r += 1
                if self.cur_ind >= self.num_data:
                    self.logger.info("One epoch end")
                    self.cur_ind = 0
                    if self.shuffle_indexes:
                        random.shuffle(self.indexes)
        else:
            rollouts = [self.search_space.random_sample() for _ in range(n)]
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

    # ---- APIs that is not necessary ----
    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer, perf_name):
        num_rollouts_to_keep = 200
        self.gt_rollouts = self.gt_rollouts + rollouts
        self.gt_scores = self.gt_scores + \
            [r.get_perf(perf_name) for r in rollouts]
        if len(self.gt_rollouts) >= num_rollouts_to_keep:
            best_inds = np.argpartition(self.gt_scores, -num_rollouts_to_keep)[
                -num_rollouts_to_keep:
            ]
            self.gt_rollouts = [self.gt_rollouts[ind] for ind in best_inds]
            self.gt_scores = [self.gt_scores[ind] for ind in best_inds]
        return 0.0

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        return collections.OrderedDict()

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101EvoController(BaseController):
    NAME = "nasbench-101-evo"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-101",
        mode="eval",
        population_nums=100,
        parent_pool_size=10,
        mutation_prob=1.0,
        schedule_cfg=None,
    ):
        super(NasBench101EvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

        # get pickle data
        # base_dir = os.path.join(utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-101")
        # train_data_path = os.path.join(base_dir, "nasbench_allv.pkl")
        # valid_data_path = os.path.join(base_dir, "nasbench_allv_valid.pkl")
        # with open(train_data_path, "rb") as f:
        #     train_data = pickle.load(f)
        # with open(valid_data_path, "rb") as f:
        #     valid_data = pickle.load(f)

        # self.all_data = train_data + valid_data
        # self.num_arch = len(self.all_data)

        # get the infinite iterator of the model matrix and ops
        self.mutation_prob = mutation_prob
        self.parent_pool_size = parent_pool_size
        self.cur_perf = None
        self.cur_solution = self.search_space.random_sample_arch()
        self.population_nums = population_nums
        self.population = collections.OrderedDict()
        # population_ind = np.random.choice(np.arange(len(self.search_space.nasbench.fixed_statistics)),
        # size=self.population_nums, replace=False)
        # for i in range(self.population_nums):
        #     rollout = self.search_space.random_sample()
        #     geno = rollout.genotype
        #     self.search_space.nasbench.query(geno)["validataion_accuracy"]
        #     data_i = self.all_data[population_ind[i]]
        #     geno = self.search_space.genotype(data_i[0])
        #     self.population[geno] = data_i[1]

        self.gt_rollouts = []
        self.gt_scores = []

    def set_init_population(self, rollout_list, perf_name):
        # clear the current population
        self.population = collections.OrderedDict()
        for r in rollout_list:
            self.population[r.genotype] = r.get_perf(perf_name)

    def sample(self, n, batch_size=None):
        # assert batch_size is None
        if self.mode == "eval":
            # Return n archs seen(self.gt_rollouts) with best rewards
            self.logger.info(
                "Return the best {} rollouts in the population".format(n))
            best_inds = np.argpartition(self.gt_scores, -n)[-n:]
            return [self.gt_rollouts[ind] for ind in best_inds]

        if len(self.population) < self.population_nums:
            self.logger.info("Population not full, random sample {}".format(n))
            return [self.search_space.random_sample() for _ in range(n)]

        population_items = list(self.population.items())
        choices = np.random.choice(
            np.arange(len(self.population)), size=self.parent_pool_size, replace=False
        )
        selected_pop = [population_items[i] for i in choices]
        new_archs = sorted(selected_pop, key=lambda x: x[1], reverse=True)
        rollouts = []
        ss = self.search_space
        cur_matrix, cur_ops = (
            new_archs[0][0].original_matrix,
            new_archs[0][0].original_ops,
        )
        while 1:
            new_matrix, new_ops = cur_matrix.copy(), cur_ops.copy()
            edge_mutation_prob = self.mutation_prob / ss.num_vertices
            for src in range(0, ss.num_vertices - 1):
                for dst in range(src + 1, ss.num_vertices):
                    if np.random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = self.mutation_prob / ss.num_ops
            for ind in range(1, ss.num_vertices - 1):
                if random.random() < op_mutation_prob:
                    available = [
                        o
                        for o in ss.nasbench.config["available_ops"]
                        if o != new_ops[ind]
                    ]
                    new_ops[ind] = np.random.choice(available)

            newspec = _ModelSpec(new_matrix, new_ops)
            if ss.nasbench.is_valid(newspec):
                rollouts.append(
                    NasBench101Rollout(
                        new_matrix,
                        ss.op_to_idx(cur_ops),
                        search_space=self.search_space,
                    )
                )
                return rollouts

    def step(self, rollouts, optimizer, perf_name):
        # best_rollout = rollouts[0]
        # for r in rollouts:
        #     if r.get_perf(perf_name) > best_rollout.get_perf(perf_name):
        #         best_rollout = r
        for best_rollout in rollouts:
            self.population[best_rollout.genotype] = best_rollout.get_perf(
                perf_name)
        if len(self.population) > self.population_nums:
            for key in list(self.population.keys())[
                : len(self.population) - self.population_nums
            ]:
                self.population.pop(key)

        # Save the best rollouts
        num_rollouts_to_keep = 200
        self.gt_rollouts = self.gt_rollouts + rollouts
        self.gt_scores = self.gt_scores + \
            [r.get_perf(perf_name) for r in rollouts]
        if len(self.gt_rollouts) >= num_rollouts_to_keep:
            best_inds = np.argpartition(self.gt_scores, -num_rollouts_to_keep)[
                -num_rollouts_to_keep:
            ]
            self.gt_rollouts = [self.gt_rollouts[ind] for ind in best_inds]
            self.gt_scores = [self.gt_scores[ind] for ind in best_inds]

        return 0

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

    # ---- APIs that is not necessary ----
    def set_device(self, device):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101SAController(BaseController):
    NAME = "nasbench-101-sa"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-101",
        mode="eval",
        temperature=1000,
        anneal_coeff=0.95,
        mutation_edges_prob=0.5,
        schedule_cfg=None,
    ):
        super(NasBench101SAController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

        # get the infinite iterator of the model matrix and ops
        self.temperature = temperature
        self.anneal_coeff = anneal_coeff
        self.mutation_edges_prob = mutation_edges_prob
        self.cur_perf = None
        self.cur_solution = self.search_space.random_sample_arch()

    def reinit(self):
        self.cur_perf = None
        self.cur_solution = self.search_space.random_sample_arch()

    def set_init_population(self, rollout_list, perf_name):
        # set the initialization to the best rollout in the list
        perf_list = [r.get_perf(perf_name) for r in rollout_list]
        best_rollout = rollout_list[np.argmax(perf_list)]
        self.cur_solution = best_rollout.arch
        self.cur_perf = best_rollout.get_perf(perf_name)
        self.logger.info(
            "Set the initialization rollout: {}; perf: {}".format(
                best_rollout, self.cur_perf
            )
        )

    def sample(self, n, batch_size=None):
        if self.mode == "eval":
            # return the current rollout
            return [
                NasBench101Rollout(*self.cur_solution,
                                   search_space=self.search_space)
            ] * n

        assert batch_size is None
        rollouts = []
        cur_matrix, cur_ops = self.cur_solution
        ss = self.search_space
        for n_r in range(n):
            if np.random.rand() < self.mutation_edges_prob:
                while 1:
                    edge_ind = np.random.randint(
                        0, ss.num_possible_edges, size=1)
                    while (
                        graph_util.num_edges(cur_matrix) == ss.max_edges
                        and cur_matrix[ss.idx[0][edge_ind], ss.idx[1][edge_ind]] == 0
                    ):
                        edge_ind = np.random.randint(
                            0, ss.num_possible_edges, size=1)
                    new_matrix = cur_matrix.copy()
                    new_matrix[ss.idx[0][edge_ind], ss.idx[1][edge_ind]] = (
                        1 - cur_matrix[ss.idx[0][edge_ind],
                                       ss.idx[1][edge_ind]]
                    )
                    new_rollout = NasBench101Rollout(
                        new_matrix, cur_ops, search_space=self.search_space
                    )
                    try:
                        ss.nasbench._check_spec(new_rollout.genotype)
                    except api.OutOfDomainError:
                        # ignore out-of-domain archs (disconnected)
                        continue
                    else:
                        cur_matrix = new_matrix
                        break
            else:
                while 1:
                    ops_ind = np.random.randint(0, ss.num_ops, size=1)[0]
                    new_ops = np.random.randint(0, ss.num_op_choices, size=1)[0]
                    while new_ops == cur_ops[ops_ind]:
                        new_ops = np.random.randint(0, ss.num_op_choices, size=1)[0]
                    cur_ops[ops_ind] = new_ops
                    new_rollout = NasBench101Rollout(cur_matrix, cur_ops,
                            search_space=self.search_space)
                    try:
                        ss.nasbench._check_spec(new_rollout.genotype)
                    except api.OutOfDomainError:
                        continue
                    else:
                        break
            rollouts.append(NasBench101Rollout(
                cur_matrix,
                cur_ops,
                search_space=self.search_space
            ))
        return rollouts

    def step(self, rollouts, optimizer, perf_name):
        # assert len(rollouts) == 1
        ind = np.argmax([r.get_perf(perf_name) for r in rollouts])
        rollout = rollouts[ind]
        new_perf = rollout.get_perf(perf_name)
        if self.cur_perf is None or self.cur_perf < new_perf:
            self.cur_perf = new_perf
            self.cur_solution = rollout.arch
        elif np.exp((new_perf - self.cur_perf) / self.temperature) > np.random.rand():
            self.cur_perf = new_perf
            self.cur_solution = rollout.arch
        self.temperature *= self.anneal_coeff
        return 0

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

    # ---- APIs that is not necessary ----
    def set_device(self, device):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101Evaluator(BaseEvaluator):
    NAME = "nasbench-101"

    def __init__(
        self,
        dataset,
        weights_manager,
        objective,
        rollout_type="nasbench-101",
        use_epoch=108,
        use_mean_valid_as_reward=False,
        schedule_cfg=None,
    ):
        super(NasBench101Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type
        )

        self.use_epoch = use_epoch
        self.use_mean_valid_as_reward = use_mean_valid_as_reward

    @classmethod
    def supported_data_types(cls):
        # cifar10
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101", "compare"]

    def suggested_controller_steps_per_epoch(self):
        return None

    def suggested_evaluator_steps_per_epoch(self):
        return 0

    def evaluate_rollouts(
        self,
        rollouts,
        is_training=False,
        portion=None,
        eval_batches=None,
        return_candidate_net=False,
        callback=None,
    ):
        if self.rollout_type == "compare":
            eval_rollouts = sum([[r.rollout_1, r.rollout_2]
                                 for r in rollouts], [])
        else:
            eval_rollouts = rollouts

        for rollout in eval_rollouts:
            if not self.use_mean_valid_as_reward:
                query_res = rollout.search_space.nasbench.query(
                    rollout.genotype)
                # could use other performance, this functionality is not compatible with objective
                rollout.set_perf(query_res["validation_accuracy"])

            # use mean for test acc
            # can use other fidelity too
            res = rollout.search_space.nasbench.get_metrics_from_spec(rollout.genotype)[
                1
            ][self.use_epoch]
            mean_valid_acc = np.mean(
                [s_res["final_validation_accuracy"] for s_res in res]
            )
            if self.use_mean_valid_as_reward:
                rollout.set_perf(mean_valid_acc)
            rollout.set_perf(mean_valid_acc, name="mean_valid_acc")
            mean_test_acc = np.mean(
                [s_res["final_test_accuracy"] for s_res in res])
            rollout.set_perf(mean_test_acc, name="mean_test_acc")

        if self.rollout_type == "compare":
            num_r = len(rollouts)
            for i_rollout in range(num_r):
                diff = (
                    eval_rollouts[2 * i_rollout + 1].perf["reward"]
                    - eval_rollouts[2 * i_rollout].perf["reward"]
                )
                better = diff > 0
                rollouts[i_rollout].set_perfs(
                    collections.OrderedDict(
                        [
                            ("compare_result", better),
                            ("diff", diff),
                        ]
                    )
                )
        return rollouts

    # ---- APIs that is not necessary ----
    def update_evaluator(self, controller):
        pass

    def update_rollouts(self, rollouts):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


# ---- embedders for NASBench-101 ----
# TODO: the multi stage trick could apply for all the embedders
class NasBench101_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "nb101-lstm"

    def __init__(
        self,
        search_space,
        num_hid=100,
        emb_hid=100,
        num_layers=1,
        use_mean=False,
        use_hid=False,
        schedule_cfg=None,
    ):
        super(NasBench101_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.op_emb = nn.Embedding(
            self.search_space.num_op_choices, self.emb_hid)
        self.conn_emb = nn.Embedding(2, self.emb_hid)

        self.rnn = nn.LSTM(
            input_size=self.emb_hid,
            hidden_size=self.num_hid,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.out_dim = num_hid
        self._triu_indices = np.triu_indices(VERTICES, k=1)

    def forward(self, archs):
        x_1 = np.array([arch[0][self._triu_indices] for arch in archs])
        x_2 = np.array([arch[1] for arch in archs])

        conn_embs = self.conn_emb(torch.LongTensor(
            x_1).to(self.op_emb.weight.device))
        op_embs = self.op_emb(torch.LongTensor(
            x_2).to(self.op_emb.weight.device))
        emb = torch.cat((conn_embs, op_embs), dim=-2)

        out, (h_n, _) = self.rnn(emb)

        if self.use_hid:
            y = h_n[0]
        else:
            if self.use_mean:
                y = torch.mean(out, dim=1)
            else:
                # return final output
                y = out[:, -1, :]
        return y


class NasBench101_SimpleSeqEmbedder(ArchEmbedder):
    NAME = "nb101-seq"

    def __init__(self, search_space, use_all_adj_items=False, schedule_cfg=None):
        super(NasBench101_SimpleSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.use_all_adj_items = use_all_adj_items
        self.out_dim = 49 + 5 if use_all_adj_items else 21 + 5
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        if self.use_all_adj_items:
            x = np.concatenate(
                [
                    np.array([arch[0].reshape(-1) for arch in archs]),
                    np.array([arch[1] for arch in archs]),
                ],
                axis=-1,
            )
        else:
            triu_indices = np.triu_indices(VERTICES, k=1)
            x_1 = np.array([arch[0][triu_indices] for arch in archs])
            x_2 = np.array([arch[1] for arch in archs])
            x = np.concatenate([x_1, x_2], axis=-1)
        return self._placeholder_tensor.new(x)


class NasBench101ReNASEmbedder(ArchEmbedder):
    NAME = "nb101-renas"

    def __init__(self, search_space, use_type_matrix_only=False, schedule_cfg=None):
        super(NasBench101ReNASEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.use_type_matrix_only = use_type_matrix_only
        # calculate FLOPs of each cell
        self.conv1 = nn.Conv2d(
            1 if self.use_type_matrix_only else 19, 38, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(38)
        self.conv2 = nn.Conv2d(38, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.out_dim = 6272  # 128 * 7 * 7

        # for now, i hard code these infos...
        # 0 -> 2, 1 -> 3, 2 -> 4, 3(none) -> 0
        self.op_ind_map = np.array([2, 3, 4, 0])
        self.kernel_mul = np.array([0, 0, 1, 9, 0, 0])  # 1x1, 3x3 conv

    def _calculate_flops(self, op_ind):
        pass

    def _compute_vertex_channels(self, input_channels, output_channels, matrix):
        """Copied from nasbench.lib.model_builder

        Computes the number of channels at every vertex.

        Given the input channels and output channels, this calculates the number of
        channels at each interior vertex. Interior vertices have the same number of
        channels as the max of the channels of the vertices it feeds into. The output
        channels are divided amongst the vertices that are directly connected to it.
        When the division is not even, some vertices may receive an extra channel to
        compensate.

        Args:
          input_channels: input channel count.
          output_channels: output channel count.
          matrix: adjacency matrix for the module (pruned by model_spec).

        Returns:
          list of channel counts, in order of the vertices.
        """
        num_vertices = np.shape(matrix)[0]

        vertex_channels = [0] * num_vertices
        vertex_channels[0] = input_channels
        vertex_channels[num_vertices - 1] = output_channels
        # if num_vertices == 2:
        #     # Edge case where module only has input and output vertices
        #   return vertex_channels

        # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
        # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
        in_degree = np.sum(matrix[1:], axis=0)
        interior_channels = output_channels // in_degree[num_vertices - 1]
        # Remainder to add
        correction = output_channels % in_degree[num_vertices - 1]

        # Set channels of vertices that flow directly to output
        for v in range(1, num_vertices - 1):
            if matrix[v, num_vertices - 1]:
                vertex_channels[v] = interior_channels
                if correction:
                    vertex_channels[v] += 1
                    correction -= 1

        # Set channels for all other vertices to the max of the out edges, going
        # backwards. (num_vertices - 2) index skipped because it only connects to
        # output.
        for v in range(num_vertices - 3, 0, -1):
            if not matrix[v, num_vertices - 1]:
                for dst in range(v + 1, num_vertices - 1):
                    if matrix[v, dst]:
                        vertex_channels[v] = max(
                            vertex_channels[v], vertex_channels[dst]
                        )
                        assert vertex_channels[v] > 0

        # Sanity check, verify that channels never increase and final channels add up.
        final_fan_in = 0
        for v in range(1, num_vertices - 1):
            if matrix[v, num_vertices - 1]:
                final_fan_in += vertex_channels[v]
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    assert vertex_channels[v] >= vertex_channels[dst]
        assert final_fan_in == output_channels or num_vertices == 2
        # num_vertices == 2 means only input/output nodes, so 0 fan-in

        return vertex_channels

    def embed_and_transform_arch(self, archs):
        arch_feats = []
        for arch in archs:
            op_inds = np.concatenate([[1], self.op_ind_map[arch[1]], [5]])
            if not self.use_type_matrix_only:
                features = []
                for c_pair in [
                    (128, 128),
                    (128, 256),
                    (256, 256),
                    (256, 512),
                    (512, 512),
                ]:
                    # o_cs = nasbench.lib.model_builder.compute_vertex_channels(*c_pair, arch[0])
                    o_cs = self._compute_vertex_channels(*c_pair, arch[0])
                    # the ReNAS paper do not specifiy how they handle this
                    spatial_sz = 28 / (c_pair[1] // 128)
                    spatial_mul = spatial_sz * spatial_sz
                    params = self.kernel_mul[op_inds] * arch[0] * o_cs * o_cs
                    # add 1x1 projection params/flops from the input node
                    params[0] = params[0] + arch[0][0] * o_cs[0] * o_cs  # 1x1
                    flops = params * spatial_mul
                    features.append([flops, params])
                arch_feats.append(
                    np.stack(
                        [op_inds * arch[0]]
                        + features[0] * 3
                        + features[1]
                        + features[2] * 2
                        + features[3]
                        + features[4] * 2
                    )
                )
            else:
                arch_feats.append(np.expand_dims(op_inds * arch[0], axis=0))
        ims = torch.FloatTensor(np.stack(arch_feats)).to(
            self.conv1.weight.device)
        return ims

    def forward(self, archs):
        ims = self.embed_and_transform_arch(archs)
        ims = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(ims))))))
        return ims.reshape([ims.shape[0], -1])


class NasBench101ArchEmbedder(ArchEmbedder):
    NAME = "nb101-gcn"

    def __init__(
        self,
        search_space,
        embedding_dim=48,
        hid_dim=48,
        gcn_out_dims=[128, 128],
        gcn_kwargs=None,
        dropout=0.0,
        use_global_node=False,
        use_final_only=False,
        schedule_cfg=None,
    ):
        super(NasBench101ArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.none_op_ind = self.search_space.none_op_ind
        self.embedding_dim = embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_global_node = use_global_node
        self.use_final_only = use_final_only
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices

        self.input_op_emb = nn.Embedding(1, self.embedding_dim)
        # zero is ok
        self.output_op_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)))
        # requires_grad=False)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.embedding_dim)))

        self.op_emb = nn.Embedding(self.num_op_choices, self.embedding_dim)
        self.x_hidden = nn.Linear(self.embedding_dim, self.hid_dim)

        # init graph convolutions
        self.gcns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphConvolution(
                in_dim, dim, **(gcn_kwargs or {})))
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.weight.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.weight.new(
            [arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim=2,
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim=1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices + 1, 1), device=adjs.device
                    ),
                ),
                dim=2,
            )
        node_embs = self.op_emb(op_inds)  # (batch_size, vertices - 2, emb_dim)
        b_size = node_embs.shape[0]
        if self.use_global_node:
            node_embs = torch.cat(
                (
                    self.input_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    node_embs,
                    self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        else:
            node_embs = torch.cat(
                (
                    self.input_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    node_embs,
                    self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_inds

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        adjs, x, op_inds = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding
            # throw away padded info here
            y = torch.cat(
                (
                    y[:, :-1, :]
                    * (op_inds != self.none_op_ind)[:,
                                                    :, None].to(torch.float32),
                    y[:, -1:, :],
                ),
                dim=1,
            )
            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NasBench101FlowArchEmbedder(ArchEmbedder):
    NAME = "nb101-flow"

    def __init__(
        self,
        search_space,
        op_embedding_dim=48,
        node_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        share_op_attention=False,
        other_node_zero=False,
        gcn_kwargs=None,
        use_bn=False,
        use_global_node=False,
        use_final_only=False,
        input_op_emb_trainable=False,
        dropout=0.0,
        schedule_cfg=None,
    ):
        super(NasBench101FlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.use_global_node = use_global_node
        self.share_op_attention = share_op_attention
        self.input_op_emb_trainable = input_op_emb_trainable
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind

        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad=not other_node_zero
        )
        # self.middle_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)
        # # zero is ok
        # self.output_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad=self.input_op_emb_trainable,
        )
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(
                self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim if not self.share_op_attention else dim,
                    has_attention=not self.share_op_attention,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.new([arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim=2,
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim=1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices, 1), device=adjs.device),
                ),
                dim=2,
            )
        # (batch_size, vertices - 2, op_emb_dim)
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        else:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                ),
                dim=1,
            )
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat(
                    [b_size, self.vertices - 1, 1]),
            ),
            dim=1,
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs, op_inds

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat(
                    (
                        y[:, :-2, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -2:, :],
                    ),
                    dim=1,
                )
            else:
                y = torch.cat(
                    (
                        y[:, :-1, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -1:, :],
                    ),
                    dim=1,
                )

            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NasBench101FbFlowArchEmbedder(ArchEmbedder):
    """
    Implement of TA-GATES architecture embedder on NAS-Bench-101.
    """
    NAME = "nb101-fbflow"

    def __init__(
        self,
        search_space,
        op_embedding_dim: int = 48,
        node_embedding_dim: int = 48,
        hid_dim: int = 96,
        gcn_out_dims: List[int] = [128, 128, 128, 128, 128],
        share_op_attention: bool = False,
        other_node_zero: bool = False,
        gcn_kwargs: dict = None,
        use_bn: bool = False,
        use_global_node: bool = False,
        use_final_only: bool = False,
        input_op_emb_trainable: bool = False,
        dropout: float = 0.,

        ## newly added
        # construction (tagates)
        num_time_steps: int = 2,
        fb_conversion_dims: List[int] = [128, 128],
        backward_gcn_out_dims: List[int] = [128, 128, 128, 128, 128],
        updateopemb_method: str = "concat_ofb", # concat_ofb, concat_fb, concat_b
        updateopemb_dims: List[int] = [128],
        updateopemb_scale: float = 0.1,
        b_use_bn: bool = False,
        # construction (l): concat arch-level zeroshot as l
        concat_arch_zs_as_l_dimension=None,
        concat_l_layer: int = 0,
        # construction (symmetry breaking)
        symmetry_breaking_method: str = None, # None, "random", "param_zs", "param_zs_add"
        concat_param_zs_as_opemb_dimension = None,
        concat_param_zs_as_opemb_mlp: List[int] = [64, 128],
        concat_param_zs_as_opemb_scale: float = 0.1,

        # gradident flow configurations
        detach_vinfo: bool = False,
        updateopemb_detach_opemb: bool = True,
        updateopemb_detach_finfo: bool = True,

        mask_nonparametrized_ops: bool = False,
        schedule_cfg = None
    ) -> None:
        super(NasBench101FbFlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.use_global_node = use_global_node
        self.share_op_attention = share_op_attention
        self.input_op_emb_trainable = input_op_emb_trainable
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind

        # newly added
        self.detach_vinfo = detach_vinfo
        self.num_time_steps = num_time_steps
        self.fb_conversion_dims = fb_conversion_dims
        self.backward_gcn_out_dims = backward_gcn_out_dims
        self.b_use_bn = b_use_bn
        self.updateopemb_method = updateopemb_method
        self.updateopemb_detach_opemb = updateopemb_detach_opemb
        self.updateopemb_detach_finfo = updateopemb_detach_finfo
        self.updateopemb_dims = updateopemb_dims
        self.updateopemb_scale = updateopemb_scale
        # concat arch-level zs as l
        self.concat_arch_zs_as_l_dimension = concat_arch_zs_as_l_dimension
        self.concat_l_layer = concat_l_layer
        if self.concat_arch_zs_as_l_dimension is not None:
            assert self.concat_l_layer < len(self.fb_conversion_dims)

        # symmetry breaking
        self.symmetry_breaking_method = symmetry_breaking_method
        self.concat_param_zs_as_opemb_dimension = concat_param_zs_as_opemb_dimension
        assert self.symmetry_breaking_method in {None, "param_zs", "random", "param_zs_add"}
        self.concat_param_zs_as_opemb_scale = concat_param_zs_as_opemb_scale

        if self.symmetry_breaking_method == "param_zs_add":
            in_dim = self.concat_param_zs_as_opemb_dimension
            self.param_zs_embedder = []
            for embedder_dim in concat_param_zs_as_opemb_mlp:
                self.param_zs_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.param_zs_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.param_zs_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.param_zs_embedder = nn.Sequential(*self.param_zs_embedder)

        self.mask_nonparametrized_ops = mask_nonparametrized_ops

        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad = not other_node_zero
        )

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(
            torch.zeros(1, self.op_embedding_dim),
            requires_grad = self.input_op_emb_trainable
        )
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(
                torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(
                self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else\
                    (self.op_embedding_dim if not self.share_op_attention else dim),
                    has_attention=not self.share_op_attention,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

        # init backward graph convolutions
        self.b_gcns = []
        self.b_bns = []
        if self.concat_arch_zs_as_l_dimension is not None \
           and self.concat_l_layer == len(self.fb_conversion_dims) - 1:
            in_dim = self.fb_conversion_dims[-1] + self.concat_arch_zs_as_l_dimension
        else:
            in_dim = self.fb_conversion_dims[-1]
        for dim in self.backward_gcn_out_dims:
            self.b_gcns.append(
                DenseGraphFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else self.op_embedding_dim,
                    has_attention = not self.share_op_attention,
                    reverse = True,
                    **(gcn_kwargs or {})
                )
            )
            in_dim = dim
            if self.b_use_bn:
                self.b_bns.append(nn.BatchNorm1d(self.vertices))
        self.b_gcns = nn.ModuleList(self.b_gcns)
        if self.b_use_bn:
            self.b_bns = nn.ModuleList(self.b_bns)
        self.num_b_gcn_layers = len(self.b_gcns)

        # init the network to convert forward output info into backward input info
        if self.num_time_steps > 1:
            self.fb_conversion_list = []
            dim = self.gcn_out_dims[-1]
            num_fb_layers = len(fb_conversion_dims)
            self._num_before_concat_l = None
            for i_dim, fb_conversion_dim in enumerate(fb_conversion_dims):
                self.fb_conversion_list.append(nn.Linear(dim, fb_conversion_dim))
                if i_dim < num_fb_layers - 1:
                    self.fb_conversion_list.append(nn.ReLU(inplace=False))
                if self.concat_arch_zs_as_l_dimension is not None and \
                   self.concat_l_layer == i_dim:
                    dim = fb_conversion_dim + self.concat_arch_zs_as_l_dimension
                    self._num_before_concat_l = len(self.fb_conversion_list)
                else:
                    dim = fb_conversion_dim
            self.fb_conversion = nn.Sequential(*self.fb_conversion_list)

            # init the network to get delta op_emb
            if self.updateopemb_method == "concat_ofb":
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] \
                         + self.op_embedding_dim
            elif self.updateopemb_method == "concat_fb":
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1]
            elif self.updateopemb_method == "concat_b":
                in_dim = self.backward_gcn_out_dims[-1]
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.new([arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device = adjs.device)
            tmp_cat = torch.cat(
                (
                    tmp_ones,
                    (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                    tmp_ones,
                ),
                dim = 2
            )
            adjs = torch.cat(
                (
                    torch.cat((adjs, tmp_cat), dim = 1),
                    torch.zeros(
                        (adjs.shape[0], self.vertices, 1), device = adjs.device),
                ),
                dim = 2
            )
        # (batch_size, vertices - 2, op_emb_dim)
        op_embs = self.op_emb(op_inds)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                    self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                ),
                dim = 1
            )
        else:
            op_embs = torch.cat(
                (
                    self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                    op_embs,
                    self.output_op_emb.weight.unsqueeze(
                        0).repeat([b_size, 1, 1]),
                ),
                dim = 1
            )
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat(
                    [b_size, self.vertices - 1, 1]),
            ),
            dim = 1
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs, op_inds

    def _forward_pass(self, x, adjs, auged_op_emb) -> Tensor:
        # --- forward pass ---
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, auged_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training = self.training)

        return y

    def _backward_pass(self, y, adjs, zs_as_l, auged_op_emb) -> Tensor:
        # --- backward pass ---
        # get the information of the output node
        # b_info = torch.cat(
        #     (
        #         torch.zeros([y.shape[0], self.vertices - 1, y.shape[-1]], device=y.device),
        #         y[:, -1:, :]
        #     ),
        #     dim=1
        # )
        b_info = y[:, -1:, :]
        if self.detach_vinfo:
            b_info = b_info.detach()
        if self.concat_arch_zs_as_l_dimension:
            # process before concat l
            b_info = self.fb_conversion[:self._num_before_concat_l](b_info)
            # concat l
            b_info = torch.cat((b_info, zs_as_l.unsqueeze(-2)), dim = -1)
            if not self.concat_l_layer == len(self.fb_conversion_list) - 1:
                # process after concat l
                b_info = self.fb_conversion[self._num_before_concat_l:](b_info)
        else:
            b_info = self.fb_conversion(b_info)
        b_info = torch.cat(
            (
                torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device = y.device),
                b_info
            ),
            dim = 1
        )

        # start backward flow
        b_adjs = adjs.transpose(1, 2)
        b_y = b_info
        for i_layer, gcn in enumerate(self.b_gcns):
            b_y = gcn(b_y, b_adjs, auged_op_emb)
            if self.b_use_bn:
                shape_y = b_y.shape
                b_y = self.b_bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1]))\
                            .reshape(shape_y)
            if i_layer != self.num_b_gcn_layers - 1:
                b_y = F.relu(b_y)
                b_y = F.dropout(b_y, self.dropout, training = self.training)

        return b_y

    def _update_op_emb(self, y, b_y, op_emb, concat_op_emb_mask) -> Tensor:
        # --- UpdateOpEmb ---
        if self.updateopemb_method == "concat_ofb":
            in_embedding = torch.cat(
                (
                    op_emb.detach() if self.updateopemb_detach_opemb else op_emb,
                    y.detach() if self.updateopemb_detach_finfo else y,
                    b_y
                ),
                dim = -1)
        elif self.updateopemb_method == "concat_fb":
            in_embedding = torch.cat(
                (
                    y.detach() if self.updateopemb_detach_finfo else y,
                    b_y
                ), dim = -1)
        elif self.updateopemb_method == "concat_b":
            in_embedding = b_y
        update = self.updateop_embedder(in_embedding)

        if self.mask_nonparametrized_ops:
            update = update * concat_op_emb_mask

        op_emb = op_emb + self.updateopemb_scale * update
        return op_emb

    def _final_process(self, y: Tensor, op_inds) -> Tensor:
        ## --- output ---
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat(
                    (
                        y[:, :-2, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -2:, :],
                    ),
                    dim = 1
                )
            else:
                y = torch.cat(
                    (
                        y[:, :-1, :]
                        * (op_inds !=
                           self.none_op_ind)[:, :, None].to(torch.float32),
                        y[:, -1:, :],
                    ),
                    dim = 1
                )

            y = torch.mean(y, dim = 1)  # average across nodes (bs, god)

        return y

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_p = archs
                zs_as_l = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        concat_op_emb_mask = ((op_inds == 0) | (op_inds == 1))
        concat_op_emb_mask = F.pad(concat_op_emb_mask, (1, 1), mode = "constant")
        concat_op_emb_mask = concat_op_emb_mask.unsqueeze(-1).to(torch.float32)

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_emb).normal_() * 0.1
            op_emb = op_emb + noise
        elif self.symmetry_breaking_method == "param_zs":
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            assert zs_as_p.shape[-1] == self.concat_param_zs_as_opemb_dimension
        elif self.symmetry_breaking_method == "param_zs_add":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            zs_as_p = self.param_zs_embedder(zs_as_p)
            op_emb = op_emb + zs_as_p * self.concat_param_zs_as_opemb_scale

        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)

        for t in range(self.num_time_steps):
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_emb = torch.cat((op_emb, zs_as_p), dim = -1)
            else:
                auged_op_emb = op_emb

            y = self._forward_pass(x, adjs, auged_op_emb)

            if t == self.num_time_steps - 1:
                break

            b_y = self._backward_pass(y, adjs, zs_as_l, auged_op_emb)
            op_emb = self._update_op_emb(y, b_y, op_emb, concat_op_emb_mask)

        ## --- output ---
        # y: (batch_size, vertices, gcn_out_dims[-1])
        return self._final_process(y, op_inds)


class NasBench101FbFlowAnyTimeArchEmbedder(NasBench101FbFlowArchEmbedder):
    NAME = "nb101-fbflow-anytime"

    def forward(self, archs, any_time: bool = False):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        if not any_time:
            return super(NasBench101FbFlowAnyTimeArchEmbedder, self).forward(archs)

        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_p = archs
                zs_as_l = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        concat_op_emb_mask = ((op_inds == 0) | (op_inds == 1))
        concat_op_emb_mask = F.pad(concat_op_emb_mask, (1, 1), mode = "constant")
        concat_op_emb_mask = concat_op_emb_mask.unsqueeze(-1).to(torch.float32)

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_emb).normal_() * 0.1
            op_emb = op_emb + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            assert zs_as_p.shape[-1] == self.concat_param_zs_as_opemb_dimension
        elif self.symmetry_breaking_method == "param_zs_add":
            zs_as_p = self.input_op_emb.new(np.array(zs_as_p))
            zs_as_p = self.param_zs_embedder(zs_as_p)
            op_emb = op_emb + zs_as_p * self.concat_param_zs_as_opemb_scale

        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)

        y_cache = []

        for t in range(self.num_time_steps):
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_emb = torch.cat((op_emb, zs_as_p), dim = -1)
            else:
                auged_op_emb = op_emb

            y = self._forward_pass(x, adjs, auged_op_emb)
            y_cache.append(self._final_process(y, op_inds))

            if t == self.num_time_steps - 1:
                break

            b_y = self._backward_pass(y, adjs, zs_as_l, auged_op_emb)
            op_emb = self._update_op_emb(y, b_y, op_emb, concat_op_emb_mask)

        return y_cache


class NasBench101SuperNet(BaseWeightsManager, nn.Module):
    NAME = "nasbench-101"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        gpus=tuple(),
        num_classes=10,
        init_channels=16,
        stem_multiplier=1,
        max_grad_norm=5.0,
        dropout_rate=0.1,
        use_stem="conv_bn_relu_3x3",
        stem_stride=1,
        stem_affine=True,
        post_process_op="relu_conv_bn_1x1",
        candidate_eval_no_grad=True,
        cell_group_kwargs=None,
    ):

        super(NasBench101SuperNet, self).__init__(
            search_space, device, rollout_type)
        nn.Module.__init__(self)

        self.gpus = gpus
        self.num_classes = num_classes
        self.init_channels = init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem

        self._num_init = self.search_space.num_init_nodes
        self._cell_layout = self.search_space.cell_layout
        self._reduce_cgs = self.search_space.reduce_cell_groups
        self._num_layers = self.search_space.num_layers

        # training
        self.max_grad_norm = max_grad_norm
        self.dropout_rate = dropout_rate

        self.candidate_eval_no_grad = candidate_eval_no_grad

        if not self.use_stem:
            c_stem = 3
            init_strides = [1] * self._num_init
        elif isinstance(self.use_stem, (list, tuple)):
            self.stems = []
            c_stem = self.stem_multiplier * self.init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stems.append(
                    get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )
            self.stems = nn.ModuleList(self.stems)
            init_strides = [stem_stride] * self._num_init
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )
            init_strides = [1] * self._num_init

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        strides = [
            2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)
        ]

        for i_layer, stride in enumerate(strides):
            if cell_group_kwargs is not None:
                # support passing in different kwargs when instantializing
                # cell class for different cell groups
                kwargs = {
                    k: v
                    for k, v in cell_group_kwargs[self._cell_layout[i_layer]].items()
                }
            else:
                kwargs = {}
            # A patch: Can specificy input/output channels by hand in configuration,
            # instead of relying on the default
            # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
            _num_channels = num_channels if "C_in" not in kwargs else kwargs.pop(
                "C_in")
            _num_out_channels = (
                num_channels *
                stride if "C_out" not in kwargs else kwargs.pop("C_out")
            )
            cell = NasBench101SharedCell(
                NasBench101SharedOp,
                self.search_space,
                layer_index=i_layer,
                num_channels=_num_channels,
                num_out_channels=_num_out_channels,
                stride=stride,
                steps=self.search_space.get_num_steps(i_layer),
                **kwargs
            )
            self.cells.append(cell)
            if stride > 1:
                num_channels *= stride

        self.post_process = get_op(post_process_op)(
            num_channels * self.search_space.get_num_steps(i_layer),
            _num_out_channels,
            1,
            False,
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = Identity()
        self.classifier = nn.Linear(_num_out_channels, self.num_classes)

        self.to(self.device)

    def _is_reduce(self, layer_idx):
        return self._cell_layout[layer_idx] in self._reduce_cgs

    def set_device(self, device):
        self.device = device
        self.to(device)

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

    def assemble_candidate(self, rollout):
        return NasBench101CandidateNet(
            self, rollout, gpus=self.gpus, eval_no_grad=self.candidate_eval_no_grad
        )

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.max_grad_norm)
        # apply the gradients
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

    def forward(self, inputs, genotypes, **kwargs):
        if not self.use_stem:
            stemed = inputs
            states = [inputs] * self._num_init
        elif isinstance(self.use_stem, (list, tuple)):
            states = []
            stemed = inputs
            for stem in self.stems:
                stemed = stem(stemed)
                states.append(stemed)
        else:
            stemed = self.stem(inputs)
            states = [stemed] * self._num_init

        state = states[-1]
        for cg_idx, cell in zip(self._cell_layout, self.cells):
            # genotype = genotypes[cg_idx]
            state = cell(state, genotypes, **kwargs)

        state = self.post_process(state)
        out = self.global_pooling(state)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


class NasBench101CandidateNet(CandidateNet):
    def __init__(self, super_net, rollout, gpus=tuple(), eval_no_grad=True):
        super(NasBench101CandidateNet, self).__init__(
            eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = super_net.device
        self.gpus = gpus
        self.search_space = super_net.search_space

        self._flops_calculated = False
        self.total_flops = 0

        self.rollout = rollout

    def get_device(self):
        return self._device

    def forward(self, inputs, single=False):  # pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs={"single": True})

    def _forward(self, inputs):
        return self.super_net.forward(
            inputs,
            (self.rollout.genotype.original_matrix,
             self.rollout.genotype.original_ops),
        )


class NasBench101SharedCell(nn.Module):
    def __init__(
        self,
        op_cls,
        search_space,
        layer_index,
        num_channels,
        num_out_channels,
        stride,
        steps=4,
        **op_kwargs
    ):
        super(NasBench101SharedCell, self).__init__()

        self.search_space = search_space
        self.layer_index = layer_index
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self._steps = steps
        self.stride = stride

        self._primitives = [
            op for op in self.search_space.ops_choices if op != "none"]
        self.reducer = nn.Sequential()
        if stride == 2:
            self.reducer = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        C_in = (
            self.num_channels if layer_index == 0 else self.num_channels * self._steps
        )
        self.shared_ops = nn.ModuleList()
        self.projections = nn.ModuleList()
        for i_step in range(self._steps):
            shared_op = op_cls(
                self.num_out_channels,
                self.num_out_channels,
                primitives=self._primitives,
                op_mapping=search_space.op_mapping,
                **op_kwargs
            )
            self.shared_ops.append(shared_op)

            projection = get_op("conv_bn_relu_1x1")(
                C_in, self.num_out_channels, 1, False
            )
            self.projections.append(projection)

        # from input node to output node
        self.projections.append(
            get_op("conv_bn_relu_1x1")(
                C=C_in,
                C_out=self.num_out_channels * self._steps,
                stride=1,
                affine=False,
            )
        )

    def forward(self, inputs, genotype):
        matrix, ops = genotype
        states = [inputs]

        ops_idx = self.search_space.op_to_idx(ops)
        connections, concat_nodes = self.search_space.matrix_to_connection(
            matrix)

        states[0] = self.reducer(states[0])

        for to_ in range(1, self._steps + 1):
            from_nodes = connections.get(to_, [])
            # index of nodes starts with 1 rather than 0
            projected_input = self.projections[to_ - 1](states[0])
            new_states = [projected_input, *states[1:]]
            out = sum(
                self.shared_ops[to_ - 1](new_states[from_], ops_idx[to_ - 1])
                for from_ in from_nodes
            )
            if out is 0:
                out = new_states[-1]
            states.append(out)

        concat_weights = torch.zeros([1 + self._steps]).to(states[0].device)
        concat_weights[concat_nodes] = 1

        projected_input = self.projections[-1](states[0])

        cat_tensor = torch.cat(
            [w * t for w, t in zip(concat_weights[1:], states[1:])], dim=1
        )

        # @TODO: projected_input should be masked (in my view
        # but nasbench-1shot do not apply any mask on it.
        out = concat_weights[0] * projected_input + cat_tensor
        return out


class NasBench101SharedOp(nn.Module):
    def __init__(self, C, C_out, primitives, op_mapping={}):
        super(NasBench101SharedOp, self).__init__()

        self.primitives = primitives
        self.op_mapping = op_mapping

        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = get_op(op_mapping.get(primitive, primitive))(
                C, C_out, 1, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C_out, affine=False))
            self.p_ops.append(op)

    def forward(self, inputs, op_idx):
        return self.p_ops[op_idx](inputs)


# class NasBench101Trainer(BaseTrainer):
#     NAME = "nasbench-101"

#     def __init__(self, controller, evaluator, rollout_type,
#                  epochs=100,
#                  schedule_cfg=None):
#         super(NasBench101Trainer, self).__init__(controller, evaluator, rollout_type, schedule_cfg)

#         self.search_space = controller.search_space

#     @classmethod
#     def supported_rollout_types(cls):
#         return ["nasbench-101", "compare"]

#     def train(self):
#         pass

#     def test(self):
#         pass

#     def derive(self, n, steps=None):
#         pass

#     def save(self, path):
#         pass

#     def load(self, path):
#         pass
