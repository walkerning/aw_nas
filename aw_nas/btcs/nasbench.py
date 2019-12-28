"""
NASBench-101 search space, rollout, controller, evaluator.
During the development,
referred https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
"""

import os
import random
import collections

import numpy as np

from nasbench import api
from nasbench.lib import graph_util

from aw_nas import utils
from aw_nas.utils.exception import expect
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.controller.base import BaseController
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.rollout.compare import CompareRollout

VERTICES = 7
MAX_EDGES = 9


class NasBench101SearchSpace(SearchSpace):
    NAME = "nasbench-101"

    def __init__(self, multi_fidelity=False):
        self.ops_choices = [
            "conv1x1-bn-relu",
            "conv3x3-bn-relu",
            "maxpool3x3"
        ]
        self.ops_choice_to_idx = {choice: i for i, choice in enumerate(self.ops_choices)}

        self.multi_fidelity = multi_fidelity
        self.num_vertices = VERTICES
        self.max_edges = MAX_EDGES
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices) # 3
        self.num_ops = self.num_vertices - 2 # 5
        self.idx = np.triu_indices(self.num_vertices, k=1)

        self._init_nasbench()

    def __getstate__(self):
        state = super(NasBench101SearchSpace, self).__getstate__().copy()
        del state["nasbench"]
        return state

    def __setstate__(self, state):
        super(NasBench101SearchSpace, self).__setstate__(state)
        self._init_nasbench()

    # ---- APIs ----
    def random_sample(self):
        return NasBench101Rollout(
            *self.random_sample_arch(),
            search_space=self)

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        matrix, ops = arch
        return self.construct_modelspec(edges=None, matrix=matrix, ops=ops)

    def rollout_from_genotype(self, genotype):
        # TODO
        pass

    def plot_arch(self, genotypes, filename, label, **kwargs):
        # TODO
        pass

    def distance(self, arch1, arch2):
        pass

    # ---- helpers ----
    def _init_nasbench(self):
        # the arch -> performances dataset
        self.base_dir = os.path.join(utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-101")
        if self.multi_fidelity:
            self.nasbench = api.NASBench(os.path.join(self.base_dir, "nasbench_full.tfrecord"))
        else:
            self.nasbench = api.NASBench(os.path.join(self.base_dir, "nasbench_only108.tfrecord"))

    def edges_to_matrix(self, edges):
        matrix = np.zeros([self.num_vertices, self.num_vertices], dtype=np.int8)
        matrix[self.idx] = edges
        return matrix

    def op_to_idx(self, ops):
        return [self.ops_choice_to_idx[op] for op in ops if op not in {"input", "output"}]

    def matrix_to_edges(self, matrix):
        return matrix[self.idx]

    def construct_modelspec(self, edges, matrix, ops):
        if matrix is None:
            assert edges is not None
            matrix = self.edges_to_matrix(edges)

        expect(graph_util.num_edges(matrix) <= self.max_edges,
               "number of edges could not exceed {}".format(self.max_edges))

        labeling = [self.ops_choices[op_ind] for op_ind in ops]
        labeling = ["input"] + list(labeling) + ["output"]
        model_spec = api.ModelSpec(matrix, labeling)
        return model_spec

    def random_sample_arch(self):
        # FIXME: not uniform, and could be illegal,
        #   if there is not edge from the INPUT or no edge to the OUTPUT,
        #   maybe just check and reject
        splits = np.array(
            sorted([0] + list(np.random.randint(
                0, self.max_edges + 1,
                size=self.num_possible_edges - 1)) + [self.max_edges]))
        edges = np.minimum(splits[1:] - splits[:-1], 1)
        matrix = self.edges_to_matrix(edges)
        ops = np.random.randint(0, self.num_op_choices, size=self.num_ops)
        return matrix, ops


class NasBench101Rollout(BaseRollout):
    NAME = "nasbench-101"

    def __init__(self, matrix, ops, search_space):
        self.arch = (matrix, ops)
        self.search_space = search_space
        self.perf = collections.OrderedDict()
        self._genotype = None

    def set_candidate_net(self, c_net):
        raise Exception("Should not be called")

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(
            self.genotype, filename,
            label=label, edge_labels=edge_labels)

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

    def __repr__(self):
        return "NasBench101Rollout(matrix={arch}, perf={perf})"\
            .format(arch=self.arch, perf=self.perf)


class NasBench101CompareController(BaseController):
    NAME = "nasbench-101-compare"

    def __init__(self, search_space, rollout_type="compare", mode="eval",
                 shuffle_indexes=True,
                 schedule_cfg=None):
        super(NasBench101CompareController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        self.shuffle_indexes = shuffle_indexes

        # get the infinite iterator of the model matrix and ops
        self.fixed_statistics = list(self.search_space.nasbench.fixed_statistics.values())
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
                search_space=self.search_space
            )
            if self.comp_indexes[self.cur_comp_ind] != self.indexes[self.cur_ind]:
                fixed_stat_2 = self.fixed_statistics[self.comp_indexes[self.cur_comp_ind]]
                rollout_2 = NasBench101Rollout(
                    fixed_stat_2["module_adjacency"],
                    self.search_space.op_to_idx(fixed_stat_2["module_operations"]),
                    search_space=self.search_space
                )
                rollouts.append(CompareRollout(rollout_1=rollout_1, rollout_2=rollout_2))
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
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101Controller(BaseController):
    NAME = "nasbench-101"

    def __init__(self, search_space, rollout_type="nasbench-101", mode="eval",
                 shuffle_indexes=True,
                 schedule_cfg=None):
        super(NasBench101Controller, self).__init__(search_space, rollout_type, mode, schedule_cfg)

        self.shuffle_indexes = shuffle_indexes

        # get the infinite iterator of the model matrix and ops
        self.fixed_statistics = list(self.search_space.nasbench.fixed_statistics.values())
        self.num_data = len(self.fixed_statistics)
        self.indexes = list(np.arange(self.num_data))
        self.cur_ind = 0

    def sample(self, n, batch_size=None):
        assert batch_size is None
        rollouts = []
        n_r = 0
        while n_r < n:
            fixed_stat = self.fixed_statistics[self.indexes[self.cur_ind]]
            rollouts.append(NasBench101Rollout(
                fixed_stat["module_adjacency"],
                self.search_space.op_to_idx(fixed_stat["module_operations"]),
                search_space=self.search_space
            ))
            self.cur_ind += 1
            n_r += 1
            if self.cur_ind >= self.num_data:
                self.logger.info("One epoch end")
                self.cur_ind = 0
                if self.shuffle_indexes:
                    random.shuffle(self.indexes)
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

    # ---- APIs that is not necessary ----
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101Evaluator(BaseEvaluator):
    NAME = "nasbench-101"

    def __init__(self, dataset, weights_manager, objective, rollout_type="nasbench-101",
                 schedule_cfg=None):
        super(NasBench101Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type)

    @classmethod
    def supported_data_types(cls):
        # cifar10
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101", "compare"]

    def suggested_controller_steps_per_epoch(self):
        return 0

    def suggested_evaluator_steps_per_epoch(self):
        return None

    def evaluate_rollouts(self, rollouts, is_training=False, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        if self.rollout_type == "compare":
            eval_rollouts = sum([[r.rollout_1, r.rollout_2] for r in rollouts], [])
        else:
            eval_rollouts = rollouts

        for rollout in eval_rollouts:
            query_res = rollout.search_space.nasbench.query(rollout.genotype)
            # TODO: could use other performance, this functionality is not compatible with objective
            # multiple fidelity too
            rollout.set_perf(query_res["validation_accuracy"])

        if self.rollout_type == "compare":
            num_r = len(rollouts)
            for i_rollout in range(num_r):
                diff = eval_rollouts[2 * i_rollout + 1].perf["reward"] - \
                       eval_rollouts[2 * i_rollout].perf["reward"]
                better = diff > 0
                rollouts[i_rollout].set_perfs(collections.OrderedDict(
                    [
                        ("compare_result", better),
                        ("diff", diff),
                    ]))
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
