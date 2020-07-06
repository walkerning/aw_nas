"""
NASBench-101 search space, rollout, controller, evaluator.
During the development,
referred https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
"""

import os
import re
import random
import collections

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nasbench import api
from nasbench.lib import graph_util, config

from aw_nas import utils
from aw_nas.utils.exception import expect
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.controller.base import BaseController
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.rollout.compare import CompareRollout
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.utils import DenseGraphConvolution, DenseGraphFlow

VERTICES = 7
MAX_EDGES = 9
_nasbench_cfg = config.build_config()

def _literal_np_array(arr):
    if arr is None:
        return None
    return "np.array({})".format(np.array2string(arr, separator=",").replace("\n", " "))

class _ModelSpec(api.ModelSpec):
    def __repr__(self):
        return "_ModelSpec({}, {}; pruned_matrix={}, pruned_ops={})".format(
            _literal_np_array(self.original_matrix), self.original_ops,
            _literal_np_array(self.matrix), self.ops)

    def hash_spec(self, *args, **kwargs):
        return super(_ModelSpec, self).hash_spec(_nasbench_cfg["available_ops"])

class NasBench101SearchSpace(SearchSpace):
    NAME = "nasbench-101"

    def __init__(self, multi_fidelity=False, load_nasbench=True,
                 compare_reduced=True, compare_use_hash=False, validate_spec=True):
        super(NasBench101SearchSpace, self).__init__()

        self.ops_choices = [
            "conv1x1-bn-relu",
            "conv3x3-bn-relu",
            "maxpool3x3",
            "none",
        ]
        # operations: "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"
        self.ops_choice_to_idx = {choice: i for i, choice in enumerate(self.ops_choices)}

        self.multi_fidelity = multi_fidelity
        self.load_nasbench = load_nasbench
        self.compare_reduced = compare_reduced
        self.compare_use_hash = compare_use_hash

        self.num_vertices = VERTICES
        self.max_edges = MAX_EDGES
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices) # 3 + 1 (none)
        self.num_ops = self.num_vertices - 2 # 5
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
            padded_adj = np.concatenate((adj[:-1],
                                         np.zeros((VERTICES - num_v, num_v), dtype=np.int8),
                                         adj[-1:]))
            padded_adj = np.concatenate((padded_adj[:, :-1],
                                         np.zeros((VERTICES, VERTICES - num_v)),
                                         padded_adj[:, -1:]), axis=1)
            padded_ops = ops + [3] * (7 - num_v)
            adj, ops = padded_adj, padded_ops
        return (adj, ops)

    def _random_sample_ori(self):
        while 1:
            matrix = np.random.choice([0, 1], size=(self.num_vertices, self.num_vertices))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(self.ops_choices[:-1], size=(self.num_vertices)).tolist()
            ops[0] = "input"
            ops[-1] = "output"
            spec = _ModelSpec(matrix=matrix, ops=ops)
            if self.validate_spec and not self.nasbench.is_valid(spec):
                continue
            return NasBench101Rollout(
                spec.original_matrix, ops=self.op_to_idx(spec.original_ops), search_space=self)

    def _random_sample_me(self):
        while 1:
            splits = np.array(
                sorted([0] + list(np.random.randint(
                    0, self.max_edges + 1,
                    size=self.num_possible_edges - 1)) + [self.max_edges]))
            edges = np.minimum(splits[1:] - splits[:-1], 1)
            matrix = self.edges_to_matrix(edges)
            ops = np.random.randint(0, self.num_op_choices - 1, size=self.num_ops)
            rollout = NasBench101Rollout(
                matrix, ops, search_space=self)
            try:
                self.nasbench._check_spec(rollout.genotype)
            except api.OutOfDomainError:
                # ignore out-of-domain archs (disconnected)
                continue
            else:
                return rollout

    # optional API
    def genotype_from_str(self, genotype_str):
        return eval(re.search("(_ModelSpec\(.+);", genotype_str).group(1) + ")")

    # ---- APIs ----
    def random_sample(self):
        return self._random_sample_ori()

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        matrix, ops = arch
        return self.construct_modelspec(edges=None, matrix=matrix, ops=ops)

    def rollout_from_genotype(self, genotype):
        return NasBench101Rollout(genotype.original_matrix,
                                  ops=self.op_to_idx(genotype.original_ops),
                                  search_space=self)

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
            yield [NasBench101Rollout(
                list_[r_ind]["module_adjacency"],
                self.op_to_idx(list_[r_ind]["module_operations"]),
                search_space=self)
                   for r_ind in indexes[ind:end_ind]]
            ind = end_ind

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-101"]

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

    def __eq__(self, other):
        if self.search_space.compare_reduced:
            if self.search_space.compare_use_hash:
                # compare using hash, isomorphic archs would be equal
                return self.genotype.hash_spec() == \
                    other.genotype.hash_spec()
            else:
                # compared using reduced archs
                return (np.array(self.genotype.matrix).tolist() == \
                        np.array(other.genotype.matrix).tolist()) \
                        and list(self.genotype.ops) == list(other.genotype.ops)

        # compared using original/non-reduced archs, might be wrong
        return (np.array(other.arch[0]).tolist(), list(other.arch[1])) \
            == (np.array(self.arch[0]).tolist(), list(self.arch[1]))

class NasBench101CompareController(BaseController):
    NAME = "nasbench-101-compare"

    def __init__(self, search_space, device, rollout_type="compare", mode="eval",
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
    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer, perf_name):
        return 0.

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101Controller(BaseController):
    NAME = "nasbench-101"

    def __init__(self, search_space, device, rollout_type="nasbench-101", mode="eval",
                 shuffle_indexes=True, avoid_repeat=False,
                 schedule_cfg=None):
        super(NasBench101Controller, self).__init__(search_space, rollout_type, mode, schedule_cfg)

        self.shuffle_indexes = shuffle_indexes
        self.avoid_repeat = avoid_repeat

        # get the infinite iterator of the model matrix and ops
        self.fixed_statistics = list(self.search_space.nasbench.fixed_statistics.values())
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
            self.logger.info("Return the best {} rollouts in the population".format(n))
            sampled_rollouts = []
            n_evaled = len(self.gt_rollouts)
            if n_evaled < n:
                sampled_rollouts = [self.search_space.random_sample() for _ in range(n - n_evaled)]
            best_inds = np.argpartition(self.gt_scores, -n)[-n:]
            return sampled_rollouts + [self.gt_rollouts[ind] for ind in best_inds]

        if self.avoid_repeat:
            assert batch_size is None
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
        self.gt_scores = self.gt_scores + [r.get_perf(perf_name) for r in rollouts]
        if len(self.gt_rollouts) >= num_rollouts_to_keep:
            best_inds = np.argpartition(
                self.gt_scores, -num_rollouts_to_keep)[-num_rollouts_to_keep:]
            self.gt_rollouts = [self.gt_rollouts[ind] for ind in best_inds]
            self.gt_scores = [self.gt_scores[ind] for ind in best_inds]
        return 0.

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        return collections.OrderedDict()

    def save(self, path):
        pass

    def load(self, path):
        pass


class NasBench101EvoController(BaseController):
    NAME = "nasbench-101-evo"

    def __init__(self, search_space, device, rollout_type="nasbench-101", mode="eval",
                 population_nums=100, parent_pool_size=10, mutation_prob=1.0,
                 schedule_cfg=None):
        super(NasBench101EvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

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
        assert batch_size is None
        if self.mode == "eval":
            # Return n archs seen(self.gt_rollouts) with best rewards
            self.logger.info("Return the best {} rollouts in the population".format(n))
            best_inds = np.argpartition(self.gt_scores, -n)[-n:]
            return [self.gt_rollouts[ind] for ind in best_inds]

        if len(self.population) < self.population_nums:
            self.logger.info("Population not full, random sample {}".format(n))
            return [self.search_space.random_sample() for _ in range(n)]

        population_items = list(self.population.items())
        choices = np.random.choice(np.arange(len(self.population)),
                                   size=self.parent_pool_size, replace=False)
        selected_pop = [population_items[i] for i in choices]
        new_archs = sorted(selected_pop, key=lambda x: x[1], reverse=True)
        rollouts = []
        ss = self.search_space
        cur_matrix, cur_ops = new_archs[0][0].original_matrix, new_archs[0][0].original_ops
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
                    available = [o for o in ss.nasbench.config['available_ops'] if o != new_ops[ind]]
                    new_ops[ind] = np.random.choice(available)

            newspec = _ModelSpec(new_matrix, new_ops)
            if ss.nasbench.is_valid(newspec):
                rollouts.append(NasBench101Rollout(
                    new_matrix,
                    ss.op_to_idx(cur_ops),
                    search_space=self.search_space
                ))
                return rollouts

    def step(self, rollouts, optimizer, perf_name):
        # best_rollout = rollouts[0]
        # for r in rollouts:
        #     if r.get_perf(perf_name) > best_rollout.get_perf(perf_name):
        #         best_rollout = r
        for best_rollout in rollouts:
            self.population[best_rollout.genotype] = best_rollout.get_perf(perf_name)
        if len(self.population) > self.population_nums:
            for key in list(self.population.keys())[:len(self.population) - self.population_nums]:
                self.population.pop(key)

        # Save the best rollouts
        num_rollouts_to_keep = 200
        self.gt_rollouts = self.gt_rollouts + rollouts
        self.gt_scores = self.gt_scores + [r.get_perf(perf_name) for r in rollouts]
        if len(self.gt_rollouts) >= num_rollouts_to_keep:
            best_inds = np.argpartition(
                self.gt_scores, -num_rollouts_to_keep)[-num_rollouts_to_keep:]
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

    def __init__(self, search_space, device, rollout_type="nasbench-101", mode="eval",
                 temperature=1000, anneal_coeff=0.95, mutation_edges_prob=0.5,
                 schedule_cfg=None):
        super(NasBench101SAController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

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
        self.logger.info("Set the initialization rollout: {}; perf: {}".format(
            best_rollout, self.cur_perf))

    def sample(self, n, batch_size=None):
        if self.mode == "eval":
            # return the current rollout
            return [NasBench101Rollout(
                *self.cur_solution, search_space=self.search_space)] * n

        assert batch_size is None
        rollouts = []
        cur_matrix, cur_ops = self.cur_solution
        ss = self.search_space
        for n_r in range(n):
            if np.random.rand() < self.mutation_edges_prob:
                while 1:
                    edge_ind = np.random.randint(0, ss.num_possible_edges, size=1)
                    while graph_util.num_edges(cur_matrix) == ss.max_edges and \
                          cur_matrix[ss.idx[0][edge_ind], ss.idx[1][edge_ind]] == 0:
                        edge_ind = np.random.randint(0, ss.num_possible_edges, size=1)
                    new_matrix = cur_matrix.copy()
                    new_matrix[ss.idx[0][edge_ind], ss.idx[1][edge_ind]] \
                        = 1 - cur_matrix[ss.idx[0][edge_ind], ss.idx[1][edge_ind]]
                    new_rollout = NasBench101Rollout(new_matrix, cur_ops,
                                                     search_space=self.search_space)
                    try:
                        ss.nasbench._check_spec(new_rollout.genotype)
                    except api.OutOfDomainError:
                        # ignore out-of-domain archs (disconnected)
                        continue
                    else:
                        cur_matrix = new_matrix
                        break
            else:
                ops_ind = np.random.randint(0, ss.num_ops, size=1)[0]
                new_ops = np.random.randint(0, ss.num_op_choices - 1, size=1)[0]
                while new_ops == cur_ops[ops_ind]:
                    new_ops = np.random.randint(0, ss.num_op_choices - 1, size=1)[0]
                cur_ops[ops_ind] = new_ops
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

    def __init__(self, dataset, weights_manager, objective, rollout_type="nasbench-101",
                 use_epoch=108, use_mean_valid_as_reward=False,
                 schedule_cfg=None):
        super(NasBench101Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type)

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

    def evaluate_rollouts(self, rollouts, is_training=False, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        if self.rollout_type == "compare":
            eval_rollouts = sum([[r.rollout_1, r.rollout_2] for r in rollouts], [])
        else:
            eval_rollouts = rollouts

        for rollout in eval_rollouts:
            if not self.use_mean_valid_as_reward:
                query_res = rollout.search_space.nasbench.query(rollout.genotype)
                # could use other performance, this functionality is not compatible with objective
                rollout.set_perf(query_res["validation_accuracy"])

            # use mean for test acc
            # can use other fidelity too
            res = rollout.search_space.nasbench.get_metrics_from_spec(
                rollout.genotype)[1][self.use_epoch]
            mean_valid_acc = np.mean([s_res["final_validation_accuracy"] for s_res in res])
            if self.use_mean_valid_as_reward:
                rollout.set_perf(mean_valid_acc)
            rollout.set_perf(mean_valid_acc, name="mean_valid_acc")
            mean_test_acc = np.mean([s_res["final_test_accuracy"] for s_res in res])
            rollout.set_perf(mean_test_acc, name="mean_test_acc")

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


# ---- embedders for NASBench-101 ----
# TODO: the multi stage trick could apply for all the embedders
class NasBench101_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "nb101-lstm"

    def __init__(self, search_space, num_hid=100, emb_hid=100, num_layers=1,
                 use_mean=False, use_hid=False,
                 schedule_cfg=None):
        super(NasBench101_LSTMSeqEmbedder, self).__init__(schedule_cfg)
                
        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid
        
        self.op_emb = nn.Embedding(self.search_space.num_op_choices, self.emb_hid)
        self.conn_emb = nn.Embedding(2, self.emb_hid)

        self.rnn = nn.LSTM(input_size=self.emb_hid,
                           hidden_size=self.num_hid, num_layers=self.num_layers,
                           batch_first=True)

        self.out_dim = num_hid
        self._triu_indices = np.triu_indices(VERTICES, k=1)

    def forward(self, archs):
        x_1 = np.array([arch[0][self._triu_indices] for arch in archs])
        x_2 = np.array([arch[1] for arch in archs])

        conn_embs = self.conn_emb(torch.LongTensor(x_1).to(self.op_emb.weight.device))
        op_embs = self.op_emb(torch.LongTensor(x_2).to(self.op_emb.weight.device))
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
                [np.array([arch[0].reshape(-1) for arch in archs]),
                 np.array([arch[1] for arch in archs])], axis=-1)
        else:
            triu_indices = np.triu_indices(VERTICES, k=1)
            x_1 = np.array([arch[0][triu_indices] for arch in archs])
            x_2 = np.array([arch[1] for arch in archs])
            x = np.concatenate([x_1, x_2], axis=-1)
        return self._placeholder_tensor.new(x)

class NasBench101ArchEmbedder(ArchEmbedder):
    NAME = "nb101-gcn"

    def __init__(self, search_space, embedding_dim=48, hid_dim=48, gcn_out_dims=[128, 128],
                 gcn_kwargs=None, dropout=0.,
                 use_global_node=False,
                 use_final_only=False,
                 schedule_cfg=None):
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
            self.global_op_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)))

        self.op_emb = nn.Embedding(self.num_op_choices, self.embedding_dim)
        self.x_hidden = nn.Linear(self.embedding_dim, self.hid_dim)

        # init graph convolutions
        self.gcns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphConvolution(in_dim, dim, **(gcn_kwargs or {})))
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.weight.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.weight.new([arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat((
                tmp_ones,
                (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                tmp_ones), dim=2)
            adjs = torch.cat(
                (torch.cat((adjs, tmp_cat), dim=1),
                 torch.zeros((adjs.shape[0], self.vertices + 1, 1), device=adjs.device)), dim=2)
        node_embs = self.op_emb(op_inds) # (batch_size, vertices - 2, emb_dim)
        b_size = node_embs.shape[0]
        if self.use_global_node:
            node_embs = torch.cat((self.input_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                                   node_embs,
                                   self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                                   self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1])),
                                  dim=1)
        else:
            node_embs = torch.cat((self.input_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                                   node_embs,
                                   self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1])),
                                  dim=1)
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
            y = y[:, 1:, :] # do not keep the inputs node embedding
            # throw away padded info here
            y = torch.cat((
                y[:, :-1, :] * (op_inds != self.none_op_ind)[:, :, None].to(torch.float32),
                y[:, -1:, :]), dim=1)
            y = torch.mean(y, dim=1) # average across nodes (bs, god)
        return y


class NasBench101FlowArchEmbedder(ArchEmbedder):
    NAME = "nb101-flow"

    def __init__(self, search_space, op_embedding_dim=48,
                 node_embedding_dim=48, hid_dim=96, gcn_out_dims=[128, 128],
                 share_op_attention=False,
                 other_node_zero=False, gcn_kwargs=None,
                 use_bn=False,
                 use_global_node=False,
                 use_final_only=False,
                 input_op_emb_trainable=False,
                 dropout=0., schedule_cfg=None):
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
            torch.zeros(1, self.node_embedding_dim),
            requires_grad=not other_node_zero)
        # self.middle_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)
        # # zero is ok
        # self.output_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(torch.zeros(1, self.op_embedding_dim),
                                         requires_grad=self.input_op_emb_trainable)
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert len(np.unique(self.gcn_out_dims)) == 1, \
                "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphFlow(
                in_dim, dim, self.op_embedding_dim if not self.share_op_attention else dim,
                has_attention=not self.share_op_attention, **(gcn_kwargs or {})))
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
            tmp_cat = torch.cat((
                tmp_ones,
                (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                tmp_ones), dim=2)
            adjs = torch.cat(
                (torch.cat((adjs, tmp_cat), dim=1),
                 torch.zeros((adjs.shape[0], self.vertices, 1), device=adjs.device)), dim=2)
        op_embs = self.op_emb(op_inds) # (batch_size, vertices - 2, op_emb_dim)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                 op_embs,
                 self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                 self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1])),
                dim=1)
        else:
            op_embs = torch.cat(
                (self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                 op_embs,
                 self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1])),
                dim=1)
        node_embs = torch.cat(
            (self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
             self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1])),
            dim=1)
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
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :] # do not keep the inputs node embedding

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat((
                    y[:, :-2, :] * (op_inds != self.none_op_ind)[:, :, None].to(torch.float32),
                    y[:, -2:, :]), dim=1)
            else:
                y = torch.cat((
                    y[:, :-1, :] * (op_inds != self.none_op_ind)[:, :, None].to(torch.float32),
                    y[:, -1:, :]), dim=1)

            y = torch.mean(y, dim=1) # average across nodes (bs, god)
        return y


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
