"""
NASBench-101 search space, rollout, controller, evaluator.
During the development,
referred https://github.com/automl/nas_benchmarks/blob/master/tabular_benchmarks/nas_cifar10.py
"""

import os
import random
import collections

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nasbench import api
from nasbench.lib import graph_util

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


class NasBench101SearchSpace(SearchSpace):
    NAME = "nasbench-101"

    def __init__(self, multi_fidelity=False, load_nasbench=True):
        self.ops_choices = [
            "conv1x1-bn-relu",
            "conv3x3-bn-relu",
            "maxpool3x3"
        ]
        self.ops_choice_to_idx = {choice: i for i, choice in enumerate(self.ops_choices)}

        self.multi_fidelity = multi_fidelity
        self.load_nasbench = load_nasbench
        self.num_vertices = VERTICES
        self.max_edges = MAX_EDGES
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices) # 3
        self.num_ops = self.num_vertices - 2 # 5
        self.idx = np.triu_indices(self.num_vertices, k=1)

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

class NasBench101ArchEmbedder(ArchEmbedder):
    NAME = "nb101-gcn"

    def __init__(self, search_space, embedding_dim=48, hid_dim=48, gcn_out_dims=[128, 128],
                 gcn_kwargs=None, dropout=0., schedule_cfg=None):
        super(NasBench101ArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.embedding_dim = embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.verticies = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices

        self.input_op_emb = nn.Embedding(1, self.embedding_dim)
        # zero is ok
        self.output_op_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
                                          requires_grad=False)

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
        node_embs = self.op_emb(op_inds) # (batch_size, vertices - 2, emb_dim)
        b_size = node_embs.shape[0]
        node_embs = torch.cat((self.input_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                               node_embs, self.output_op_emb.unsqueeze(0).repeat([b_size, 1, 1])),
                              dim=1)
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        adjs, x = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        y = y[:, 1:, :] # do not keep the inputs node embedding
        y = torch.mean(y, dim=1) # average across nodes (bs, god)
        return y


class NasBench101FlowArchEmbedder(ArchEmbedder):
    NAME = "nb101-flow"

    def __init__(self, search_space, op_embedding_dim=48,
                 node_embedding_dim=48, hid_dim=96, gcn_out_dims=[128, 128],
                 share_op_attention=False,
                 other_node_zero=False, gcn_kwargs=None,
                 dropout=0., schedule_cfg=None):
        super(NasBench101FlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.share_op_attention = share_op_attention
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices

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
        self.input_op_emb = nn.Parameter(torch.zeros(1, self.op_embedding_dim), requires_grad=False)
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert len(np.unique(self.gcn_out_dims)) == 1, \
                "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphFlow(
                in_dim, dim, self.op_embedding_dim if not self.share_op_attention else dim,
                has_attention=not self.share_op_attention, **(gcn_kwargs or {})))
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.new([arch[1] for arch in archs]).long()
        op_embs = self.op_emb(op_inds) # (batch_size, vertices - 2, op_emb_dim)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
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
        return adjs, x, op_embs

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        adjs, x, op_emb = self.embed_and_transform_arch(archs)
        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_emb)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        y = y[:, 1:, :] # do not keep the inputs node embedding
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
