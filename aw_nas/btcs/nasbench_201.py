"""
NASBench-201 search space, rollout, embedder
"""

import os
import copy
import collections

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nas_201_api import NASBench201API as API

from aw_nas import utils
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.controller.base import BaseController
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.utils import DenseGraphSimpleOpEdgeFlow

VERTICES = 4


class NasBench201SearchSpace(SearchSpace):
    NAME = "nasbench-201"

    def __init__(self, load_nasbench=True):
        super(NasBench201SearchSpace, self).__init__()

        self.ops_choices = [
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3"
        ]
        self.ops_choice_to_idx = {choice: i for i, choice in enumerate(self.ops_choices)}

        self.load_nasbench = load_nasbench
        self.num_vertices = VERTICES
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices) # 5
        self.num_ops = self.num_vertices * (self.num_vertices - 1) // 2
        self.idx = np.tril_indices(self.num_vertices, k=1)

        if self.load_nasbench:
            self._init_nasbench()

    def __getstate__(self):
        state = super(NasBench201SearchSpace, self).__getstate__().copy()
        del state["api"]
        return state

    def __setstate__(self, state):
        super(NasBench201SearchSpace, self).__setstate__(state)
        if self.load_nasbench:
            self._init_nasbench()

    # ---- APIs ----
    def random_sample(self):
        return NasBench201Rollout(
            self.random_sample_arch(),
            search_space=self)

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        return self.matrix2str(arch)

    def rollout_from_genotype(self, genotype):
        return NasBench201Rollout(API.str2matrix(genotype), search_space=self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        # TODO
        #raise NotImplementedError()
        pass

    def distance(self, arch1, arch2):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201"]

    # ---- helpers ----
    def matrix2str(self, arch):
        node_strs = []
        for i_node in range(1, self.num_vertices):
            node_strs.append("|" + "|".join(["{}~{}".format(
                self.ops_choices[int(arch[i_node, i_input])], i_input)
                                             for i_input in range(0, i_node)]) + "|")
        return "+".join(node_strs)

    def _init_nasbench(self):
        # the arch -> performances dataset
        self.base_dir = os.path.join(utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-201")
        self.api = API(os.path.join(self.base_dir, "NAS-Bench-201-v1_0-e61699.pth"))

    def op_to_idx(self, ops):
        return [self.ops_choice_to_idx[op] for op in ops]

    def random_sample_arch(self):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        arch[np.tril_indices(self.num_vertices, k=-1)] = np.random.randint(
            low=0, high=self.num_op_choices, size=self.num_ops)
        return arch


class NasBench201Rollout(BaseRollout):
    NAME = "nasbench-201"

    def __init__(self, matrix, search_space):
        super(NasBench201Rollout, self).__init__()

        self.arch = matrix
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
        return "NasBench201Rollout(matrix={arch}, perf={perf})"\
            .format(arch=self.arch, perf=self.perf)


class NasBench201EvoController(BaseController):
    NAME = "nasbench-201-evo"

    def __init__(self, search_space, device, rollout_type="nasbench-201", mode="eval",
                 population_nums=100,
                 schedule_cfg=None):
        super(NasBench201EvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        # get the infinite iterator of the model matrix and ops
        self.mode = mode
        self.num_vertices = self.search_space.num_vertices
        self.cur_solution = self.search_space.random_sample_arch()
        self.population_nums = population_nums
        self.population = collections.OrderedDict()
        self.num_arch = len(self.search_space.api)
        population_ind = np.random.choice(
            np.arange(self.num_arch), size=self.population_nums, replace=False)
        for i in range(self.population_nums):
            arch_res = self.search_space.api.query_by_index(population_ind[i])
            accs = np.mean([res.eval_acc1es["ori-test@199"]
                            for res in arch_res.query("cifar10").values()]) / 100.
            self.population[arch_res.arch_str] = accs

    def reinit(self):
        population_ind = np.random.choice(
            np.arange(self.num_arch), size=self.population_nums, replace=False)
        for i in range(self.population_nums):
            arch_res = self.search_space.api.query_by_index(population_ind[i])
            accs = np.mean([res.eval_acc1es["ori-test@199"]
                            for res in arch_res.query("cifar10").values()]) / 100.
            self.population[arch_res.arch_str] = accs

    def set_init_population(self, rollout_list, perf_name):
        # clear the current population
        self.population = collections.OrderedDict()
        for r in rollout_list:
            self.population[r.genotype] = r.get_perf(perf_name)

    def sample(self, n, batch_size=None):
        assert batch_size is None
        new_archs = sorted(self.population.items(), key=lambda x: x[1], reverse=True)
        if self.mode == "eval":
            best_sets = []
            for n_r in range(n):
                best_sets.append(NasBench201Rollout(
                    self.search_space.api.str2matrix(new_archs[n_r][0]), self.search_space))
            return best_sets
        rollouts = []
        for n_r in range(n):
            try_times = 0
            while True:
                rand_ind = np.random.randint(0, self.search_space.idx[0].shape[0])
                neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
                arch_mat = self.search_space.api.str2matrix(new_archs[n_r][0])
                while neighbor_choice == arch_mat[self.search_space.idx[0][rand_ind],\
                         self.search_space.idx[1][rand_ind]]:
                    neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
                new_choice = copy.deepcopy(arch_mat)
                new_choice[self.search_space.idx[0][rand_ind], self.search_space.idx[1][rand_ind]]\
                    = neighbor_choice
                try_times += 1
                if self.search_space.genotype(new_choice) not in self.population.keys():
                    break
            rollouts.append(NasBench201Rollout(new_choice, self.search_space))
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201"]

    def step(self, rollouts, optimizer, perf_name):
        best_rollout = rollouts[0]
        for r in rollouts:
            if r.get_perf(perf_name) > best_rollout.get_perf(perf_name):
                best_rollout = r
        self.population.pop(list(self.population.keys())[0])
        self.population[best_rollout.genotype] = best_rollout.get_perf(perf_name)
        return 0

    # ---- APIs that is not necessary ----
    def set_mode(self, mode):
        self.mode = mode

    def set_device(self, device):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass



class NasBench201SAController(BaseController):
    NAME = "nasbench-201-sa"

    def __init__(self, search_space, device, rollout_type="nasbench-201", mode="eval",
                 temperature=1000, anneal_coeff=0.98,
                 schedule_cfg=None):
        super(NasBench201SAController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        # get the infinite iterator of the model matrix and ops
        self.num_vertices = self.search_space.num_vertices
        self.temperature = temperature
        self.anneal_coeff = anneal_coeff

        # random sample as the init arch
        self.cur_solution = self.search_space.random_sample_arch()
        self.cur_perf = None

    def reinit(self):
        # random sample as the init arch
        self.cur_solution = self.search_space.random_sample_arch()
        self.cur_perf = None

    def set_init_population(self, rollout_list, perf_name):
        # set the initialization to the best rollout in the list
        perf_list = [r.get_perf(perf_name) for r in rollout_list]
        best_rollout = rollout_list[np.argmax(perf_list)]
        self.cur_solution = best_rollout.arch
        self.cur_perf = best_rollout.get_perf(perf_name)
        self.logger.info("Set the initialization rollout: {}; perf: {}".format(
            best_rollout, self.cur_perf))
        
    def sample(self, n, batch_size=None):
        assert batch_size is None

        if self.mode == "eval":
            return [NasBench201Rollout(self.cur_solution, self.search_space)] * n

        rollouts = []
        for n_r in range(n):
            rand_ind = np.random.randint(0, self.search_space.idx[0].shape[0])
            neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
            while neighbor_choice == self.cur_solution[self.search_space.idx[0][rand_ind],\
                     self.search_space.idx[1][rand_ind]]:
                neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
            new_choice = copy.deepcopy(self.cur_solution)
            new_choice[self.search_space.idx[0][rand_ind], self.search_space.idx[1][rand_ind]] = neighbor_choice
            rollouts.append(NasBench201Rollout(new_choice, self.search_space))
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201"]

    def step(self, rollouts, optimizer, perf_name):
        ind = np.argmax([r.get_perf(perf_name) for r in rollouts])
        rollout = rollouts[ind]
        new_perf = rollout.get_perf(perf_name)

        prob = np.random.rand()
        if self.cur_perf is None or self.cur_perf < new_perf:
            self.cur_perf = new_perf
            self.cur_solution = rollouts[0].arch
        elif np.exp(-(self.cur_perf - new_perf) / self.temperature) > prob:
            self.cur_perf = new_perf
            self.cur_solution = rollouts[0].arch
        self.temperature *= self.anneal_coeff
        return 0

    # ---- APIs that is not necessary ----
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


# ---- embedders for NASBench-201 ----
class NasBench201_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "nb201-lstm"

    def __init__(self, search_space, num_hid=100, emb_hid=100, num_layers=1,
                 use_mean=False, use_hid=False,
                 schedule_cfg=None):
        super(NasBench201_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.op_emb = nn.Embedding(self.search_space.num_op_choices, self.emb_hid)

        self.rnn = nn.LSTM(input_size=self.emb_hid,
                           hidden_size=self.num_hid, num_layers=self.num_layers,
                           batch_first=True)

        self.out_dim = num_hid
        self._tril_indices = np.tril_indices(self.search_space.num_vertices, k=-1)

    def forward(self, archs):
        x = [arch[self._tril_indices] for arch in archs]
        embs = self.op_emb(torch.LongTensor(x).to(self.op_emb.weight.device))
        out, (h_n, _) = self.rnn(embs)

        if self.use_hid:
            y = h_n[0]
        else:
            if self.use_mean:
                y = torch.mean(out, dim=1)
            else:
                # return final output
                y = out[:, -1, :]
        return y


class NasBench201_SimpleSeqEmbedder(ArchEmbedder):
    NAME = "nb201-seq"

    def __init__(self, search_space, schedule_cfg=None):
        super(NasBench201_SimpleSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.out_dim = self.search_space.num_ops
        self.num_node = self.search_space.num_vertices

        self._tril_indices = np.tril_indices(self.num_node, k=-1)
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        x = [arch[self._tril_indices] for arch in archs]
        return self._placeholder_tensor.new(x)


class NasBench201FlowArchEmbedder(ArchEmbedder):
    NAME = "nb201-flow"

    def __init__(self, search_space, op_embedding_dim=48,
                 node_embedding_dim=48, hid_dim=96, gcn_out_dims=[128, 128],
                 share_op_attention=False,
                 gcn_kwargs=None,
                 use_bn=False,
                 use_final_only=False,
                 share_self_op_emb=False,
                 dropout=0., schedule_cfg=None):
        super(NasBench201FlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.share_op_attention = share_op_attention
        self.share_self_op_emb = share_self_op_emb
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind

        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim),
            requires_grad=False)

        # the last embedding is the output op emb
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(self.op_embedding_dim).normal_())
        else:
            self.self_op_emb = None

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
            self.gcns.append(DenseGraphSimpleOpEdgeFlow(
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
        adjs = self.op_emb.weight.new(archs).long()
        op_embs = self.op_emb(adjs) # (batch_size, vertices, vertices, op_emb_dim)
        b_size = op_embs.shape[0]
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
        adjs, x, op_embs = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_embs, self_op_emb=self.self_op_emb)
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
            y = torch.mean(y, dim=1) # average across nodes (bs, god)
        return y


class NasBench201Evaluator(BaseEvaluator):
    NAME = "nasbench-201"

    def __init__(self, dataset, weights_manager, objective, rollout_type="nasbench-201",
                 schedule_cfg=None):
        super(NasBench201Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type)

    @classmethod
    def supported_data_types(cls):
        # cifar10
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201", "compare"]

    def suggested_controller_steps_per_epoch(self):
        return None

    def suggested_evaluator_steps_per_epoch(self):
        return None

    def evaluate_rollouts(self, rollouts, is_training=False, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        if self.rollout_type == "compare":
            eval_rollouts = sum([[r.rollout_1, r.rollout_2] for r in rollouts], [])
        else:
            eval_rollouts = rollouts

        for rollout in eval_rollouts:
            query_idx = rollout.search_space.api.query_index_by_arch(rollout.genotype)
            query_res = rollout.search_space.api.query_by_index(query_idx)
            rollout_perf = np.mean([res.eval_acc1es['ori-test@199'] for res in query_res.query('cifar10').values()]) / 100.
            # TODO: could use other performance, this functionality is not compatible with objective
            # multiple fidelity too
            rollout.set_perf(rollout_perf)

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


