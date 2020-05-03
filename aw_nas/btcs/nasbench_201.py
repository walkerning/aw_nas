"""
NASBench-201 search space, rollout, embedder
"""

import os
import re
import six
import copy
import random
import collections
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nas_201_api import NASBench201API as API

from aw_nas import utils, ops
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.controller.base import BaseController
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.utils import DenseGraphSimpleOpEdgeFlow, data_parallel, use_params
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.final.base import FinalModel


class NasBench201SearchSpace(SearchSpace):
    NAME = "nasbench-201"

    def __init__(self, num_layers=17, vertices=4, load_nasbench=True, ops_choices=(
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3")
            ):
        super(NasBench201SearchSpace, self).__init__()

        self.ops_choices = ops_choices
        self.ops_choice_to_idx = {choice: i for i, choice in enumerate(self.ops_choices)}

        self.load_nasbench = load_nasbench
        self.num_vertices = vertices
        self.num_layers = num_layers
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices) # 5
        self.num_ops = self.num_vertices * (self.num_vertices - 1) // 2
        self.idx = np.tril_indices(self.num_vertices, k=-1)

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

    def plot_arch(self, genotypes, filename, label, plot_format="pdf", **kwargs):
        matrix = self.str2matrix(genotypes)

        from graphviz import Digraph
        graph = Digraph(
            format=plot_format,
            # https://stackoverflow.com/questions/4714262/graphviz-dot-captions
            body=["label=\"{l}\"".format(l=label),
                  "labelloc=top", "labeljust=left"],
            edge_attr=dict(fontsize="20", fontname="times"),
            node_attr=dict(style="filled", shape="rect",
                           align="center", fontsize="20",
                           height="0.5", width="0.5",
                           penwidth="2", fontname="times"),
            engine="dot")
        graph.body.extend(["rankdir=LR"])
        graph.node(str(0), fillcolor="darkseagreen2")
        graph.node(str(self.num_vertices - 1), fillcolor="palegoldenrod")
        [graph.node(str(i), fillcolor="lightblue") for i in range(1, self.num_vertices-1)]

        for to_, from_ in zip(*self.idx):
            op_name = self.ops_choices[int(matrix[to_, from_])]
            if op_name == "none":
                continue
            graph.edge(str(from_), str(to_), label=op_name, fillcolor="gray")

        graph.render(filename, view=False)
        fnames = []
        fnames.append(("cell", filename + ".{}".format(plot_format)))
        return fnames

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

    def str2matrix(self, str_):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        split_str = str_.split("+")
        for ind, s in enumerate(split_str):
            geno = [name for name in s.split("|") if name != ""]
            for g in geno:
                name, conn = g.split("~")
                to_ = ind + 1
                from_ = int(conn)
                arch[to_][from_] = self.ops_choices.index(name)
        return arch

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

    def batch_rollouts(self, batch_size, shuffle=True, max_num=None):
        len_ = ori_len_ = len(self.api)
        if max_num is not None:
            len_ = min(max_num, len_)
        indexes = np.arange(ori_len_)
        np.random.shuffle(indexes)
        ind = 0
        while ind < len_:
            end_ind = min(len_, ind + batch_size)
            yield [NasBench201Rollout(matrix=self.api.str2matrix(self.api.arch(r_ind)),
                                      search_space=self) for r_ind in indexes[ind:end_ind]]
            ind = end_ind


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


class NasBench201RSController(BaseController):
    NAME = "nasbench-201-rs"

    def __init__(self, search_space, device, rollout_type="nasbench-201", mode="eval", check_valid=True, avoid_repeat=False, fair=False, deiso=False,
                 schedule_cfg=None):
        super(NasBench201RSController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        # get the infinite iterator of the model matrix and ops
        self.mode = mode
        self.num_vertices = self.search_space.num_vertices
        self.cur_solution = self.search_space.random_sample_arch()
        self.num_op_choices = self.search_space.num_op_choices
        self.num_ops = self.search_space.num_ops
        self.check_valid = check_valid
        self.avoid_repeat = avoid_repeat
        self.fair = fair
        self.deiso = deiso
        base_dir = os.path.join(utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-201")
        if self.deiso:
            fo = open(os.path.join(base_dir, "non-isom{}.txt".format(self.num_op_choices)))
            self.lines = fo.readlines()
            fo.close()
            arch_num_list = [106, 1093, 6466]
            self.arch_num = arch_num_list[self.num_op_choices - 3]
        elif self.avoid_repeat and self.num_op_choices != 5:
            fo = open(os.path.join(base_dir, "isom{}.txt".format(self.num_op_choices)))
            self.lines = fo.readlines()
            fo.close()
            arch_num_list = [729, 4096]
            self.arch_num = arch_num_list[self.num_op_choices - 3]

    def random_sample_nonisom(self):
        ind = np.random.randint(low=0, high=self.arch_num)
        arch = self.search_space.str2matrix(self.lines[ind].strip())
        return NasBench201Rollout(arch, self.search_space)

    def check_valid_arch(self, arch):
        valid_arch = False
        valid_input = [0]
        for to_ in range(1, self.num_vertices):
            for input_ in valid_input:
                if arch[to_][input_] > 0:
                    valid_input.append(to_)
                    break
        valid_output = [self.search_space.num_vertices - 1]
        for from_ in range(self.search_space.num_vertices - 2, -1, -1):
            for output_ in valid_output:
                if arch[output_][from_] > 0:
                    valid_output.append(from_)
        for input_ in valid_input:
            for output_ in valid_output:
                if self.search_space.ops_choices[int(arch[output_][input_])].find("conv") != -1:
                    valid_arch = True
        return valid_arch
                
    def sample(self, n=1, batch_size=None):
        rollouts = []
        if self.avoid_repeat:
            if self.deiso or self.num_op_choices != 5:
                assert n == self.arch_num
                for i in range(n):
                    line = self.lines[i].strip()
                    rollouts.append(NasBench201Rollout(self.search_space.str2matrix(line), self.search_space))
            else:
                indexes = np.random.choice(np.arange(15625), size=n, replace=False)
                for i in indexes:
                    rollouts.append(NasBench201Rollout(self.search_space.api.str2matrix(self.search_space.api.query_by_index(i).arch_str), self.search_space))
            return rollouts
        if self.fair:
            assert n == self.num_op_choices
            archs = np.zeros([self.num_ops, self.num_op_choices])
            for i in range(self.num_ops):
                archs[i, :] = np.random.permutation(np.arange(self.num_op_choices))
            for i in range(self.num_op_choices):
                arch = np.zeros([self.num_vertices, self.num_vertices])
                ind = 0
                for from_ in range(self.num_vertices - 1):
                    for to_ in range(from_ + 1, self.num_vertices):
                        arch[to_, from_] = archs[ind, i]
                        ind += 1
                if self.check_valid_arch(arch) or not self.check_valid:
                    rollouts.append(NasBench201Rollout(arch, self.search_space))
            return rollouts
        for i in range(n):
            while 1:
                if self.deiso:
                    new_rollout = self.random_sample_nonisom()
                else:
                    new_rollout = self.search_space.random_sample()
                if self.check_valid_arch(new_rollout.arch) or not self.check_valid:
                    rollouts.append(new_rollout)
                    break
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201"]

    # ---- APIs that is not necessary ----
    def set_mode(self, mode):
        self.mode = mode

    def step(self, rollouts, optimizer, perf_name):
        pass

    def set_device(self, device):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


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
                 sample_query=True,
                 schedule_cfg=None):
        super(NasBench201Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type)

        self.sample_query = sample_query

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

            # should use valid acc for search
            results = list(query_res.query("cifar10-valid").values())
            if self.sample_query:
                # use one run with random seed as the reward
                sampled_index = random.randint(0, len(results) - 1)
                reward = (results[sampled_index].eval_acc1es["x-valid@199"]) / 100.
            else:
                reward = np.mean(
                    [res.eval_acc1es["x-valid@199"] for res in results]) / 100.
            rollout.set_perf(reward, name="reward")
            rollout.set_perf(np.mean(
                [res.eval_acc1es["ori-test@199"] for res in results]) / 100.,
                             name="partial_test_acc")
            rollout.set_perf(np.mean(
                [res.eval_acc1es["ori-test@199"]
                 for res in query_res.query("cifar10").values()]) / 100., name="test_acc")

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


class NB201SharedCell(nn.Module):
    def __init__(self, op_cls, search_space, layer_index, num_channels, num_out_channels, stride):
        super(NB201SharedCell, self).__init__()
        self.search_space = search_space
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index

        self._vertices = self.search_space.num_vertices
        self._primitives = self.search_space.ops_choices

        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                self.edges[from_][to_] = op_cls(self.num_channels, self.num_out_channels,
                                                stride=self.stride, primitives=self._primitives)
                self.edge_mod.add_module("f_{}_t_{}".format(from_, to_), self.edges[from_][to_])
        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

    def on_replicate(self):
        # Although this edges is easy to understand, when paralleized,
        # the reference relationship between `self.edge` and modules under `self.edge_mod`
        # will not get updated automatically.

        # So, after each replicate, we should initialize a new edges dict
        # and update the reference manually.
        self.edges = defaultdict(dict)
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_ = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)] = edge_mod

    def forward(self, inputs, genotype):
        states_ = [inputs]
        for to_ in range(1, self._vertices):
            state_ = 0.
            for from_ in range(to_):
                out = self.edges[from_][to_](states_[from_], int(genotype[to_][from_]))
                state_ = state_ + out
            states_.append(state_)
        return states_[-1]

    def sub_named_members(self, genotype,
                          prefix="", member="parameters", check_visited=False):
        prefix = prefix + ("." if prefix else "")
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                edge_share_op = self.edges[from_][to_]
                for n, v in edge_share_op.sub_named_members(
                          int(genotype[to_][from_]),
                          prefix=prefix + "edge_mod.f_{}_t_{}".format(from_, to_),
                          member=member):
                    yield n, v


class NB201SharedOp(nn.Module):
    """
    The operation on an edge, consisting of multiple primitives.
    """

    def __init__(self, C, C_out, stride, primitives):
        super(NB201SharedOp, self).__init__()
        self.primitives = primitives
        self.stride = stride
        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = ops.get_op(primitive)(C, C_out, stride, False)
            self.p_ops.append(op)

    def forward(self, x, op_type):
        return self.p_ops[op_type](x)

    def sub_named_members(self, op_type, prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        for n, v in getattr(self.p_ops[op_type], "named_" + member)(prefix="{}p_ops.{}"\
                                  .format(prefix, op_type)):
            yield n, v

class NB201CandidateNet(CandidateNet):

    def __init__(self, super_net, rollout, member_mask, gpus=tuple(), cache_named_members=False, eval_no_grad=True):
        super(NB201CandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.search_space = super_net.search_space
        self.member_mask = member_mask
        self.cache_named_members = cache_named_members
        self._cached_np = None
        self._cached_nb = None

        self._flops_calculated = False
        self.total_flops = 0

        self.genotype_arch = rollout.arch
        self.genotype = rollout.genotype
 
    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0
        self.super_net.reset_flops()

    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net.forward(inputs, self.genotype_arch)

    def forward(self, inputs, single=False): #pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs={"single": True})

    def _forward_with_params(self, inputs, params, **kwargs): #pylint: disable=arguments-differ
        with use_params(self.super_net, params):
            return self.forward(inputs, **kwargs)

    def plot_arch(self):
        return self.super_net.search_space.plot_arch(self.genotype, "./nb201_search", "nb201_search")

    def named_parameters(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                # use cached members
                if self._cached_np is None:
                    self._cached_np = []
                    for n, v in self.active_named_members(member="parameters", prefix=""):
                        self._cached_np.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_np:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(member="parameters", prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_parameters(prefix=prefix):
                yield n, v

    def named_buffers(self, prefix="", recurse=True): #pylint: disable=arguments-differ
        if self.member_mask:
            if self.cache_named_members:
                if self._cached_nb is None:
                    self._cached_nb = []
                    for n, v in self.active_named_members(member="buffers", prefix=""):
                        self._cached_nb.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_nb:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(member="buffers", prefix=prefix):
                    yield n, v
        else:
            for n, v in self.super_net.named_buffers(prefix=prefix):
                yield n, v

    def active_named_members(self, member, prefix="", recurse=True, check_visited=False):
        """
        Get the generator of name-member pairs active
        in this candidate network. Always recursive.
        """
        # memo, there are potential weight sharing, e.g. when `tie_weight` is True in rnn_super_net,
        # encoder/decoder share weights. If there is no memo, `sub_named_members` will return
        # 'decoder.weight' and 'encoder.weight', both refering to the same parameter, whereasooo
        # `named_parameters` (with memo) will only return 'encoder.weight'. For possible future
        # weight sharing, use memo to keep the consistency with the builtin `named_parameters`.
        memo = set()
        for n, v in self.super_net.sub_named_members(self.genotype_arch,
                                                     prefix=prefix,
                                                     member=member,
                                                     check_visited=check_visited):
            if v in memo:
                continue
            memo.add(v)
            yield n, v

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        member_lst = []
        for n, v in itertools.chain(self.active_named_members(member="parameters", prefix=""),
                                    self.active_named_members(member="buffers", prefix="")):
            member_lst.append((n, v))
        state_dict = OrderedDict(member_lst)
        return state_dict


class NB201SharedNet(BaseWeightsManager, nn.Module):

    NAME = "nasbench-201"

    def __init__(self, search_space, device, rollout_type="nasbench-201",
                 gpus=tuple(),
                 num_classes=10, init_channels=16, stem_multiplier=1,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 candidate_member_mask=True, candidate_cache_named_members=False,
                 candidate_eval_no_grad=True):
        super(NB201SharedNet, self).__init__(search_space, device, rollout_type)
        nn.Module.__init__(self)

        cell_cls = NB201SharedCell
        op_cls = NB201SharedOp

        # optionally data parallelism in SharedNet
        self.gpus = gpus

        self.num_classes = num_classes
        # init channel number of the first cell layers,
        # x2 after every reduce cell
        self.init_channels = init_channels
        # channels of stem conv / init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem

        # training
        self.max_grad_norm = max_grad_norm
        self.dropout_rate = dropout_rate

        # search space configs
        self._num_vertices = self.search_space.num_vertices
        self._ops_choices = self.search_space.ops_choices
        self._num_layers = self.search_space.num_layers

        ## initialize sub modules
        if not self.use_stem:
            c_stem = 3
        elif isinstance(self.use_stem, (list, tuple)):
            self.stems = []
            c_stem = self.stem_multiplier * self.init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stems.append(ops.get_op(stem_type)(
                    c_in, c_stem, stride=stem_stride, affine=stem_affine))
            self.stems = nn.ModuleList(self.stems)
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(3, c_stem,
                                                  stride=stem_stride, affine=stem_affine)

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]

        for i_layer, stride in enumerate(strides):
            _num_channels = num_channels if i_layer != 0 else c_stem
            if stride > 1:
                num_channels *= stride
            # A patch: Can specificy input/output channels by hand in configuration,
            # instead of relying on the default
            # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
            _num_out_channels = num_channels
            if stride == 1:
                cell = cell_cls(op_cls,
                            self.search_space,
                            layer_index=i_layer,
                            num_channels=_num_channels,
                            num_out_channels=_num_out_channels,
                            stride=stride)
            else:
                cell = ops.get_op("NB201ResidualBlock")(_num_channels, _num_out_channels,
                                                 stride=2, affine=True)
            self.cells.append(cell)
        self.lastact = nn.Sequential(nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(num_channels, self.num_classes)

        self.to(self.device)
        self.candidate_member_mask = candidate_member_mask
        self.candidate_cache_named_members = candidate_cache_named_members
        self.candidate_eval_no_grad = candidate_eval_no_grad
        self.set_hook()
        self._flops_calculated = False
        self.total_flops = 0

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def set_hook(self):
        for name, module in self.named_modules():
            if "auxiliary" in name:
                continue
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    inputs[0].size(2) * inputs[0].size(3) / \
                                    (module.stride[0] * module.stride[1] * module.groups)
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def sub_named_members(self, genotype,
                          prefix="", member="parameters", check_visited=False):
        prefix = prefix + ("." if prefix else "")

        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "cells":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix+mod_name):
                yield n, v

        for cell_idx, cell in enumerate(self.cells):
            for n, v in cell.sub_named_members(genotype,
                                               prefix=prefix + "cells.{}".format(cell_idx),
                                               member=member,
                                               check_visited=check_visited):
                yield n, v

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return NB201CandidateNet(self, rollout,
                               gpus=self.gpus,
                               member_mask=self.candidate_member_mask,
                               cache_named_members=self.candidate_cache_named_members,
                               eval_no_grad=self.candidate_eval_no_grad)

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201"]

    def _is_reduce(self, layer_idx):
        return layer_idx in [(self._num_layers + 1) // 3 - 1, (self._num_layers + 1) * 2 // 3 - 1]

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, inputs, genotype, **kwargs): #pylint: disable=arguments-differ
        if not self.use_stem:
            states = inputs
        elif isinstance(self.use_stem, (list, tuple)):
            stemed = inputs
            for stem in self.stems:
                stemed = stem(stemed)
            states = stemed
        else:
            stemed = self.stem(inputs)
            states = stemed
        for cell in self.cells:
            states = cell(states, genotype, **kwargs)
        out = self.lastact(states)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()


    def save(self, path):
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    @classmethod
    def supported_data_types(cls):
        return ["image"]


class NB201GenotypeModel(FinalModel):
    NAME = "nb201_final_model"

    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(self, search_space, device, genotypes,
                 num_classes=10, init_channels=36, stem_multiplier=1,
                 dropout_rate=0.1, dropout_path_rate=0.2,
                 auxiliary_head=False, auxiliary_cfg=None,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 schedule_cfg=None):
        super(NB201GenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        assert isinstance(genotypes, str)
        self.genotype_arch = self.search_space.api.str2matrix(genotypes)

        self.num_classes = num_classes
        self.init_channels = init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem

        # training
        self.dropout_rate = dropout_rate
        self.dropout_path_rate = dropout_path_rate
        self.auxiliary_head = auxiliary_head

        # search space configs
        self._num_vertices = self.search_space.num_vertices
        self._ops_choices = self.search_space.ops_choices
        self._num_layers = self.search_space.num_layers

        ## initialize sub modules
        if not self.use_stem:
            c_stem = 3
        elif isinstance(self.use_stem, (list, tuple)):
            self.stems = []
            c_stem = self.stem_multiplier * self.init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stems.append(ops.get_op(stem_type)(
                    c_in, c_stem, stride=stem_stride, affine=stem_affine))
            self.stems = nn.ModuleList(self.stems)
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(3, c_stem,
                                                  stride=stem_stride, affine=stem_affine)

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]
        for i_layer, stride in enumerate(strides):
            _num_channels = num_channels if i_layer != 0 else c_stem
            if stride > 1:
                num_channels *= stride
            _num_out_channels = num_channels
            if stride == 1:
                cell = NB201GenotypeCell(self.search_space,
                                   self.genotype_arch,
                                   layer_index=i_layer,
                                   num_channels=_num_channels,
                                   num_out_channels=_num_out_channels,
                                   stride=stride)
            else:
                cell = ops.get_op("NB201ResidualBlock")(_num_channels, _num_out_channels, stride=2, affine=True)
            # TODO: support specify concat explicitly
            self.cells.append(cell)

            if i_layer == (2 * self._num_layers) // 3 and self.auxiliary_head:
                if auxiliary_head == "imagenet":
                    self.auxiliary_net = AuxiliaryHeadImageNet(
                        prev_num_channels[-1], num_classes, **(auxiliary_cfg or {}))
                else:
                    self.auxiliary_net = AuxiliaryHead(
                        prev_num_channels[-1], num_classes, **(auxiliary_cfg or {}))
        self.lastact = nn.Sequential(nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(num_channels,
                                    self.num_classes)
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def set_hook(self):
        for name, module in self.named_modules():
            if "auxiliary" in name:
                continue
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += 2* inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    outputs.size(2) * outputs.size(3) / module.groups
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def forward(self, inputs): #pylint: disable=arguments-differ
        if not self.use_stem:
            states = inputs
        elif isinstance(self.use_stem, (list, tuple)):
            stemed = inputs
            for stem in self.stems:
                stemed = stem(stemed)
            states = stemed
        else:
            stemed = self.stem(inputs)
            states = stemed

        for layer_idx, cell in enumerate(self.cells):
            states = cell(states, self.dropout_path_rate)
            if layer_idx == 2 * self._num_layers // 3:
                if self.auxiliary_head and self.training:
                    logits_aux = self.auxiliary_net(states)
        out = self.lastact(states)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops/1.e6)
            self._flops_calculated = True

        if self.auxiliary_head and self.training:
            return logits, logits_aux
        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _is_reduce(self, layer_idx):
        return layer_idx in [(self._num_layers + 1) // 3 - 1, (self._num_layers + 1) * 2 // 3 - 1]


class NB201GenotypeCell(nn.Module):
    def __init__(self, search_space, genotype_arch, layer_index, num_channels, num_out_channels, stride):
        super(NB201GenotypeCell, self).__init__()
        self.search_space = search_space
        self.arch = genotype_arch
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index

        self._vertices = self.search_space.num_vertices
        self._primitives = self.search_space.ops_choices

        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module() # a stub wrapping module of all the edges
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                self.edges[from_][to_] = ops.get_op(self._primitives[int(self.arch[to_][from_])])(
                    self.num_channels, self.num_out_channels, stride=self.stride, affine=False)

                self.edge_mod.add_module("f_{}_t_{}".format(from_, to_), self.edges[from_][to_])
        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

    def forward(self, inputs, dropout_path_rate): #pylint: disable=arguments-differ
        states = [inputs]

        for to_ in range(1, self._vertices):
            state_to_ = 0.
            for from_ in range(to_):
                op = self.edges[from_][to_]
                out = op(states[from_])
                if self.training and dropout_path_rate > 0:
                    if not isinstance(op, ops.Identity):
                        out = utils.drop_path(out, dropout_path_rate)
                state_to_ = state_to_ + out
            states.append(state_to_)

        return states[-1]

    def on_replicate(self):
        # Although this edges is easy to understand, when paralleized,
        # the reference relationship between `self.edge` and modules under `self.edge_mod`
        # will not get updated automatically.

        # So, after each replicate, we should initialize a new edges dict
        # and update the reference manually.
        self.edges = defaultdict(dict)
        for edge_name, edge_mod in six.iteritems(self.edge_mod._modules):
            from_, to_  = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)] = edge_mod
