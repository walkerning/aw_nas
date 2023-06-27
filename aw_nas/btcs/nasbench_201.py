"""
NASBench-201 search space, rollout, embedder
"""

import os
import re
import copy
import random
import pickle
import itertools
import collections
from typing import List, Optional, NamedTuple, Tuple
from collections import defaultdict, OrderedDict
import contextlib

import six
import yaml
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from nas_201_api import NASBench201API as API

from aw_nas import utils, ops
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.controller.base import BaseController
from aw_nas.controller import DiffController
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.utils import (
    DenseGraphSimpleOpEdgeFlow,
    DenseGraphConvolution,
    data_parallel,
    use_params,
    softmax,
)
from aw_nas.utils.parallel_utils import _check_support_candidate_member_mask
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.final.base import FinalModel

VERTICES = 4


class NasBench201SearchSpace(SearchSpace):
    NAME = "nasbench-201"

    def __init__(
        self,
        num_layers=17,
        vertices=4,
        load_nasbench=True,
        ops_choices=(
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        ),
    ):
        super(NasBench201SearchSpace, self).__init__()

        self.ops_choices = ops_choices
        self.ops_choice_to_idx = {
            choice: i for i, choice in enumerate(self.ops_choices)
        }

        self.load_nasbench = load_nasbench
        self.num_vertices = vertices
        self.num_layers = num_layers
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices)  # 5
        self.num_ops = self.num_vertices * (self.num_vertices - 1) // 2
        self.idx = np.tril_indices(self.num_vertices, k=-1)
        self.genotype_type = str

        if self.load_nasbench:
            self._init_nasbench()

    def canonicalize(self, rollout):
        # TODO
        arch = rollout.arch
        num_vertices = rollout.search_space.num_vertices
        op_choices = rollout.search_space.ops_choices
        S = []
        S.append("0")
        res = ""
        for i in range(1, num_vertices):
            preS = []
            s = ""
            for j in range(i):
                if ((int(arch[i][j]) == 0) or (S[j] == "#")):
                    s = "#"
                elif (int(arch[i][j]) == 1):
                    s = S[j]
                else:
                    s = "(" + S[j] + ")" + "@" + op_choices[int(arch[i][j])]
                preS.append(s)
            preS.sort()
            s = ""
            for j in range(i):
                s = s + preS[j]
            S.append(s)
            res = s
        return res

    def __getstate__(self):
        state = super(NasBench201SearchSpace, self).__getstate__().copy()
        if "api" in state:
            del state["api"]
        return state

    def __setstate__(self, state):
        super(NasBench201SearchSpace, self).__setstate__(state)
        if self.load_nasbench:
            self._init_nasbench()

    # optional API
    def genotype_from_str(self, genotype_str):
        return genotype_str

    # ---- APIs ----
    def random_sample(self):
        return NasBench201Rollout(self.random_sample_arch(), search_space=self)

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
        graph.node(str(0), fillcolor="darkseagreen2")
        graph.node(str(self.num_vertices - 1), fillcolor="palegoldenrod")
        [
            graph.node(str(i), fillcolor="lightblue")
            for i in range(1, self.num_vertices - 1)
        ]

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
        return ["nasbench-201", "nasbench-201-differentiable"]

    def mutate(self, rollout):  # pylint: disable=arguments-differ
        rand_ind = np.random.randint(0, self.idx[0].shape[0])
        neighbor_choice = np.random.randint(0, self.num_op_choices)
        arch_mat = rollout.arch
        while neighbor_choice == arch_mat[self.idx[0][rand_ind], self.idx[1][rand_ind]]:
            neighbor_choice = np.random.randint(0, self.num_op_choices)
        new_arch_mat = copy.deepcopy(arch_mat)
        new_arch_mat[self.idx[0][rand_ind], self.idx[1][rand_ind]] = neighbor_choice
        return NasBench201Rollout(new_arch_mat, self)

    # ---- helpers ----
    def matrix2str(self, arch):
        node_strs = []
        for i_node in range(1, self.num_vertices):
            node_strs.append(
                "|"
                + "|".join(
                    [
                        "{}~{}".format(
                            self.ops_choices[int(arch[i_node, i_input])], i_input
                        )
                        for i_input in range(0, i_node)
                    ]
                )
                + "|"
            )
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
        self.base_dir = os.path.join(
            utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-201"
        )
        self.api = API(os.path.join(self.base_dir, "NAS-Bench-201-v1_0-e61699.pth"))

    def op_to_idx(self, ops):
        return [self.ops_choice_to_idx[op] for op in ops]

    def random_sample_arch(self):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        arch[np.tril_indices(self.num_vertices, k=-1)] = np.random.randint(
            low=0, high=self.num_op_choices, size=self.num_ops
        )
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
            yield [
                NasBench201Rollout(
                    matrix=self.api.str2matrix(self.api.arch(r_ind)), search_space=self
                )
                for r_ind in indexes[ind:end_ind]
            ]
            ind = end_ind


class FLOPSConstrainedNasBench201SearchSpace(NasBench201SearchSpace):
    NAME = "nasbench-201-flops-constrained"

    def __init__(
        self,
        upper_flops: float,
        num_layers: int = 17,
        vertices: int = 4,
        ops_choices: Tuple[str] = (
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        )
        ):
        super(FLOPSConstrainedNasBench201SearchSpace, self).__init__(
                num_layers, vertices, True, ops_choices)
        self.upper_flops = upper_flops

    def _cal_flops(self, rollout) -> float:
        query_idx = rollout.search_space.api.query_index_by_arch(rollout.genotype)
        query_res = rollout.search_space.api.query_by_index(query_idx)
        results = list(query_res.query("cifar10-valid").values())
        flops = results[0].flop
        return flops

    def random_sample(self):
        flops = np.inf
        while flops > self.upper_flops:
            rollout = super(FLOPSConstrainedNasBench201SearchSpace, self).random_sample()
            flops = self._cal_flops(rollout)
        return rollout

    def mutate(self, rollout):
        flops = np.inf
        while flops > self.upper_flops:
            child = super(FLOPSConstrainedNasBench201SearchSpace, self).mutate(rollout)
            flops = self._cal_flops(child)
        return child


class NasBench201Rollout(BaseRollout):
    NAME = "nasbench-201"
    supported_components = [("controller", "rl"), ("evaluator", "mepa")]

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
            self.genotype, filename, label=label, edge_labels=edge_labels
        )

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

    def __repr__(self):
        return "NasBench201Rollout(matrix={arch}, perf={perf})".format(
            arch=self.arch, perf=self.perf
        )


try:  # Python >= 3.6

    class DiffArch(NamedTuple):
        op_weights: torch.Tensor
        edge_norms: Optional[torch.Tensor] = None


except (SyntaxError, TypeError):
    DiffArch = NamedTuple(
        "DiffArch",
        [("op_weights", torch.Tensor), ("edge_norms", Optional[torch.Tensor])],
    )


class NasBench201DiffRollout(BaseRollout):
    NAME = "nasbench-201-differentiable"
    supported_components = [
        ("controller", "nasbench-201-gcn-differentiable"),
        ("evaluator", "mepa"),
        ("trainer", "simple"),
    ]

    def __init__(
        self, arch: List[DiffArch], sampled, logits, search_space, candidate_net=None
    ):
        super(NasBench201DiffRollout, self).__init__()

        self.arch = arch
        self.sampled = sampled
        self.logits = logits
        self.search_space = search_space
        self.candidate_net = candidate_net

        self._genotype = None
        self._discretized_arch = None
        self._edge_probs = None

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def plot_arch(self, filename, label="", edge_labels=None):
        if edge_labels is None:
            edge_labels = self.discretized_arch_and_prob[1]
        return self.search_space.plot_arch(
            self.genotype, filename, label=label, edge_labels=edge_labels
        )

    def genotype_list(self):
        return list(self.genotype._asdict().items())

    def parse(self, weights):

        probs = softmax(self.logits)
        start = 0
        n = 1
        num_steps = self.search_space.num_vertices
        arch = [[], []]
        edge_prob = []
        for _ in range(1, num_steps):
            end = start + n
            w = weights[start:end]
            prob = probs[start:end]
            edges = sorted(range(n), key=lambda x: -max(w[x]))
            arch[0] += edges
            op_lst = [np.argmax(w[edge]) for edge in edges]
            edge_prob += [
                "{:.3f}".format(prob[edge][op_id]) for edge, op_id in zip(edges, op_lst)
            ]
            arch[1] += op_lst
            n += 1
            start = end

        num = self.search_space.num_vertices
        archs = [[0 for i in range(num)] for i in range(num)]
        p = 0
        for i in range(1, num):
            for j in range(i):
                archs[i][arch[0][p]] = arch[1][p]
                p += 1

        return np.array(archs), edge_prob

    @property
    def discretized_arch_and_prob(self):
        if self._discretized_arch is None:
            if self.arch[0].edge_norms is None:
                weights = self.sampled
            else:
                edge_norms = utils.get_numpy(self.arch.edge_norms)
                weights = utils.get_numpy(self.sampled) * edge_norms

            self._discretized_arch, self._edge_probs = self.parse(weights)

        return self._discretized_arch, self._edge_probs

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(
                self.discretized_arch_and_prob[0]
            )
        return self._genotype

    def __repr__(self):
        return (
            "NasBench201DiffRollout(search_space={sn}, arch={arch}, "
            "candidate_net={cn}, perf={perf})"
        ).format(
            sn=self.search_space.NAME,
            arch=self.arch,
            cn=self.candidate_net,
            perf=self.perf,
        )


class NasBench201RSController(BaseController):
    NAME = "nasbench-201-rs"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-201",
        mode="eval",
        check_valid=True,
        avoid_repeat=False,
        fair=False,
        deiso=False,
        op_type=0,
        pickle_file="",
        text_file="",
        shuffle_indices_avoid_repeat=True,
        schedule_cfg=None,
    ):
        super(NasBench201RSController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

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
        self.pickle_file = pickle_file
        self.text_file = text_file
        self.shuffle_indices_avoid_repeat = shuffle_indices_avoid_repeat
        self.lines = None
        if self.text_file:
            with open(self.text_file) as rf:
                self.lines = rf.readlines()
        elif self.pickle_file:
            with open(self.pickle_file, "rb") as rf:
                self.lines = pickle.load(rf)
        else:
            # if neither text_file nor pickle_file is speficied,
            # assume non-isom{num op choices}.txt is under the awnas data dir
            base_dir = os.path.join(utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-201")
            isom_table_fname = os.path.join(base_dir, "non-isom{}.txt".format(self.num_op_choices))
            if self.deiso:
                assert os.path.exists(isom_table_fname)
                with open(isom_table_fname) as rf:
                    self.lines = rf.readlines()
        if self.lines is not None:
            self.arch_num = len(self.lines)
        else:
            self.arch_num = 15625

        if self.deiso:
            print("Deiso arch num: ", self.arch_num)

        self.index = 0
        self.indices = np.arange(self.arch_num)
        if self.shuffle_indices_avoid_repeat:
            np.random.shuffle(self.indices)

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
                if (
                    self.search_space.ops_choices[int(arch[output_][input_])].find(
                        "conv"
                    )
                    != -1
                ):
                    valid_arch = True
        return valid_arch

    def sample(self, n=1, batch_size=None):
        rollouts = []
        if self.avoid_repeat:
            if self.deiso or self.num_op_choices != 5:
                # assert n == self.arch_num
                for i in range(n):
                    line = self.lines[i].strip()
                    rollouts.append(
                        NasBench201Rollout(
                            self.search_space.str2matrix(line), self.search_space
                        )
                    )
            elif self.pickle_file:
                for line in self.lines:
                    rollouts.append(NasBench201Rollout(line[0], self.search_space))
            else:
                next_index = self.index + n
                # indexes = np.random.choice(np.arange(15625), size=n, replace=False)
                if self.text_file:
                    rollouts = [NasBench201Rollout(
                        self.search_space.str2matrix(self.lines[self.indices[i]].strip()),
                        self.search_space)
                                for i in range(self.index, min(next_index, 15625))]
                else:
                    rollouts = [NasBench201Rollout(
                        self.search_space.api.str2matrix(
                            self.search_space.api.query_by_index(self.indices[i]).arch_str
                        ),
                        self.search_space,
                    ) for i in range(self.index, min(next_index, 15625))]

                if next_index >= 15625:
                    # reshuffle the indices
                    if self.shuffle_indices_avoid_repeat:
                        np.random.shuffle(self.indices)
                    next_index = next_index - 15625
                    if self.text_file:
                        rollouts += [NasBench201Rollout(
                            self.search_space.str2matrix(self.lines[self.indices[i]].strip()),
                            self.search_space)
                                     for i in range(0, next_index)]
                    else:
                        rollouts += [NasBench201Rollout(
                            self.search_space.api.str2matrix(
                                self.search_space.api.query_by_index(self.indices[i]).arch_str
                            ),
                            self.search_space)
                                     for i in range(0, next_index)]

                self.index = next_index
            return rollouts

        if self.fair:
            assert n == self.num_op_choices
            archs = np.zeros([self.num_op_choices,
                              self.search_space.num_vertices,
                              self.search_space.num_vertices])
            ops = np.array([
                np.random.permutation(np.arange(self.num_op_choices))
                for _ in range(self.num_ops)
            ]).T
            for i in range(self.num_op_choices):
                archs[i][self.search_space.idx] = ops[i]
            rollouts = [NasBench201Rollout(arch, self.search_space) for arch in archs
                        if self.check_valid_arch(arch) or not self.check_valid]
            return rollouts

        for i in range(n):
            while 1:
                if self.deiso:
                    new_rollout = self.random_sample_nonisom()
                elif self.pickle_file:
                    new_rollout = NasBench201Rollout(
                        self.lines[np.random.randint(0, len(self.lines))][0],
                        self.search_space,
                    )
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
        self.logger.info("nasbench-201-rs controller would not be loaded from the disk")


class GCN(nn.Module):
    def __init__(self, num_vertices, layers, size):
        super(GCN, self).__init__()

        self.gcns = []
        for i in range(layers):
            self.gcns.append(
                DenseGraphConvolution(
                    in_features=size,
                    out_features=size,
                    plus_I=False,
                    normalize=False,
                    bias=False,
                )
            )
        self.gcns = nn.ModuleList(self.gcns)

        self.layers = layers
        self.num_vertices = num_vertices

    def forward(self, x):
        adj = np.zeros((self.num_vertices, self.num_vertices), dtype=np.float32)
        for i in range(self.num_vertices):
            for j in range(i):
                adj[j][i] = 1.0 / (j + 1)
        adj = (torch.from_numpy(adj) + torch.eye(self.num_vertices, dtype=torch.float32)).cuda()

        out = x
        for i in range(self.layers):
            out = self.gcns[i](out, adj)
            if i != self.layers - 1:
                out = F.relu(out)

        return out


class MLP(nn.Module):
    def __init__(self, num_vertices, layers, size):
        super(MLP, self).__init__()

        self.num_vertices = num_vertices
        self.net = []
        for i in range(1, layers + 1):
            self.net.append(nn.Linear(size[i - 1], size[i]))

        self.net = nn.ModuleList(self.net)
        self.layers = layers

    def forward_single(self, x):
        out = x
        for i in range(self.layers):
            out = self.net[i](out)
            if i != self.layers - 1:
                out = F.relu(out)

        return out

    def forward(self, x):
        prob = []
        for i in range(self.num_vertices):
            for j in range(i):
                out = self.forward_single(torch.cat([x[j], x[i]]))
                prob.append(out)
        return prob


class NasBench201DiffController(DiffController, nn.Module):
    """
    Differentiable controller for nasbench-201.
    """

    NAME = "nasbench-201-differentiable"

    SCHEDULABLE_ATTRS = [
        "gumbel_temperature",
        "entropy_coeff",
        "force_uniform"
    ]

    def __init__(self, search_space: SearchSpace, device: torch.device, 
                 rollout_type: str = "nasbench-201-differentiable",
                 use_prob: bool = False, gumbel_hard: bool = False, 
                 gumbel_temperature: float = 1.0, entropy_coeff: float = 0.01, 
                 max_grad_norm: float = None, force_uniform: bool = False, 
                 inspect_hessian_every: int = -1, schedule_cfg = None):
        BaseController.__init__(self, search_space, rollout_type, schedule_cfg = schedule_cfg)
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.use_prob = use_prob
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        self.inspect_hessian_every = inspect_hessian_every
        self.inspect_hessian = False

        self.cg_alpha = nn.Parameter(1e-3 * 
            torch.randn(self.search_space.num_possible_edges, self.search_space.num_op_choices)
        )

        # meta learning related
        self.params_clone = None
        self.buffers_clone = None
        self.grad_clone = None
        self.grad_count = 0

        self.to(self.device)

    def sample(self, n: int = 1, batch_size: int = None):
        assert batch_size is None or batch_size == 1, "Do not support sample batch size for now"
        rollouts = []

        for _ in range(n):
            alpha = torch.zeros_like(self.cg_alpha) if self.force_uniform else self.cg_alpha
            
            if self.use_prob:
                sampled = F.softmax(alpha / self.gumbel_temperature, dim = -1)
            else:
                # gumbel sampling
                sampled, _ = utils.gumbel_softmax(alpha, self.gumbel_temperature, hard = False)

            op_weights_list = utils.straight_through(sampled) if self.gumbel_hard else sampled
            sampled_list = utils.get_numpy(sampled)
            logits_list = utils.get_numpy(alpha)

            arch_list = [
                DiffArch(op_weights = op_weights, edge_norms = None) 
                for op_weights in op_weights_list
            ]

            rollouts.append(
                NasBench201DiffRollout(
                    arch_list, sampled_list, logits_list, self.search_space
                )
            )

        return rollouts

    def _entropy_loss(self):
        if self.entropy_coeff is not None:
            prob = F.softmax(self.cg_alpha, dim = -1)
            return - self.entropy_coeff * (torch.log(prob) * prob).sum()
        return 0.

    def summary(self, rollouts, log: bool = False, log_prefix: str = "", step: int = None):
        num = len(rollouts)
        logits_list = [[utils.get_numpy(logits) for logits in r.logits] for r in rollouts]
        if self.gumbel_hard:
            cg_logprob = 0.
        cg_entro = 0.
        for rollout, logits in zip(rollouts, logits_list):
            prob = utils.softmax(logits)
            logprob = np.log(prob)
            if self.gumbel_hard:
                op_weights = [arch.op_weights.tolist() for arch in rollout.arch]
                inds = np.argmax(utils.get_numpy(op_weights), axis=-1)
                cg_logprob += np.sum(logprob[range(len(inds)), inds])
            cg_entro += -(prob * logprob).sum()

        # mean across rollouts
        if self.gumbel_hard:
            cg_logprob /= num
            cg_logprobs_str = "{:.2f}".format(cg_logprob)

        cg_entro /= num
        cg_entro_str = "{:.2f}".format(cg_entro)

        if log:
            # maybe log the summary
            self.logger.info("%s%d rollouts: %s ENTROPY: %2f (%s)",
                    log_prefix, num,
                    "-LOG_PROB: %.2f (%s) ;" % (-cg_logprob, cg_logprobs_str) \
                        if self.gumbel_hard else "",
                    cg_entro, cg_entro_str)
            if step is not None and not self.writer.is_none():
                if self.gumbel_hard:
                    self.writer.add_scalar("log_prob", cg_logprob, step)
                self.writer.add_scalar("entropy", cg_entro, step)

        stats = [("ENTRO", cg_entro)]
        if self.gumbel_hard:
            stats += [("LOGPROB", cg_logprob)]
        return OrderedDict(stats)

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201-differentiable"]


class NasBench201GcnController(BaseController, nn.Module):
    """
    Implementation following Neural Graph Embedding for Neural Architecture Search, AAAI 2020
    """

    NAME = "nasbench-201-gcn-differentiable"

    def __init__(
        self,
        search_space,
        device="cuda",
        mode="val",
        rollout_type="nasbench-201-differentiable",
        embed_size=10,
        gcn_layers=5,
        mlp_layers=3,
        mlp_size=[15, 10],
        use_prob=False,
        gumbel_hard=False,
        gumbel_temp=1.0,
        use_edge_norm=False,
        entropy_coeff=0.01,
        max_grad_norm=None,
        force_uniform=False,
        inspect_hessian_every=-1,
        schedule_cfg=None,
    ):

        super(NasBench201GcnController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )
        nn.Module.__init__(self)

        self.num_vertices = self.search_space.num_vertices
        self.embed_size = embed_size
        self.node_embed = nn.Parameter(
            1e-3 * torch.randn(self.num_vertices, self.embed_size)
        )
        self.gcn_layers = gcn_layers
        self.mlp_layers = mlp_layers
        self.mlp_size = (
            [self.embed_size * 2] + mlp_size + [self.search_space.num_op_choices]
        )
        self.gcn = GCN(self.num_vertices, self.gcn_layers, self.embed_size)
        self.mlp = MLP(self.num_vertices, self.mlp_layers, self.mlp_size)
        self.prob = None

        self.use_prob = use_prob
        self.gumbel_hard = gumbel_hard
        self.gumbel_temp = gumbel_temp

        self.use_edge_norm = use_edge_norm

        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        self.inspect_hessian_every = inspect_hessian_every
        self.inspect_hessian = False

        self.device = device
        self.mode = mode
        self.set_device(device)
        self.set_mode(mode)

    def on_epoch_start(self, epoch):
        super(NasBench201GcnController, self).on_epoch_start(epoch)
        if self.inspect_hessian_every >= 0 and epoch % self.inspect_hessian_every == 0:
            self.inspect_hessian = True

    def set_mode(self, mode):
        self.mode = mode

    def set_device(self, device):
        self.device = device
        self.to(torch.device(device))

    def get_prob(self):
        prob = self.gcn(self.node_embed)
        prob = self.mlp(prob)
        return prob

    def forward(self, n=1):
        return self.sample(n=n)

    def sample(self, n=1, batch_size=None):
        assert batch_size is None or batch_size == 1, "Do not support sample batch size for now"
        self.probs = self.get_prob()
        rollouts = []
        for _ in range(n):
            op_weights_list = []
            sampled_list = []
            logits_list = []

            for prob in self.probs:
                if self.force_uniform:
                    prob = torch.zeros_like(prob)

                if self.use_prob:
                    sampled = F.softmax(prob / self.gumbel_temp, dim=-1)
                else:
                    sampled, _ = utils.gumbel_softmax(
                        prob, self.gumbel_temp, hard=False
                    )

                if self.gumbel_hard:
                    op_weights = utils.straight_through(sampled)
                else:
                    op_weights = sampled

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampled))
                logits_list.append(utils.get_numpy(prob))

            arch_list = [
                DiffArch(op_weights=op_weights, edge_norms=None)
                for op_weights in op_weights_list
            ]

            rollouts.append(
                NasBench201DiffRollout(
                    arch_list, sampled_list, logits_list, self.search_space
                )
            )

        return rollouts

    def save(self, path):
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    def _entropy_loss(self):
        if self.entropy_coeff is not None:
            probs = [F.softmax(prob, dim=-1) for prob in self.probs]
            return self.entropy_coeff * sum(
                -(torch.log(prob) * prob).sum() for prob in probs
            )
        return 0

    def gradient(self, loss, return_grads=True, zero_grads=True):
        if zero_grads:
            self.zero_grad()

        _loss = loss + self._entropy_loss()
        _loss.backward()
        if return_grads:
            return utils.get_numpy(_loss), [
                (k, v.grad.clone()) for k, v in self.named_parameters()
            ]
        return utils.get_numpy(_loss)

    def step_current_gradient(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step_gradient(self, gradients, optimizer):
        self.zero_grad()
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad

        if self.max_grad_norm is not None:
            torch.nn.utls.clip_grad_norm_(self.parameters(), self.max_grad_norm)

        optimizer.step()

    def step(self, rollouts, optimizer=None, perf_name="reward"):
        self.zero_grad()
        losses = [r.get_perf(perf_name) for r in rollouts]
        [l.backward() for l in losses]
        optimizer.step()
        return np.mean([l.detach().cpu().numpy() for l in losses])

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        return None

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201-differentiable"]


class NasBench201EvoController(BaseController):
    NAME = "nasbench-201-evo"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-201",
        mode="eval",
        population_nums=100,
        schedule_cfg=None,
    ):
        super(NasBench201EvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

        # get the infinite iterator of the model matrix and ops
        self.mode = mode
        self.num_vertices = self.search_space.num_vertices
        self.cur_solution = self.search_space.random_sample_arch()
        self.population_nums = population_nums
        self.population = collections.OrderedDict()
        self.num_arch = len(self.search_space.api)
        population_ind = np.random.choice(
            np.arange(self.num_arch), size=self.population_nums, replace=False
        )
        for i in range(self.population_nums):
            arch_res = self.search_space.api.query_by_index(population_ind[i])
            accs = (
                np.mean(
                    [
                        res.eval_acc1es["ori-test@199"]
                        for res in arch_res.query("cifar10").values()
                    ]
                )
                / 100.0
            )
            self.population[arch_res.arch_str] = accs

    def reinit(self):
        population_ind = np.random.choice(
            np.arange(self.num_arch), size=self.population_nums, replace=False
        )
        for i in range(self.population_nums):
            arch_res = self.search_space.api.query_by_index(population_ind[i])
            accs = (
                np.mean(
                    [
                        res.eval_acc1es["ori-test@199"]
                        for res in arch_res.query("cifar10").values()
                    ]
                )
                / 100.0
            )
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
                best_sets.append(
                    NasBench201Rollout(
                        self.search_space.api.str2matrix(new_archs[n_r][0]),
                        self.search_space,
                    )
                )
            return best_sets
        rollouts = []
        for n_r in range(n):
            try_times = 0
            while True:
                rand_ind = np.random.randint(0, self.search_space.idx[0].shape[0])
                neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
                arch_mat = self.search_space.api.str2matrix(new_archs[n_r][0])
                while (
                    neighbor_choice
                    == arch_mat[
                        self.search_space.idx[0][rand_ind],
                        self.search_space.idx[1][rand_ind],
                    ]
                ):
                    neighbor_choice = np.random.randint(
                        0, self.search_space.num_op_choices
                    )
                new_choice = copy.deepcopy(arch_mat)
                new_choice[
                    self.search_space.idx[0][rand_ind],
                    self.search_space.idx[1][rand_ind],
                ] = neighbor_choice
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

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-201",
        mode="eval",
        temperature=1000,
        anneal_coeff=0.98,
        schedule_cfg=None,
    ):
        super(NasBench201SAController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg
        )

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
        self.logger.info(
            "Set the initialization rollout: {}; perf: {}".format(
                best_rollout, self.cur_perf
            )
        )

    def sample(self, n, batch_size=None):
        assert batch_size is None

        if self.mode == "eval":
            return [NasBench201Rollout(self.cur_solution, self.search_space)] * n

        rollouts = []
        for n_r in range(n):
            rand_ind = np.random.randint(0, self.search_space.idx[0].shape[0])
            neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
            while (
                neighbor_choice
                == self.cur_solution[
                    self.search_space.idx[0][rand_ind],
                    self.search_space.idx[1][rand_ind],
                ]
            ):
                neighbor_choice = np.random.randint(0, self.search_space.num_op_choices)
            new_choice = copy.deepcopy(self.cur_solution)
            new_choice[
                self.search_space.idx[0][rand_ind], self.search_space.idx[1][rand_ind]
            ] = neighbor_choice
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
class NasBench201_LineGraphEmbedder(ArchEmbedder):
    NAME = "nb201-linegcn"

    def __init__(
        self,
        search_space,
        op_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        dropout=0.0,
        gcn_kwargs=None,
        use_bn=False,
        use_cat=False,
        schedule_cfg=None,
    ):
        super(NasBench201_LineGraphEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_embedding_dim = op_embedding_dim
        self.hid_dim = hid_dim
        self.gcn_out_dims = gcn_out_dims
        self.use_bn = use_bn
        self.dropout = dropout
        self.use_cat = use_cat

        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices

        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.x_hidden = nn.Linear(self.op_embedding_dim, self.hid_dim)

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphConvolution(in_dim, dim, **(gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.out_dim = in_dim * (1 if not self.use_cat else 6)

        adj = torch.tensor(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0],
                ],
                dtype=np.float32,
            )
        )
        self.register_buffer("adj", adj)
        self.idx = list(zip(*[[1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]))

    def embed_and_transform_arch(self, archs):
        op_inds = self.op_emb.weight.new([arch[self.idx] for arch in archs]).long()
        embs = self.op_emb(op_inds)  # batch_size x 6 x op_embedding_dim
        b_size = embs.shape[0]
        x = self.x_hidden(embs)
        adjs = self.adj.unsqueeze(0).repeat([b_size, 1, 1])
        return adjs, x

    def forward(self, archs):
        adjs, x = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        return y.reshape(y.shape[0], -1) if self.use_cat else torch.mean(y, dim=1)


class NasBench201_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "nb201-lstm"

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
        super(NasBench201_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.op_emb = nn.Embedding(self.search_space.num_op_choices, self.emb_hid)

        self.rnn = nn.LSTM(
            input_size=self.emb_hid,
            hidden_size=self.num_hid,
            num_layers=self.num_layers,
            batch_first=True,
        )

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

    def __init__(
        self,
        search_space,
        op_embedding_dim=48,
        node_embedding_dim=48,
        hid_dim=96,
        gcn_out_dims=[128, 128],
        share_op_attention=False,
        gcn_kwargs=None,
        use_bn=False,
        use_final_only=False,
        share_self_op_emb=False,
        dropout=0.0,
        schedule_cfg=None,
    ):
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
            torch.zeros(1, self.node_embedding_dim), requires_grad=False
        )

        # the last embedding is the output op emb
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(
                torch.FloatTensor(self.op_embedding_dim).normal_()
            )
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphSimpleOpEdgeFlow(
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
        adjs = self.op_emb.weight.new(archs).long()
        op_embs = self.op_emb(adjs)  # (batch_size, vertices, vertices, op_emb_dim)
        b_size = op_embs.shape[0]
        node_embs = torch.cat(
            (
                self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1]),
            ),
            dim=1,
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs

    def forward(self, archs, return_all=False):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        adjs, x, op_embs = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_embs, self_op_emb=self.self_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if return_all:
            return y
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding
            y = torch.mean(y, dim=1)  # average across nodes (bs, god)
        return y


class NB201FBFlowArchEmbedder(ArchEmbedder):
    """
    Implement of TA-GATES architecture embedder on NAS-Bench-201.
    """
    NAME = "nb201-fbflow"

    def __init__(
        self,
        search_space,
        op_embedding_dim: int = 48,
        node_embedding_dim: int = 48,
        hid_dim: int = 96,
        gcn_out_dims: List[int] = [128, 128, 128],
        share_op_attention: bool = False,
        gcn_kwargs: dict = None,
        use_bn: bool = False,
        use_final_only: bool = False,
        share_self_op_emb: bool = False,
        dropout: float = 0.,
        init_input_node_emb: bool = True,

        # construction configurations
        # construction (tagates)
        num_time_steps: int = 2,
        fb_conversion_dims: List[int] = [128, 128],
        backward_gcn_out_dims: List[int] = [128, 128, 128],
        updateopemb_method: str = "concat_ofb_message", # concat_ofb_message, concat_ofb
        updateopemb_scale: float = 0.1,
        updateopemb_dims: List[int] = [128],
        b_use_bn: bool = False,
        # construction (l): concat arch-level zeroshot as l
        concat_arch_zs_as_l_dimension = None,
        concat_l_layer: int = 0,
        # construction (symmetry breaking)
        symmetry_breaking_method: str = None, # None, "random", "param_zs", "param_zs_add"
        concat_param_zs_as_opemb_dimension = None,
        concat_param_zs_as_opemb_mlp = [64, 128],
        param_zs_add_coeff = 1.,

        # gradient flow configurations
        detach_vinfo: bool = False,
        updateopemb_detach_opemb: bool = True,
        updateopemb_detach_finfo: bool = True,

        schedule_cfg = None
    ):
        super(NB201FBFlowArchEmbedder, self).__init__(schedule_cfg)

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
        self.init_input_node_emb = init_input_node_emb

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

        if self.symmetry_breaking_method == "param_zs_add":
            in_dim = self.concat_param_zs_as_opemb_dimension
            self.param_zs_embedder = []
            for embedder_dim in concat_param_zs_as_opemb_mlp:
                self.param_zs_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.param_zs_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.param_zs_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.param_zs_embedder = nn.Sequential(*self.param_zs_embedder)
            self.param_zs_add_coeff = param_zs_add_coeff

        ## --- init GATES parts ---
        if self.init_input_node_emb:
            self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        else:
            self.input_node_emb = None

        self.other_node_emb = nn.Parameter(
            torch.zeros(1, self.node_embedding_dim), requires_grad = False
        )

        # the last embedding is the output op emb
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(
                torch.FloatTensor(self.op_embedding_dim).normal_()
            )
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hid_dim)

        if self.share_op_attention:
            assert (
                len(np.unique(self.gcn_out_dims)) == 1
            ), "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        if self.num_time_steps > 1 and "message" in self.updateopemb_method:
            addi_kwargs = {"return_message": True}
            self.use_message = True
        else:
            addi_kwargs = {}
            self.use_message = False

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hid_dim
        gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
        gcn_kwargs.update(addi_kwargs)
        for dim in self.gcn_out_dims:
            self.gcns.append(
                DenseGraphSimpleOpEdgeFlow(
                    in_dim,
                    dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else\
                    (self.op_embedding_dim if not self.share_op_attention else dim),
                    has_attention = not self.share_op_attention,
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

        # --- init TA-GATES parts ---
        if self.num_time_steps > 1:
            # for caculating parameterized op mask, only update the emb of parametrized operations
            self._parametrized_op_emb = [
                [float("conv" in op_name)] for op_name in self.search_space.ops_choices]
            self._parametrized_op_emb = nn.Parameter(
                torch.tensor(self._parametrized_op_emb, dtype = torch.float32), requires_grad = False)

            # init backward graph convolutions
            self.b_gcns = []
            self.b_bns = []
            if self.concat_arch_zs_as_l_dimension is not None \
               and self.concat_l_layer == self.fb_conversion_dims - 1:
                in_dim = self.fb_conversion_dims[-1] + self.concat_arch_zs_as_l_dimension
            else:
                in_dim = self.fb_conversion_dims[-1]
            b_gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
            b_gcn_kwargs.update(addi_kwargs)
            # the final output node (concated all internal nodes in DARTS & NB301)
            for dim in self.backward_gcn_out_dims:
                self.b_gcns.append(DenseGraphSimpleOpEdgeFlow(
                    in_dim, dim,
                    self.op_embedding_dim + self.concat_param_zs_as_opemb_dimension \
                    if symmetry_breaking_method == "param_zs" else self.op_embedding_dim,
                    reverse = True,
                    **(b_gcn_kwargs or {})))
                in_dim = dim
                if self.use_bn:
                    self.b_bns.append(nn.BatchNorm1d(self.vertices))
            self.b_gcns = nn.ModuleList(self.b_gcns)
            if self.b_use_bn:
                self.b_bns = nn.ModuleList(self.b_bns)
            self.num_b_gcn_layers = len(self.b_gcns)

            # init the network to convert forward output info into backward input info
            self.fb_conversion_list = []
            # concat the embedding all cell groups, and then do the f-b conversion
            dim = self.gcn_out_dims[-1]
            num_fb_layers = len(fb_conversion_dims)
            self._num_before_concat_l = None
            for i_dim, fb_conversion_dim in enumerate(fb_conversion_dims):
                self.fb_conversion_list.append(nn.Linear(dim, fb_conversion_dim))
                if i_dim < num_fb_layers - 1:
                    self.fb_conversion_list.append(nn.ReLU(inplace = False))
                if self.concat_arch_zs_as_l_dimension is not None and \
                   self.concat_l_layer == i_dim:
                    dim = fb_conversion_dim + self.concat_arch_zs_as_l_dimension
                    self._num_before_concat_l = len(self.fb_conversion_list)
                else:
                    dim = fb_conversion_dim
            self.fb_conversion = nn.Sequential(*self.fb_conversion_list)

            # init the network to get delta op_emb
            if self.updateopemb_method in {"concat_ofb", "concat_ofb_message"}:
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] + self.op_embedding_dim
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_embedding_dim))
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

    def embed_and_transform_arch(self, archs, input_node_emb):
        adjs = self.op_emb.weight.new(archs).long()
        op_embs = self.op_emb(adjs)  # (batch_size, vertices, vertices, op_emb_dim)
        b_size = op_embs.shape[0]
        if self.init_input_node_emb:
            node_embs = torch.cat(
                (
                    self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1])
                ),
                dim = 1
            )
        else:
            node_embs = torch.cat(
                (
                    input_node_emb.unsqueeze(-2),
                    self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1])
                ),
                dim = 1
            )
        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        return adjs, x, op_embs

    def forward(self, archs, input_node_emb = None):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_l = archs
                zs_as_p = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_embs = self.embed_and_transform_arch(archs, input_node_emb)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.op_emb.weight.new(zs_as_p)
        elif self.symmetry_breaking_method == "param_zs_add":
            zs_as_p = self.op_emb.weight.new(zs_as_p)
            op_embs = op_embs + self.param_zs_add_coeff * self.param_zs_embedder(zs_as_p)

        if self.num_time_steps > 1:
            # calculate op mask
            opemb_update_mask = F.embedding(adjs, self._parametrized_op_emb)
        else:
            opemb_update_mask = None

        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim = -1)
            else:
                auged_op_embs = op_embs

            y, message = self._forward_pass(x, adjs, auged_op_embs)

            if t == self.num_time_steps - 1:
                break

            b_y, b_message = self._backward_pass(y, adjs, zs_as_l, auged_op_embs)
            op_embs = self._update_op_emb(y, b_y, op_embs, message, b_message, opemb_update_mask)

        y = self._final_process(y)
        return y

    def _backward_pass(self, y, adjs, zs_as_l, auged_op_embs) -> Tensor:
        # --- backward pass ---
        b_info = y[:, -1:, :]
        if self.detach_vinfo:
            b_info = b_info.detach()
        if self.concat_arch_zs_as_l_dimension:
            # process before concat l
            b_info = self.fb_conversion[:self._num_before_concat_l](b_info)
            # concat l
            b_info = torch.cat((b_info, zs_as_l.unsqueeze(-2)), dim = -1)
            if not self.concat_l_layer == len(self.fb_converseion_list) - 1:
                # process after concat l
                b_info = self.fb_conversion[self._num_before_concat_l:](b_info)
        else:
            b_info = self.fb_conversion(b_info)
        b_info = torch.cat(
            (
                torch.zeros([y.shape[0], self.vertices - 1, b_info.shape[-1]], device = y.device),
                b_info
            ), dim = 1
        )

        # start backward flow
        b_message = None
        b_adjs = adjs.transpose(1, 2)
        b_y = b_info
        b_op_embs = auged_op_embs.transpose(1, 2)
        for i_layer, gcn in enumerate(self.b_gcns):
            if self.use_message:
                b_y, b_message = gcn(b_y, b_adjs, b_op_embs, self_op_emb = self.self_op_emb)
            else:
                b_y = gcn(b_y, b_adjs, b_op_embs, self_op_emb = self.self_op_emb)
            if self.use_bn:
                shape_y = b_y.shape
                b_y = self.bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
            if i_layer != self.num_gcn_layers - 1:
                b_y = F.relu(b_y)
            b_y = F.dropout(b_y, self.dropout, training = self.training)
        return b_y, b_message

    def _forward_pass(self, x, adjs, auged_op_embs) -> Tensor:
        y = x
        message = None
        for i_layer, gcn in enumerate(self.gcns):
            if self.use_message:
                y, message = gcn(y, adjs, auged_op_embs, self_op_emb = self.self_op_emb)
            else:
                y = gcn(y, adjs, auged_op_embs, self_op_emb = self.self_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training = self.training)

        return y, message

    def _update_op_emb(self, y: Tensor, b_y: Tensor, op_embs: Tensor, message: Tensor, b_message: Tensor, opemb_update_mask: Tensor) -> Tensor:
        # --- UpdateOpEmb ---
        if self.updateopemb_method == "concat_ofb":
            unsqueeze_y = y.unsqueeze(-2).repeat([1, 1, self.vertices, 1])
            unsqueeze_b_y = b_y.unsqueeze(-3).repeat([1, self.vertices, 1, 1])
            in_embedding = torch.cat(
                    (
                        op_embs.detach() if self.updateopemb_detach_opemb else op_embs,
                        unsqueeze_y.detach() if self.updateopemb_detach_finfo else unsqueeze_y,
                        unsqueeze_b_y
                    ), dim = -1
            )
        elif self.updateopemb_method == "concat_ofb_message": # use_message==True
            in_embedding = torch.cat(
                    (
                        op_embs.detach() if self.updateopemb_detach_opemb else op_embs,
                        message.detach() if self.updateopemb_detach_finfo else message,
                        b_message.transpose(1, 2)
                    ), dim = -1
            )
        else:
            raise Exception()

        update = self.updateop_embedder(in_embedding)
        update = update * opemb_update_mask
        op_embs = op_embs + self.updateopemb_scale * update
        return op_embs

    def _final_process(self, y: Tensor) -> Tensor:
        # ---- final output ----
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding
            y = torch.mean(y, dim = 1)  # average across nodes (bs, god)
        return y


class NB201FBFlowAnyTimeArchEmbedder(NB201FBFlowArchEmbedder):
    """
    Implement of TA-GATES anytime architecture embedder on NAS-Bench-201.
    """
    NAME = "nb201-fbflow-anytime"

    def forward(self, archs, input_node_emb = None, any_time: bool = False):
        """
        Feed-forward calculation.

        Args:
            arch: The architectures to be embedded.
            input_node_emb: Default: `None`.
            any_time (bool): Whether use the anytime mode. If `True`, the embedding of every time steps will be returned.
                             Else, only the embedding of the last step will be returned.
        """
        if not any_time:
            return super(NB201FBFlowAnyTimeArchEmbedder, self).forward(archs, input_node_emb)

        if isinstance(archs, tuple):
            if len(archs) == 2:
                archs, zs_as_l = archs
                zs_as_p = None
            elif len(archs) == 3:
                archs, zs_as_l, zs_as_p = archs
            else:
                raise Exception()
        else:
            zs_as_l = zs_as_p = None

        adjs, x, op_embs = self.embed_and_transform_arch(archs, input_node_emb)
        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = self.op_emb.weight.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = self.op_emb.weight.new(zs_as_p)

        if self.num_time_steps > 1:
            # calculate op mask
            opemb_update_mask = F.embedding(adjs, self._parametrized_op_emb)
        else:
            opemb_update_mask = None

        y_cache = []
        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim = -1)
            else:
                auged_op_embs = op_embs

            y, message = self._forward_pass(x, adjs, auged_op_embs)
            y_cache.append(self._final_process(y))

            if t == self.num_time_steps - 1:
                break

            b_y, b_message = self._backward_pass(y, adjs, zs_as_l, auged_op_embs)
            op_embs = self._update_op_emb(y, b_y, op_embs, message, b_message, opemb_update_mask)

        return y_cache


class NasBench201Evaluator(BaseEvaluator):
    NAME = "nasbench-201"

    def __init__(
        self,
        dataset,
        weights_manager,
        objective,
        rollout_type="nasbench-201",
        sample_query=True,
        schedule_cfg=None,
    ):
        super(NasBench201Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type
        )

        self.sample_query = sample_query

    @classmethod
    def supported_data_types(cls):
        # cifar10
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201", "compare", "nasbench-201-differentiable"]

    def suggested_controller_steps_per_epoch(self):
        return None

    def suggested_evaluator_steps_per_epoch(self):
        return None

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
                reward = (results[sampled_index].eval_acc1es["x-valid@199"]) / 100.0
            else:
                reward = (
                    np.mean([res.eval_acc1es["x-valid@199"] for res in results]) / 100.0
                )
            rollout.set_perf(reward, name="reward")
            rollout.set_perf(
                np.mean([res.eval_acc1es["ori-test@199"] for res in results]) / 100.0,
                name="partial_test_acc",
            )
            rollout.set_perf(
                np.mean(
                    [
                        res.eval_acc1es["ori-test@199"]
                        for res in query_res.query("cifar10").values()
                    ]
                )
                / 100.0,
                name="test_acc",
            )

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


class NB201DiffSharedCell(nn.Module):
    def __init__(self, op_cls, search_space, layer_index, 
            num_channels, num_out_channels, stride, bn_affine: bool = False):
        super(NB201DiffSharedCell, self).__init__()
        self.search_space = search_space
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index

        self._vertices = self.search_space.num_vertices
        self._primitives = self.search_space.ops_choices

        self.edges = defaultdict(dict)
        self.edge_mod = torch.nn.Module()  # a stub wrapping module of all the edges
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                self.edges[from_][to_] = op_cls(
                    self.num_channels,
                    self.num_out_channels,
                    stride=self.stride,
                    primitives=self._primitives,
                    bn_affine = bn_affine
                )
                self.edge_mod.add_module(
                    "f_{}_t_{}".format(from_, to_), self.edges[from_][to_]
                )
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
        geno_ind = 0
        for to_ in range(1, self.search_space.num_vertices):
            out = torch.zeros(inputs.shape).to(inputs.device)
            for from_ in range(to_):
                for op in range(self.search_space.num_op_choices):
                    out = (
                        out
                        + self.edges[from_][to_](states_[from_], op)
                        * genotype[geno_ind][0][op]
                    )
                geno_ind += 1
            states_.append(out)
        return states_[-1]


class NB201SharedCell(nn.Module):
    def __init__(
            self, op_cls, search_space, layer_index, num_channels, num_out_channels, stride, bn_affine=False
    ):
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
        self.edge_mod = torch.nn.Module()  # a stub wrapping module of all the edges
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                self.edges[from_][to_] = op_cls(
                    self.num_channels,
                    self.num_out_channels,
                    stride=self.stride,
                    primitives=self._primitives,
                    bn_affine=bn_affine
                )
                self.edge_mod.add_module(
                    "f_{}_t_{}".format(from_, to_), self.edges[from_][to_]
                )
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

    def forward(self, inputs, genotype, **kwargs):
        states_ = [inputs]
        if "valid_input" in kwargs:
            valid_input = kwargs["valid_input"]
        else:
            valid_input = [0]
            for to_ in range(1, self.search_space.num_vertices):
                for input_ in valid_input:
                    if genotype[to_][input_] > 0:
                        valid_input.append(to_)
                        break
        if "valid_output" in kwargs:
            valid_output = kwargs["valid_output"]
        else:
            valid_output = [self.search_space.num_vertices - 1]
            for from_ in range(self.search_space.num_vertices - 2, -1, -1):
                for output_ in valid_output:
                    if genotype[output_][from_] > 0:
                        valid_output.append(from_)

        for to_ in range(1, self._vertices):
            state_ = torch.zeros(inputs.shape).to(inputs.device)
            for from_ in range(to_):
                if from_ in valid_input and to_ in valid_output:
                    out = self.edges[from_][to_](
                        states_[from_], int(genotype[to_][from_])
                    )
                    state_ = state_ + out
            states_.append(state_)
        return states_[-1]

    def sub_named_members(
        self, genotype, prefix="", member="parameters", check_visited=False
    ):
        prefix = prefix + ("." if prefix else "")
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                edge_share_op = self.edges[from_][to_]
                for n, v in edge_share_op.sub_named_members(
                    int(genotype[to_][from_]),
                    prefix=prefix + "edge_mod.f_{}_t_{}".format(from_, to_),
                    member=member,
                ):
                    yield n, v


class NB201SharedOp(nn.Module):
    """
    The operation on an edge, consisting of multiple primitives.
    """

    def __init__(self, C, C_out, stride, primitives, bn_affine=False):
        super(NB201SharedOp, self).__init__()
        self.primitives = primitives
        self.stride = stride
        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = ops.get_op(primitive)(C, C_out, stride, affine=bn_affine)
            self.p_ops.append(op)

    def forward(self, x, op_type):
        return self.p_ops[op_type](x)

    def sub_named_members(self, op_type, prefix="", member="parameters"):
        prefix = prefix + ("." if prefix else "")
        for n, v in getattr(self.p_ops[op_type], "named_" + member)(
            prefix="{}p_ops.{}".format(prefix, op_type)
        ):
            yield n, v


class NB201CandidateNet(CandidateNet):
    def __init__(
        self,
        super_net,
        rollout,
        member_mask,
        gpus=tuple(),
        cache_named_members=False,
        eval_no_grad=True
    ):
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

        # get valid input/output
        valid_input = [0]
        for to_ in range(1, self.search_space.num_vertices):
            for input_ in valid_input:
                if self.genotype_arch[to_][input_] > 0:
                    valid_input.append(to_)
                    break
        valid_output = [self.search_space.num_vertices - 1]
        for from_ in range(self.search_space.num_vertices - 2, -1, -1):
            for output_ in valid_output:
                if self.genotype_arch[output_][from_] > 0:
                    valid_output.append(from_)
        self.valid_input = set(valid_input)
        self.valid_output = set(valid_output)
        
    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0
        self.super_net.reset_flops()

    def get_device(self):
        return self._device

    def _forward(self, inputs, **kwargs):
        detach_arch = kwargs.get("detach_arch", False)
        if detach_arch:
            arch = [
                DiffArch(op_weights=op_weights.detach(), edge_norms=None)
                for op_weights, edge_norms in self.genotype_arch
            ]
        else:
            arch = self.genotype_arch
        return self.super_net.forward(
            inputs, arch, valid_input=self.valid_input, valid_output=self.valid_output)

    def forward(self, inputs, single=False, **kwargs):  # pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs, **kwargs)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        module_kwargs = {"single": True}
        module_kwargs.update(kwargs)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs=module_kwargs)

    def _forward_with_params(
        self, inputs, params, **kwargs
    ):  # pylint: disable=arguments-differ
        with use_params(self.super_net, params):
            return self.forward(inputs, **kwargs)

    def plot_arch(self):
        return self.super_net.search_space.plot_arch(
            self.genotype, "./nb201_search", "nb201_search"
        )

    def named_parameters(
        self, prefix="", recurse=True
    ):  # pylint: disable=arguments-differ
        if isinstance(self.super_net, NB201DiffSharedNet):
            # return all named parameters
            return self.super_net.named_parameters()

        if self.member_mask:
            if self.cache_named_members:
                # use cached members
                if self._cached_np is None:
                    self._cached_np = []
                    for n, v in self.active_named_members(
                        member="parameters", prefix=""
                    ):
                        self._cached_np.append((n, v))
                prefix = prefix + ("." if prefix else "")
                for n, v in self._cached_np:
                    yield prefix + n, v
            else:
                for n, v in self.active_named_members(
                    member="parameters", prefix=prefix
                ):
                    yield n, v
        else:
            for n, v in self.super_net.named_parameters(prefix=prefix):
                yield n, v

    def named_buffers(
        self, prefix="", recurse=True
    ):  # pylint: disable=arguments-differ
        if isinstance(self.super_net, NB201DiffSharedNet):
            return self.super_net.named_buffers()

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

    def active_named_members(
        self, member, prefix="", recurse=True, check_visited=False
    ):
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
        for n, v in self.super_net.sub_named_members(
            self.genotype_arch,
            prefix=prefix,
            member=member,
            check_visited=check_visited,
        ):
            if v in memo:
                continue
            memo.add(v)
            yield n, v

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        member_lst = []
        for n, v in itertools.chain(
            self.active_named_members(member="parameters", prefix=""),
            self.active_named_members(member="buffers", prefix=""),
        ):
            member_lst.append((n, v))
        state_dict = OrderedDict(member_lst)
        return state_dict


class NB201DiffCandidateNet(NB201CandidateNet):
    def __init__(
        self,
        super_net,
        rollout,
        member_mask,
        gpus=tuple(),
        cache_named_members=False,
        eval_no_grad=True
    ):
        CandidateNet.__init__(self, eval_no_grad = eval_no_grad)
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

        self.virtual_parameter_only = False

    def forward(self, inputs, single=False, **kwargs):  # pylint: disable=arguments-differ
        if single or not self.gpus or len(self.gpus) == 1:
            detach_arch = kwargs.get("detach_arch", False)
            if detach_arch:
                arch = [
                    DiffArch(op_weights=op_weights.detach(), edge_norms=None)
                    for op_weights, edge_norms in self.genotype_arch
                ]
            else:
                arch = self.genotype_arch
            return self.super_net.forward(inputs, arch)
        # return data_parallel(self.super_net, (inputs, self.genotypes_grouped), self.gpus)
        module_kwargs = {"single": True}
        module_kwargs.update(kwargs)
        return data_parallel(self, (inputs,), self.gpus, module_kwargs=module_kwargs)

    def _forward_with_params(
        self, inputs, params, single, **kwargs
    ):  # pylint: disable=arguments-differ
        with use_params(self.super_net, params):
            return self.forward(inputs, single, **kwargs)
    
    @contextlib.contextmanager
    def begin_virtual(self):
        w_clone = {k: v.clone() for k, v in self.named_parameters()}
        if not self.virtual_parameter_only:
            buffer_clone = {k: v.clone() for k, v in self.named_buffers()}

        yield

        for n, v in self.named_parameters():
            v.data.copy_(w_clone[n])
        del w_clone

        if not self.virtual_parameter_only:
            for n, v in self.named_buffers():
                v.data.copy_(buffer_clone[n])
            del buffer_clone


class BaseNB201SharedNet(BaseWeightsManager, nn.Module):
    def __init__(
        self,
        search_space,
        device,
        cell_cls,
        op_cls,
        rollout_type="nasbench-201",
        gpus=tuple(),
        num_classes=10,
        init_channels=16,
        stem_multiplier=1,
        max_grad_norm=5.0,
        dropout_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        reduce_affine=True,
        cell_bn_affine=False,
        candidate_member_mask=True,
        candidate_cache_named_members=False,
        candidate_eval_no_grad=True,
        iso_mapping_file=None
    ):
        super(BaseNB201SharedNet, self).__init__(search_space, device, rollout_type)
        nn.Module.__init__(self)

        if iso_mapping_file is not None:
            with open(iso_mapping_file, "r") as rf:
                self.iso_mapping = yaml.load(rf)
        else:
            self.iso_mapping = None

        # optionally data parallelism in SharedNet
        self.gpus = gpus
        _check_support_candidate_member_mask(self.gpus, candidate_member_mask, self.NAME)

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
                self.stems.append(
                    ops.get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )
            self.stems = nn.ModuleList(self.stems)
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        strides = [
            2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)
        ]

        for i_layer, stride in enumerate(strides):
            _num_channels = num_channels if i_layer != 0 else c_stem
            if stride > 1:
                num_channels *= stride
            # A patch: Can specificy input/output channels by hand in configuration,
            # instead of relying on the default
            # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
            _num_out_channels = num_channels
            if stride == 1:
                cell = cell_cls(
                    op_cls,
                    self.search_space,
                    layer_index=i_layer,
                    num_channels=_num_channels,
                    num_out_channels=_num_out_channels,
                    stride=stride,
                    bn_affine=cell_bn_affine
                )
            else:
                cell = ops.get_op("NB201ResidualBlock")(
                    _num_channels, _num_out_channels, stride=2, affine=reduce_affine
                )
            self.cells.append(cell)
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True)
        )
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
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += (
                    inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * inputs[0].size(2)
                    * inputs[0].size(3)
                    / (module.stride[0] * module.stride[1] * module.groups)
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def sub_named_members(
        self, genotype, prefix="", member="parameters", check_visited=False
    ):
        prefix = prefix + ("." if prefix else "")

        # the common modules that will be forwarded by every candidate
        for mod_name, mod in six.iteritems(self._modules):
            if mod_name == "cells":
                continue
            _func = getattr(mod, "named_" + member)
            for n, v in _func(prefix=prefix + mod_name):
                yield n, v

        for cell_idx, cell in enumerate(self.cells):
            for n, v in cell.sub_named_members(
                genotype,
                prefix=prefix + "cells.{}".format(cell_idx),
                member=member,
                check_visited=check_visited,
            ):
                yield n, v

    # ---- APIs ----
    def assemble_candidate(self, rollout, **kwargs):
        return NB201CandidateNet(
            self,
            rollout,
            gpus=self.gpus,
            member_mask=self.candidate_member_mask,
            cache_named_members=self.candidate_cache_named_members,
            eval_no_grad=self.candidate_eval_no_grad,
        )

    @classmethod
    def supported_rollout_types(cls):
        return ["nasbench-201", "nasbench-201-differentiable"]

    def _is_reduce(self, layer_idx):
        return layer_idx in [
            (self._num_layers + 1) // 3 - 1,
            (self._num_layers + 1) * 2 // 3 - 1,
        ]

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, inputs, genotype, **kwargs):  # pylint: disable=arguments-differ
        if self.iso_mapping:
            # map to the representing arch
            genotype = self.search_space.rollout_from_genotype(
                self.iso_mapping[NasBench201Rollout(
                    genotype, search_space=self.search_space).genotype]).arch
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
        self.zero_grad()  # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
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


class NB201SharedNet(BaseNB201SharedNet):
    NAME = "nasbench-201"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-201",
        gpus=tuple(),
        num_classes=10,
        init_channels=16,
        stem_multiplier=1,
        max_grad_norm=5.0,
        dropout_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        reduce_affine=True,
        cell_bn_affine=False,
        candidate_member_mask=True,
        candidate_cache_named_members=False,
        candidate_eval_no_grad=True,
        iso_mapping_file=None
    ):
        super(NB201SharedNet, self).__init__(
            search_space,
            device,
            cell_cls=NB201SharedCell,
            op_cls=NB201SharedOp,
            rollout_type=rollout_type,
            gpus=gpus,
            num_classes=num_classes,
            init_channels=init_channels,
            stem_multiplier=stem_multiplier,
            max_grad_norm=max_grad_norm,
            dropout_rate=dropout_rate,
            use_stem=use_stem,
            stem_stride=stem_stride,
            stem_affine=stem_affine,
            reduce_affine=reduce_affine,
            cell_bn_affine=cell_bn_affine,
            candidate_member_mask=candidate_member_mask,
            candidate_cache_named_members=candidate_cache_named_members,
            candidate_eval_no_grad=candidate_eval_no_grad,
            iso_mapping_file=iso_mapping_file
        )


class NB201DiffSharedNet(BaseNB201SharedNet):
    NAME = "nasbench-201-diff-supernet"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="nasbench-201-differentiable",
        gpus=tuple(),
        num_classes=10,
        init_channels=16,
        stem_multiplier=1,
        max_grad_norm=5.0,
        dropout_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        reduce_affine=True,
        candidate_member_mask=True,
        candidate_cache_named_members=False,
        candidate_eval_no_grad=True,
    ):
        super(NB201DiffSharedNet, self).__init__(
            search_space,
            device,
            cell_cls=NB201DiffSharedCell,
            op_cls=NB201SharedOp,
            rollout_type=rollout_type,
            gpus=gpus,
            num_classes=num_classes,
            init_channels=init_channels,
            stem_multiplier=stem_multiplier,
            max_grad_norm=max_grad_norm,
            dropout_rate=dropout_rate,
            use_stem=use_stem,
            stem_stride=stem_stride,
            stem_affine=stem_affine,
            reduce_affine=reduce_affine,
            candidate_member_mask=candidate_member_mask,
            candidate_cache_named_members=candidate_cache_named_members,
            candidate_eval_no_grad=candidate_eval_no_grad,
        )

    def assemble_candidate(self, rollout) -> NB201DiffCandidateNet:
        return NB201DiffCandidateNet(self, rollout, gpus = self.gpus,
                                     member_mask = self.candidate_member_mask,
                                     cache_named_members = self.candidate_cache_named_members,
                                     eval_no_grad = self.candidate_eval_no_grad
        )



class NB201GenotypeModel(FinalModel):
    NAME = "nb201_final_model"

    SCHEDULABLE_ATTRS = ["dropout_path_rate"]

    def __init__(
        self,
        search_space,
        device,
        genotypes,
        num_classes=10,
        init_channels=36,
        stem_multiplier=1,
        dropout_rate=0.1,
        dropout_path_rate=0.2,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        reduce_affine=True,
        schedule_cfg=None,
    ):
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
                self.stems.append(
                    ops.get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )
            self.stems = nn.ModuleList(self.stems)
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        strides = [
            2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)
        ]
        for i_layer, stride in enumerate(strides):
            _num_channels = num_channels if i_layer != 0 else c_stem
            if stride > 1:
                num_channels *= stride
            _num_out_channels = num_channels
            if stride == 1:
                cell = NB201GenotypeCell(
                    self.search_space,
                    self.genotype_arch,
                    layer_index=i_layer,
                    num_channels=_num_channels,
                    num_out_channels=_num_out_channels,
                    stride=stride,
                )
            else:
                cell = ops.get_op("NB201ResidualBlock")(
                    _num_channels, _num_out_channels, stride=2, affine=reduce_affine
                )
            # TODO: support specify concat explicitly
            self.cells.append(cell)

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(num_channels, self.num_classes)
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def set_hook(self):
        for name, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += (
                    2
                    * inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * outputs.size(2)
                    * outputs.size(3)
                    / module.groups
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def forward(self, inputs):  # pylint: disable=arguments-differ
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

        out = self.lastact(states)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))

        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops / 1.0e6)
            self._flops_calculated = True

        return logits

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _is_reduce(self, layer_idx):
        return layer_idx in [
            (self._num_layers + 1) // 3 - 1,
            (self._num_layers + 1) * 2 // 3 - 1,
        ]


class NB201GenotypeCell(nn.Module):
    def __init__(
        self,
        search_space,
        genotype_arch,
        layer_index,
        num_channels,
        num_out_channels,
        stride,
        bn_affine=False
    ):
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
        self.edge_mod = torch.nn.Module()  # a stub wrapping module of all the edges
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                self.edges[from_][to_] = ops.get_op(
                    self._primitives[int(self.arch[to_][from_])]
                )(
                    self.num_channels,
                    self.num_out_channels,
                    stride=self.stride,
                    affine=bn_affine
                )

                self.edge_mod.add_module(
                    "f_{}_t_{}".format(from_, to_), self.edges[from_][to_]
                )
        self._edge_name_pattern = re.compile("f_([0-9]+)_t_([0-9]+)")

    def forward(self, inputs, dropout_path_rate):  # pylint: disable=arguments-differ
        states = [inputs]

        for to_ in range(1, self._vertices):
            state_to_ = 0.0
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
            from_, to_ = self._edge_name_pattern.match(edge_name).groups()
            self.edges[int(from_)][int(to_)] = edge_mod
