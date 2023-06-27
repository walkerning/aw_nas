"""
NASBench-301 search space (a `cnn` search space), evaluator (API query)
"""

import copy
from collections import namedtuple
from typing import List

from ConfigSpace.read_and_write import json as cs_json
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import nasbench301 as nb

from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.common import CNNSearchSpace
from aw_nas.rollout.base import Rollout
from aw_nas.utils import DenseGraphSimpleOpEdgeFlow


class NB301SearchSpace(CNNSearchSpace):
    NAME = "nb301"

    def __init__(self, shared_primitives=(
            "sep_conv_3x3",
            "sep_conv_5x5",
            "dil_conv_3x3",
            "dil_conv_5x5",
            "max_pool_3x3",
            "avg_pool_3x3",
            "skip_connect")):
        super(NB301SearchSpace, self).__init__(
            num_cell_groups=2,
            num_init_nodes=2,
            num_layers=8, cell_layout=None,
            reduce_cell_groups=(1,),
            num_steps=4, num_node_inputs=2,
            concat_op="concat",
            concat_nodes=None,
            loose_end=False,
            shared_primitives=shared_primitives,
            derive_without_none_op=False)

    def mutate(self, rollout, node_mutate_prob=0.5):
        # NOTE: NB301 search space does not permit double connections between the same pair of nodes
        new_arch = copy.deepcopy(rollout.arch)
        # randomly select a cell gorup to modify
        mutate_i_cg = np.random.randint(0, self.num_cell_groups)
        num_prims = self._num_primitives \
                    if not self.cellwise_primitives else self._num_primitives_list[mutate_i_cg]
        _num_step = self.get_num_steps(mutate_i_cg)
        if np.random.random() < node_mutate_prob:
            # mutate connection
            # no need to mutate the connection to node 2, since no double connection is permited
            start = self.num_node_inputs
            node_mutate_idx = np.random.randint(start, len(new_arch[mutate_i_cg][0]))
            i_out = node_mutate_idx // self.num_node_inputs
            # self.num_node_inputs == 2
            # if node_mutate_idx is odd/even,
            # node_mutate_idx-1/+1 corresponds the other input node choice, respectively
            the_other_idx = node_mutate_idx - (2 * (node_mutate_idx % self.num_node_inputs) - 1)
            should_not_repeat = new_arch[mutate_i_cg][0][the_other_idx]
            out_node = i_out + self.num_init_nodes
            offset = np.random.randint(1, out_node - 1)
            old_choice = new_arch[mutate_i_cg][0][node_mutate_idx]
            new_choice = old_choice + offset # before mod
            tmp_should_not_repeat = should_not_repeat + out_node if should_not_repeat < old_choice \
                                    else should_not_repeat
            if tmp_should_not_repeat <= new_choice:
                # Should add 1 to the new choice
                new_choice += 1
            new_arch[mutate_i_cg][0][node_mutate_idx] = new_choice % out_node
        else:
            # mutate op
            op_mutate_idx = np.random.randint(0, len(new_arch[mutate_i_cg][1]))
            offset = np.random.randint(1, num_prims)
            new_arch[mutate_i_cg][1][op_mutate_idx] = \
                    (new_arch[mutate_i_cg][1][op_mutate_idx] + offset) % num_prims
        return Rollout(new_arch, info={}, search_space=self)

    def random_sample(self):
        """
        Random sample a discrete architecture.
        """
        return NB301Rollout(NB301Rollout.random_sample_arch(
            self.num_cell_groups, # =2
            self.num_steps, #=4
            self.num_init_nodes, # =2
            self.num_node_inputs, # =2
            self._num_primitives),
                            info={}, search_space=self)

    def canonicalize(self, rollout):
        # TODO
        archs = rollout.arch
        ss = rollout.search_space
        num_groups = ss.num_cell_groups
        num_vertices = ss.num_steps
        num_node_inputs = ss.num_node_inputs
        Res = ""

        for i_cg in range(num_groups):
            prims = ss.shared_primitives
            S = []
            outS = []
            S.append("0")
            S.append("1")
            arch = archs[i_cg]
            res = ""
            index = 0
            nodes = arch[0]
            ops = arch[1]
            for i_out in range(num_vertices):
                preS = []
                s = ""
                for i in range(num_node_inputs):
                    if (ops[index] == 6):
                        s = S[nodes[index]]
                    else:
                        s = "(" + S[nodes[index]] + ")" + "@" + prims[ops[index]]
                    preS.append(s)
                    index = index + 1
                preS.sort()
                s = ""
                for i in range(num_node_inputs):
                    s = s + preS[i]
                S.append(s)
                outS.append(s)
            outS.sort()
            for i_out in range(num_vertices):
                res = res + outS[i_out]
            if (i_cg < num_groups - 1):
                Res = Res + res + "&"
            else:
                Res = Res + res

        return Res

    @classmethod
    def supported_rollout_types(cls):
        return ["nb301"]


class NB301Rollout(Rollout):
    """
    For one target node, do not permit choose the same from node.
    """
    NAME = "nb301"
    supported_components = [
        ("trainer", "simple"),
        ("evaluator", "mepa"),
        ("weights_manager", "supernet")
    ]

    @classmethod
    def random_sample_arch(cls, num_cell_groups, num_steps,
                           num_init_nodes, num_node_inputs, num_primitives):
        """
        random sample
        """
        arch = []
        for i_cg in range(num_cell_groups):
            num_prims = num_primitives if isinstance(num_primitives, int) else num_primitives[i_cg]
            nodes = []
            ops = []
            _num_step = num_steps if isinstance(num_steps, int) else num_steps[i_cg]
            for i_out in range(_num_step):
                nodes += list(np.random.choice(
                    range(i_out + num_init_nodes), num_node_inputs, replace=False))
                ops += list(np.random.randint(
                    0, high=num_prims, size=num_node_inputs))
            arch.append((nodes, ops))
        return arch


class NB301Evaluator(BaseEvaluator):
    NAME = "nb301"

    def __init__(
            self,
            dataset,
            weights_manager,
            objective,
            rollout_type="nb301",
            with_noise=True,
            path=None,
            schedule_cfg=None,
    ):
        super(NB301Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type
        )

        assert path is not None, "must specify benchmark path"
        self.path = path
        self.with_noise = with_noise
        self.perf_model = nb.load_ensemble(path)
        self.genotype_type = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

    @classmethod
    def supported_data_types(cls):
        # cifar10
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["nb301"]

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
        prims = rollouts[0].search_space.shared_primitives
        for rollout in rollouts:
            # 2 cell groups: normal, reduce
            normal_arch, reduce_arch = [[
                (prims[op_idx], from_idx)
                for from_idx, op_idx in zip(*cell_arch)]
                                        for cell_arch in rollout.arch]
            genotype_config = self.genotype_type(
                normal=normal_arch, normal_concat=[2, 3, 4, 5],
                reduce=reduce_arch, reduce_concat=[2, 3, 4, 5])
            reward = self.perf_model.predict(
                config=genotype_config, representation="genotype",
                with_noise=self.with_noise)
            rollout.set_perf(reward, name="reward")
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


class NB301FBFlowArchEmbedder(ArchEmbedder):
    """
    Implement of TA-GATES architecture embedder on NAS-Bench-301.
    """
    NAME = "cellss-fbflow"

    def __init__(
        self,
        search_space,
        node_dim: int = 48,
        op_dim: int = 48,
        hidden_dim: int = 48,
        gcn_out_dims: List[int] = [128, 128, 128, 128, 128],
        other_node_zero: bool = True,
        gcn_kwargs: dict = None,
        dropout: float = 0.,
        normalize: bool = False,
        use_bn: bool = False,
        other_node_independent: bool = False,
        share_self_op_emb: bool = False,
        final_concat: bool = False,

        ## newly added
        take_adj_as_input: bool = True,
        # if take adj as input, the `share_skip_and_outskip` configuration
        # should be set correctly manually corresponding to the adj generation script

        # construction configurations
        # construction (tagates)
        num_time_steps: int = 2,
        fb_conversion_dims: List[int] = [128, 128],
        backward_gcn_out_dims: List[int] = [128, 128, 128, 128, 128],
        updateopemb_method: str = "concat_ofb_message", # concat_ofb_message, concat_ofb
        updateopemb_scale: float = 0.1,
        updateopemb_dims: List[int] = [128],
        mask_nonparametrized_ops: bool = True,
        b_use_bn: bool = False,
        share_skip_and_outskip: bool = False,
        # construction (l): concat arch-level zeroshot as l
        concat_arch_zs_as_l_dimension = None,
        concat_l_layer: int = 0,
        # construction (symmetry breaking)
        symmetry_breaking_method: str = None, # None, "random", "param_zs", "param_zs_add"
        concat_param_zs_as_opemb_dimension = None,
        concat_param_zs_as_opemb_mlp: List[int] = [64, 128],
        concat_param_zs_as_opemb_scale: float = 0.1,

        # gradient flow configurations
        detach_vinfo: bool = False,
        updateopemb_detach_opemb: bool = True,
        updateopemb_detach_finfo: bool = True,

        schedule_cfg = None
    ) -> None:
        super(NB301FBFlowArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.normalize = normalize
        self.node_dim = node_dim
        self.op_dim = op_dim
        self.hidden_dim = hidden_dim
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self.use_bn = use_bn
        self.other_node_independent = other_node_independent
        self.share_self_op_emb = share_self_op_emb
        # final concat only support the cell-ss that all nodes are concated
        # (loose-end is not supported)
        self.final_concat = final_concat

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
        self.mask_nonparametrized_ops = mask_nonparametrized_ops
        self.share_skip_and_outskip = share_skip_and_outskip
        self.take_adj_as_input = take_adj_as_input
        if self.take_adj_as_input:
            print("NOTE!!! Take adj as input. Make sure the `share_skip_and_outskip` configuration"
                  " is set correctly manually corresponding to the adjacent matrix generation script.")

        # concat arch-level zs as l
        self.concat_arch_zs_as_l_dimension = concat_arch_zs_as_l_dimension
        self.concat_l_layer = concat_l_layer
        if self.concat_arch_zs_as_l_dimension is not None:
            assert self.concat_l_layer < len(self.fb_conversion_dims) - 1
            # for nb301, cannot concat at the final layer
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
            self.param_zs_embedder.append(nn.Linear(in_dim, self.op_dim))
            self.param_zs_embedder = nn.Sequential(*self.param_zs_embedder)

        self._num_init_nodes = self.search_space.num_init_nodes
        self._num_node_inputs = self.search_space.num_node_inputs
        self._num_steps = self.search_space.num_steps
        self._num_nodes = self._num_steps + self._num_init_nodes
        self._num_cg = self.search_space.num_cell_groups
        self._num_all_nodes = self._num_nodes + 1

        # different init node embedding for different cell groups
        # but share op embedding for different cell groups
        # maybe this should be separated? at least for stride-2 op and stride-1 op
        if self.other_node_independent:
            self.init_node_emb = nn.Parameter(torch.Tensor(
                self._num_cg, self._num_nodes, self.node_dim).normal_())
        else:
            # other nodes share init embedding
            self.init_node_emb = nn.Parameter(torch.Tensor(self._num_cg, self._num_init_nodes,
                                                           self.node_dim).normal_())
            self.other_node_emb = nn.Parameter(torch.zeros(self._num_cg, 1, self.node_dim),
                                               requires_grad = not other_node_zero)

        self.num_ops = len(self.search_space.shared_primitives)
        try:
            self.none_index = self.search_space.shared_primitives.index("none")
            self.add_none_index = False
            assert self.none_index == 0, \
                "search space with none op should have none op as the first primitive"
        except ValueError:
            self.none_index = 0
            self.add_none_index = True
            self.num_ops += 1

        # Add a special op for the output node (concat in DARTS & NB301 & ENAS ...)
        if self.share_skip_and_outskip:
            try:
                self.outskip_index = self.search_space.shared_primitives.index("skip_connect") \
                                     + int(self.add_none_index)
            except ValueError:
                self.outskip_index = self.num_ops
                self.num_ops += 1
        else:
            # Add the special op
            self.outskip_index = self.num_ops
            self.num_ops += 1
        print("outskip index: {}; skip index: {}".format(
            self.outskip_index,
            self.search_space.shared_primitives.index("skip_connect") + int(self.add_none_index)))

        self.op_emb = []
        for idx in range(self.num_ops):
            if idx == self.none_index:
                emb = nn.Parameter(torch.zeros(self.op_dim), requires_grad = False)
            elif idx == self.outskip_index:
                emb = nn.Parameter(torch.zeros(self.op_dim), requires_grad = False)
            else:
                emb = nn.Parameter(torch.Tensor(self.op_dim).normal_())
            setattr(self, "op_embedding_{}".format(idx), emb)
            self.op_emb.append(emb)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(self.op_dim).normal_())
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_dim, self.hidden_dim)

        if self.num_time_steps > 1 and "message" in self.updateopemb_method:
            addi_kwargs = {"return_message": True}
            self.use_message = True
        else:
            addi_kwargs = {}
            self.use_message = False

        # for caculating parameterized op mask, only update the emb of parametrized operations
        if self.mask_nonparametrized_ops:
            self._parametrized_op_emb = [
                [float("conv" in op_name)] for op_name in self.search_space.shared_primitives]
        else:
            self._parametrized_op_emb = [
                [float("none" not in op_name)] for op_name in self.search_space.shared_primitives]

        if self.add_none_index: # add none
            self._parametrized_op_emb.insert(0, [0.])
        if self.num_ops > len(self._parametrized_op_emb): # add special
            self._parametrized_op_emb.append([0.])
        self._parametrized_op_emb = nn.Parameter(
            torch.tensor(self._parametrized_op_emb, dtype = torch.float32), requires_grad = False)

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hidden_dim
        f_gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
        f_gcn_kwargs["skip_connection_index"] = self.outskip_index
        f_gcn_kwargs.update(addi_kwargs)
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphSimpleOpEdgeFlow(
                in_dim, dim,
                self.op_dim + self.concat_param_zs_as_opemb_dimension \
                if symmetry_breaking_method == "param_zs" else\
                self.op_dim,
                **(f_gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self._num_nodes * self._num_cg))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)

        # init backward graph convolutions
        self.b_gcns = []
        self.b_bns = []
        in_dim = self.fb_conversion_dims[-1]
        b_gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
        b_gcn_kwargs["residual_only"] = 1
        b_gcn_kwargs["skip_connection_index"] = self.outskip_index
        b_gcn_kwargs.update(addi_kwargs)
        # the final output node (concated all internal nodes in DARTS & NB301)
        for dim in self.backward_gcn_out_dims:
            self.b_gcns.append(DenseGraphSimpleOpEdgeFlow(
                in_dim, dim,
                self.op_dim + self.concat_param_zs_as_opemb_dimension \
                if symmetry_breaking_method == "param_zs" else self.op_dim,
                reverse = True,
                **(b_gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.b_bns.append(nn.BatchNorm1d(self._num_nodes * self._num_cg))
        self.b_gcns = nn.ModuleList(self.b_gcns)
        if self.b_use_bn:
            self.b_bns = nn.ModuleList(self.b_bns)
        self.num_b_gcn_layers = len(self.b_gcns)

        # init the network to convert forward output info into backward input info
        if self.num_time_steps > 1:
            self.fb_conversion_list = []
            # concat the embedding all cell groups, and then do the f-b conversion
            dim = self.gcn_out_dims[-1] * self._num_cg
            num_fb_layers = len(fb_conversion_dims)
            self._num_before_concat_l = None
            for i_dim, fb_conversion_dim in enumerate(fb_conversion_dims):
                self.fb_conversion_list.append(nn.Linear(dim, fb_conversion_dim * self._num_cg))
                if i_dim < num_fb_layers - 1:
                    self.fb_conversion_list.append(nn.ReLU(inplace=False))
                if self.concat_arch_zs_as_l_dimension is not None and \
                   self.concat_l_layer == i_dim:
                    dim = fb_conversion_dim + self.concat_arch_zs_as_l_dimension
                    self._num_before_concat_l = len(self.fb_conversion_list)
                else:
                    dim = fb_conversion_dim
                dim = fb_conversion_dim * self._num_cg
            self.fb_conversion = nn.Sequential(*self.fb_conversion_list)

            # init the network to get delta op_emb
            if self.updateopemb_method in {"concat_ofb", "concat_ofb_message"}:
                in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1] \
                         + self.op_dim
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace = False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_dim))
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

        # output dimension
        if not self.final_concat:
            self.out_dim = self._num_cg * self.gcn_out_dims[-1]
        else:
            self.out_dim = self._num_cg * self.gcn_out_dims[-1] * self._num_steps

    def get_adj_dense(self, arch):
        return self._get_adj_dense(arch, self._num_init_nodes,
                                   self._num_node_inputs, self._num_nodes, self.none_index)

    def _get_adj_dense(self, arch, num_init_nodes, num_node_inputs, num_nodes, none_index): #pylint: disable=no-self-use
        """
        get dense adjecent matrix, could be batched
        """
        f_nodes = np.array(arch[:, 0, :])
        # n_d: input degree (num_node_inputs)
        # ops: (b_size * n_cg, n_steps * n_d)
        ops = np.array(arch[:, 1, :])
        if self.add_none_index:
            ops = ops + 1
        _ndim = f_nodes.ndim
        if _ndim == 1:
            f_nodes = np.expand_dims(f_nodes, 0)
            ops = np.expand_dims(ops, 0)
        else:
            assert _ndim == 2
        batch_size = f_nodes.shape[0]
        t_nodes = np.tile(
            np.repeat(np.array(range(num_init_nodes, num_nodes)), num_node_inputs)[None, :],
            [batch_size, 1]
        )
        batch_inds = np.tile(np.arange(batch_size)[:, None], [1, t_nodes.shape[1]])
        ori_indexes = np.stack((batch_inds, t_nodes, f_nodes))
        adj = torch.zeros((batch_size, num_nodes+1, num_nodes+1), dtype=torch.long)
        adj[ori_indexes] = torch.tensor(ops)
        # For NB301 & DARTS, connect the internal nodes to the output node (ENAS need other handling)
        adj[:, num_nodes, num_init_nodes:num_nodes] = self.outskip_index
        if _ndim == 1:
            adj = adj[0]
        return adj

    def embed_and_transform_arch(self, archs):
        if isinstance(archs, (np.ndarray, list, tuple)):
            archs = np.array(archs)
            if archs.ndim == 3:
                # one arch
                archs = np.expand_dims(archs, 0)
            else:
                if not archs.ndim == 4:
                    import ipdb
                    ipdb.set_trace()
                assert archs.ndim == 4

        # get adjacent matrix
        # sparse
        # archs[:, :, 0, :]: (batch_size, num_cell_groups, num_node_inputs * num_steps)
        # adjs, adj_op_inds_lst = self.get_adj_dense(archs.reshape(b_size * n_cg, 2, n_edge))
        if not self.take_adj_as_input:
            b_size, n_cg, _, n_edge = archs.shape
            adjs = self.get_adj_dense(archs.reshape(b_size * n_cg, 2, n_edge))
        else:
            b_size, n_cg, _, _ = archs.shape
            adjs = torch.tensor(archs, dtype=torch.long)
        adjs = adjs.reshape([b_size, n_cg, adjs.shape[-1], adjs.shape[-1]]).to(
            self.init_node_emb.device)

        # operation embedding
        op_embs = F.embedding(adjs, torch.stack(self.op_emb))

        # embedding of init nodes
        # TODO: output op should have a embedding maybe? (especially for hierarchical purpose)
        if self.other_node_independent:
            node_embs = self.init_node_emb.unsqueeze(0).repeat(b_size, 1, 1, 1)
        else:
            node_embs = torch.cat(
                (self.init_node_emb.unsqueeze(0).repeat(b_size, 1, 1, 1),
                 # self.other_node_emb.unsqueeze(0).repeat(b_size, 1, self._num_steps, 1)),
                 # add a special node (output concat)
                 self.other_node_emb.unsqueeze(0).repeat(b_size, 1, self._num_steps+1, 1)),
                dim=2)
        # (batch_size, num_cell_groups, num_nodes, self.node_dim)

        x = self.x_hidden(node_embs)
        # (batch_size, num_cell_groups, num_nodes, op_hid)
        return adjs, x, op_embs

    def _forward_pass(self, x, adjs, auged_op_embs) -> Tensor:
        y = x
        message = None

        for i_layer, gcn in enumerate(self.gcns):
            # outskip is handled by specifying `skip_connection_index`
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

    def _backward_pass(self, y, adjs, zs_as_l, auged_op_embs, batch_size):
        # --- backward pass ---
        # use the special final aggregation node
        b_message = None

        v_info_n = y[:, self._num_nodes, :].reshape(
            batch_size, self._num_cg * y.shape[-1])
        if self.detach_vinfo:
            v_info_n = v_info_n.detach()
        if self.concat_arch_zs_as_l_dimension:
            # process before concat l
            v_info_n = self.fb_conversion[:self._num_before_concat_l](v_info_n)
            # concat l
            v_info_n = torch.cat((v_info_n, zs_as_l.unsqueeze(-2)), dim = -1)
            assert not self.concat_l_layer == len(self.fb_converseion_list) - 1
            # process after concat l
            v_info_n = self.fb_conversion[self._num_before_concat_l:](v_info_n)
        else:
            v_info_n = self.fb_conversion(v_info_n)
        v_info_n = v_info_n.reshape(batch_size * self._num_cg, 1, -1)
        v_info_n = torch.cat(
            (
                torch.zeros([batch_size * self._num_cg, self._num_nodes, v_info_n.shape[-1]],
                            device = y.device),
                v_info_n
            ), dim = 1
        )

        # start backward flow
        b_adjs = adjs.transpose(1, 2)
        b_y = v_info_n
        b_op_embs = auged_op_embs.transpose(1, 2)
        for i_layer, gcn in enumerate(self.b_gcns):
            # outskip is handled by specifying `skip_connection_index`
            if self.use_message:
                b_y, b_message = gcn(b_y, b_adjs, b_op_embs, self_op_emb = self.self_op_emb)
            else:
                b_y = gcn(b_y, b_adjs, b_op_embs, self_op_emb = self.self_op_emb)
            if self.b_use_bn:
                shape_y = b_y.shape
                b_y = self.b_bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1]))\
                            .reshape(shape_y)
            if i_layer != self.num_b_gcn_layers - 1:
                b_y = F.relu(b_y)
            b_y = F.dropout(b_y, self.dropout, training=self.training)
            # b_y: (batch_size, num_cell_groups, num_nodes+1, b_gcn_out_dims[-1])

        return b_y, b_message

    def _update_op_emb(self, y: Tensor, b_y: Tensor, op_embs: Tensor, message: Tensor, b_message: Tensor, opemb_update_mask: Tensor) -> Tensor:
        # --- UpdateOpEmb ---
        if self.updateopemb_method == "concat_ofb":
            unsqueeze_y = y.unsqueeze(-2).repeat([1, 1, self._num_all_nodes, 1])
            unsqueeze_b_y = b_y.unsqueeze(-3).repeat([1, self._num_all_nodes, 1, 1])
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

    def forward(self, archs):
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

        # adjs: (batch_size, num_cell_groups, num_all_nodes, num_nodes)
        # x: (batch_size, num_cell_groups, num_all_nodes, op_hid)
        # op_embs: (batch_size, num_cell_groups, num_all_nodes, num_all_nodes, emb_dim)
        adjs, x, op_embs = self.embed_and_transform_arch(archs)
        batch_size = x.shape[0]

        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = op_embs.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = op_embs.new(zs_as_p)
            zs_as_p = zs_as_p.reshape([
                batch_size * self._num_cg,
                self._num_all_nodes, self._num_all_nodes, zs_as_p.shape[-1]])
        elif self.symmetry_breaking_method == "param_zs_add":
            zs_as_p = op_embs.new(zs_as_p)
            zs_as_p = self.param_zs_embedder(zs_as_p)
            op_embs = op_embs + self.concat_param_zs_as_opemb_scale * zs_as_p

        x = x.reshape([batch_size * self._num_cg,
                       self._num_all_nodes, x.shape[3]])
        adjs = adjs.reshape([batch_size * self._num_cg,
                             self._num_all_nodes, self._num_all_nodes])
        op_embs = op_embs.reshape([
            batch_size * self._num_cg,
            self._num_all_nodes, self._num_all_nodes, op_embs.shape[-1]])

        # calculate op mask
        opemb_update_mask = F.embedding(adjs, self._parametrized_op_emb)

        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim=-1)
            else:
                auged_op_embs = op_embs

            y, message = self._forward_pass(x, adjs, auged_op_embs)

            if t == self.num_time_steps - 1:
                break

            b_y, b_message = self._backward_pass(y, adjs, zs_as_l, auged_op_embs, batch_size)
            op_embs = self._update_op_emb(y, b_y, op_embs, message, b_message, opemb_update_mask)

        y = self._final_process(y, batch_size)
        return y

    def _final_process(self, y: Tensor, batch_size: int) -> Tensor:
        ## --- output ---
        if self.final_concat:
            # do not keep the init node embedding and the final special node
            y = y[:, self._num_init_nodes:-1, :]
        else:
            y = y[:, -1, :] # just use the final special node

        y = torch.reshape(y, [batch_size, -1]) # concat across cell groups, just reshape here
        return y


class NB301FBFlowAnyTimeArchEmbedder(NB301FBFlowArchEmbedder):
    """
    Implement of TA-GATES anytime architecture embedder on NAS-Bench-301.
    """
    NAME = "cellss-fbflow-anytime"

    def forward(self, archs, any_time: bool = False):
        if not any_time:
            return super(NB301FBFlowAnyTimeArchEmbedder, self).forward(archs)

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

        # adjs: (batch_size, num_cell_groups, num_all_nodes, num_nodes)
        # x: (batch_size, num_cell_groups, num_all_nodes, op_hid)
        # op_embs: (batch_size, num_cell_groups, num_all_nodes, num_all_nodes, emb_dim)
        adjs, x, op_embs = self.embed_and_transform_arch(archs)
        batch_size = x.shape[0]

        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = op_embs.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = op_embs.new(zs_as_p)
            zs_as_p = zs_as_p.reshape([
                batch_size * self._num_cg,
                self._num_all_nodes, self._num_all_nodes, zs_as_p.shape[-1]])
        elif self.symmetry_breaking_method == "param_zs_add":
            zs_as_p = op_embs.new(zs_as_p)
            zs_as_p = self.param_zs_embedder(zs_as_p)
            op_embs = op_embs + self.concat_param_zs_as_opemb_scale * zs_as_p
        
        x = x.reshape([batch_size * self._num_cg,
                       self._num_all_nodes, x.shape[3]])
        adjs = adjs.reshape([batch_size * self._num_cg,
                             self._num_all_nodes, self._num_all_nodes])
        op_embs = op_embs.reshape([
            batch_size * self._num_cg,
            self._num_all_nodes, self._num_all_nodes, op_embs.shape[-1]])

        # calculate op mask
        opemb_update_mask = F.embedding(adjs, self._parametrized_op_emb)

        y_cache = []
        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim = -1)
            else:
                auged_op_embs = op_embs

            y, message = self._forward_pass(x, adjs, auged_op_embs)
            y_cache.append(self._final_process(y, batch_size))

            if t == self.num_time_steps - 1:
                break

            b_y, b_message = self._backward_pass(y, adjs, zs_as_l, auged_op_embs, batch_size)
            op_embs = self._update_op_emb(y, b_y, op_embs, message, b_message, opemb_update_mask)

        return y_cache
