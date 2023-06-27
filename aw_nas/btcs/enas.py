"""
ENAS search space (a `cnn` search space), evaluator (API query)
"""

import copy
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas.common import CNNSearchSpace
from aw_nas.rollout.base import Rollout
from aw_nas.evaluator.arch_network import ArchEmbedder
from aw_nas.utils import DenseGraphOpEdgeFlow


class ENASDenseGraphOpEdgeFlow(DenseGraphOpEdgeFlow):
    r"""
    For ENAS search space.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        op_emb_dim: int,
        has_attention: bool = True,
        share_self_op_emb: bool = False,
        normalize: bool = False,
        bias: bool = False,
        residual_only: bool = None,
        use_sum: bool = False,
        concat: bool = None,
        has_aggregate_op: bool = False,
        reverse: bool = False,
        skip_connection_index: int = None,
        return_message: bool = False
    ) -> None:
        super(ENASDenseGraphOpEdgeFlow, self).__init__(
                in_features,
                out_features,
                op_emb_dim,
                has_attention = has_attention,
                plus_I = False,
                share_self_op_emb = share_self_op_emb,
                normalize = normalize,
                bias = bias,
                residual_only = residual_only,
                use_sum = use_sum,
                concat = concat,
                has_aggregate_op = has_aggregate_op,
                reverse = reverse
        )

        self.skip_connection_index = skip_connection_index
        self.return_message = return_message

    def forward(self, inputs, adj, adj_op_inds, op_embs, self_op_emb = None):
        # op_embs: (b, n_d, V, V, h_o)
        # note that b is batch_size x num_cg outside
        # support: (b, V, h_i)
        support = torch.matmul(inputs, self.weight)
        # attn_mask: (b, n_d, V, V, 1)
        attn_mask = (adj_op_inds > 0).unsqueeze(-1)

        attn = torch.sigmoid(self.op_attention(op_embs))
        if self.skip_connection_index is not None:
            is_skip_mask = (adj_op_inds == self.skip_connection_index).unsqueeze(-1).to(torch.float32).detach()
            attn = is_skip_mask * attn.new(np.ones((adj.shape[-1], adj.shape[-1], attn.shape[-1]))) + (1 - is_skip_mask) * attn
        attn = attn_mask * attn
        # attn: (b, n_d, V, V, h_i)

        if self.residual_only is None:
            res_output = support

        elif self.reverse:
            res_output = torch.cat(
                    (
                        torch.zeros([
                            support.shape[0],
                            support.shape[1] - self.residual_only,
                            support.shape[2]
                            ], device = support.device
                        ),
                        support[:, -self.residual_only:, :]
                    ), dim = 1
            )

        else:
            res_output = torch.cat(
                (
                    support[:, :self.residual_only, :],
                    torch.zeros([
                        support.shape[0],
                        support.shape[1] - self.residual_only,
                        support.shape[2]
                        ], device = support.device
                    )
                ), dim = 1
            )

        processed_message = attn * support.unsqueeze(1).unsqueeze(1)
        # processed_message: (b, n_d, V, V, h_i)
        processed_info = processed_message.sum(-2)
        # processed_info: (b, n_d, V, h_i)
        processed_info = processed_info.sum(1) if self.use_sum else processed_info.mean(1)
        # processed_info: (b, V, h_i)

        if self.has_aggregate_op:
            output = self.aggregate_op(processed_info) + res_output
        else:
            output = processed_info + res_output

        if self.bias is not None:
            output = output + self.bias

        if self.return_message:
            return output, processed_message

        return output


class ENASRollout(Rollout):
    r"""
    Rollout for the ENAS search space.
    """

    NAME = "enas"

    supported_components = [
        ("trainer", "simple"),
        ("evaluator", "mepa"),
        ("weights_manager", "supernet")
    ]

    @classmethod
    def random_sample_arch(cls, num_cell_groups, num_steps, num_init_nodes, num_node_inputs, num_primitives) -> list:
        r"""
        Randomly sample an architecture from the search space.
        """
        arch = []
        for i_cg in range(num_cell_groups):
            num_prims = num_primitives if isinstance(num_primitives, int) else num_primitives[i_cg]
            nodes = []
            ops = []
            _num_step = num_steps if isinstance(num_steps, int) else num_steps[i_cg]
            for i_out in range(_num_step):
                nodes += list(np.random.randint(0, high = i_out + num_init_nodes, size = num_node_inputs))
                ops += list(np.random.randint(0, high = num_prims, size = num_node_inputs))
            arch.append((nodes, ops))
        return arch


class ENASSearchSpace(CNNSearchSpace):
    r"""
    ENAS search space.
    """
    NAME = "enas"

    def __init__(
        self,
        shared_primitives: Tuple[str] = (
            "sep_conv_3x3",
            "sep_conv_5x5",
            "max_pool_3x3",
            "avg_pool_3x3",
            "skip_connect")
        ) -> None:
        super(ENASSearchSpace, self).__init__(
            num_cell_groups = 2,
            num_init_nodes = 2,
            num_layers = 20,
            cell_layout = None,
            reduce_cell_groups = (1,),
            num_steps = 5,
            num_node_inputs = 2,
            concat_op = "concat",
            concat_nodes = None,
            loose_end = True,
            shared_primitives = shared_primitives,
            derive_without_none_op = False
        )

    def random_sample(self) -> ENASRollout:
        r"""
        Random sample a discrete architecture from the search space.
        """
        arch = ENASRollout.random_sample_arch(
            self.num_cell_groups,
            self.num_steps,
            self.num_init_nodes,
            self.num_node_inputs,
            self._num_primitives
        )
        return ENASRollout(arch, info = {}, search_space = self)

    def canonicalize(self, rollout: ENASRollout) -> str:
        # TODO: Is this serious?
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
                    if (ops[index] == 4):
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
    def supported_rollout_types(cls) -> List[str]:
        return ["enas"]


class ENASFBFlowArchEmbedder(ArchEmbedder):
    r"""
    Use DenseGraphOpEdgeFlow (Support multiple edges between a pair of nodes)
    """
    NAME = "enas-fbflow"

    def __init__(
        self,
        search_space,
        node_dim: int = 32,
        op_dim: int = 32,
        hidden_dim: int = 32,
        gcn_out_dims: List[int] = [64, 64, 64, 64, 64, 64],
        other_node_zero: bool = True,
        gcn_kwargs: dict = None,
        dropout: float = 0.,
        normalize: bool = False,
        use_bn: bool = False,
        other_node_independent: bool = False,
        share_self_op_emb: bool = False,

        ## newly added
        take_adj_as_input: bool = True,
        # if take adj as input, the `share_skip_and_outskip` configuration
        # should be set correctly manually corresponding to the adj generation script

        # construction configurations
        # construction (tagates)
        num_time_steps: int = 2,
        fb_conversion_dims: List[int] = [64, 64],
        backward_gcn_out_dims: List[int] = [64, 64, 64, 64, 64, 64],
        updateopemb_method: str = "concat_ofb_message", # concat_ofb_message, concat_ofb
        updateopemb_scale: float = 0.1,
        updateopemb_dims: List[int] = [64],
        b_use_bn: bool = False,
        share_skip_and_outskip: bool = True,
        # construction (l): concat arch-level zeroshot as l
        concat_arch_zs_as_l_dimension = None,
        concat_l_layer = 0,
        # construction (symmetry breaking)
        symmetry_breaking_method: str = None, # None, "random", "param_zs"
        concat_param_zs_as_opemb_dimension = None,

        # gradient flow configurations
        detach_vinfo: bool = False,
        updateopemb_detach_opemb: bool = True,
        updateopemb_detach_finfo: bool = True,
        schedule_cfg = None
    ) -> None:
        super(ENASFBFlowArchEmbedder, self).__init__(schedule_cfg)

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
        assert self.symmetry_breaking_method in {None, "param_zs", "random"}


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
                                               requires_grad=not other_node_zero)

        self.num_ops = len(self.search_space.shared_primitives)
        try:
            self.none_index = self.search_space.shared_primitives.index("none")
            self.add_none_index = False
            assert self.none_index == 0, \
                "search space with none op should have none op as the first primitive"
        except ValueError:
            # self.none_index = len(self.search_space.shared_primitives)
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
        self._parametrized_op_emb = [
            [float("conv" in op_name)] for op_name in self.search_space.shared_primitives]
        if self.add_none_index: # add none
            self._parametrized_op_emb.insert(0, [0.])
        if self.num_ops > len(self._parametrized_op_emb): # add special
            self._parametrized_op_emb.append([0.])
        self._parametrized_op_emb = nn.Parameter(
            torch.tensor(self._parametrized_op_emb, dtype=torch.float32), requires_grad=False)

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hidden_dim
        f_gcn_kwargs = copy.deepcopy(gcn_kwargs) if gcn_kwargs is not None else {}
        f_gcn_kwargs["skip_connection_index"] = self.outskip_index
        f_gcn_kwargs.update(addi_kwargs)
        for dim in self.gcn_out_dims:
            self.gcns.append(ENASDenseGraphOpEdgeFlow(
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
            self.b_gcns.append(ENASDenseGraphOpEdgeFlow(
                in_dim, dim,
                self.op_dim + self.concat_param_zs_as_opemb_dimension \
                if symmetry_breaking_method == "param_zs" else self.op_dim,
                reverse=True,
                **(b_gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                # FIXME: i think the bn is not correct... but nothing, we do not use now
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
            # elif self.updateopemb_method == "concat_ofb_message":
            # # elif self.updateopemb_method == "concat_fb":
            # #     in_dim = self.gcn_out_dims[-1] + self.backward_gcn_out_dims[-1]
            # # elif self.updateopemb_method == "concat_b":
            # #     in_dim = self.backward_gcn_out_dims[-1]
            else:
                raise NotImplementedError()

            self.updateop_embedder = []
            for embedder_dim in self.updateopemb_dims:
                self.updateop_embedder.append(nn.Linear(in_dim, embedder_dim))
                self.updateop_embedder.append(nn.ReLU(inplace=False))
                in_dim = embedder_dim
            self.updateop_embedder.append(nn.Linear(in_dim, self.op_dim))
            self.updateop_embedder = nn.Sequential(*self.updateop_embedder)

        # output dimension
        self.out_dim = self._num_cg * self.gcn_out_dims[-1]

    def embed_and_transform_arch(self, archs):
        archs, adj_op_inds = archs

        b_size, n_cg, _, _ = archs.shape
        adjs = torch.tensor(archs, dtype=torch.long)
        adjs = adjs.reshape([b_size, n_cg, adjs.shape[-1], adjs.shape[-1]]).to(
            self.init_node_emb.device)
        adj_op_inds = torch.tensor(adj_op_inds).reshape(
            [b_size, n_cg, self._num_node_inputs,
             adjs.shape[-1], adjs.shape[-1]]).to(self.init_node_emb.device)
        # (batch_size, num_cell_groups, num_nodes, num_nodes)

        # operation embedding
        op_embs = F.embedding(adj_op_inds, torch.stack(self.op_emb))

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
        return adjs, x, adj_op_inds, op_embs

    def forward(self, archs):
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

        # adjs: (batch_size, num_cell_groups, num_all_nodes, num_nodes)
        # x: (batch_size, num_cell_groups, num_all_nodes, op_hid)
        # op_embs: (batch_size, num_cell_groups, num_node_inputs,
        #           num_all_nodes, num_all_nodes, emb_dim)
        adjs, x, adj_op_inds, op_embs = self.embed_and_transform_arch(archs)
        batch_size = x.shape[0]

        if self.concat_arch_zs_as_l_dimension is not None:
            zs_as_l = x.new(np.array(zs_as_l))
            assert zs_as_l.shape[-1] == self.concat_arch_zs_as_l_dimension

        # symmetry breaking
        if self.symmetry_breaking_method == "random":
            # random, dimension not changed: op_emb + random noise
            noise = torch.zeros_like(op_embs).normal_() * 0.1
            op_embs = op_embs + noise
        elif self.symmetry_breaking_method == "param_zs":
            # param-level zeroshot: op_emb | zeroshot
            zs_as_p = x.new(zs_as_p)
            zs_as_p = zs_as_p.reshape([
                batch_size * self._num_cg, self._num_node_inputs,
                self._num_all_nodes, self._num_all_nodes, zs_as_p.shape[-1]])

        x = x.reshape([batch_size * self._num_cg,
                       self._num_all_nodes, x.shape[3]])
        adjs = adjs.reshape([batch_size * self._num_cg,
                             self._num_all_nodes, self._num_all_nodes])
        op_embs = op_embs.reshape([
            batch_size * self._num_cg, self._num_node_inputs,
            self._num_all_nodes, self._num_all_nodes, op_embs.shape[-1]])
        adj_op_inds = adj_op_inds.reshape([
            batch_size * self._num_cg, self._num_node_inputs,
            self._num_all_nodes, self._num_all_nodes])

        # calculate op mask
        opemb_update_mask = F.embedding(adj_op_inds, self._parametrized_op_emb)

        for t in range(self.num_time_steps):
            # concat zeroshot onto the op embedding for forward and backward
            if self.symmetry_breaking_method == "param_zs":
                # param-level zeroshot: op_emb | zeroshot
                auged_op_embs = torch.cat((op_embs, zs_as_p), dim=-1)
            else:
                auged_op_embs = op_embs

            y = x
            for i_layer, gcn in enumerate(self.gcns):
                # outskip is handled by specifying `skip_connection_index`
                if self.use_message:
                    y, message = gcn(y, adjs, adj_op_inds, auged_op_embs,
                                     self_op_emb=self.self_op_emb)
                else:
                    y = gcn(y, adjs, adj_op_inds, auged_op_embs, self_op_emb=self.self_op_emb)
                if self.use_bn:
                    shape_y = y.shape
                    y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
                if i_layer != self.num_gcn_layers - 1:
                    y = F.relu(y)
                y = F.dropout(y, self.dropout, training=self.training)
                # y: (batch_size * num_cell_groups, num_nodes+1, gcn_out_dims[-1])

            if t == self.num_time_steps - 1:
                break

            # --- backward pass ---
            # use the special final aggregation node
            v_info_n = y[:, self._num_nodes, :].reshape(
                batch_size, self._num_cg * y.shape[-1])
            if self.detach_vinfo:
                v_info_n = v_info_n.detach()
            if self.concat_arch_zs_as_l_dimension:
                # process before concat l
                v_info_n = self.fb_conversion[:self._num_before_concat_l](v_info_n)
                # concat l
                v_info_n = torch.cat((v_info_n, zs_as_l.unsqueeze(-2)), dim=-1)
                assert not self.concat_l_layer == len(self.fb_converseion_list) - 1
                # process after concat l
                v_info_n = self.fb_conversion[self._num_before_concat_l:](v_info_n)
            else:
                v_info_n = self.fb_conversion(v_info_n)
            v_info_n = v_info_n.reshape(batch_size * self._num_cg, 1, -1)
            v_info_n = torch.cat(
                (
                    torch.zeros([batch_size * self._num_cg, self._num_nodes, v_info_n.shape[-1]],
                                device=y.device),
                    v_info_n
                ), dim=1
            )

            # start backward flow
            b_adjs = adjs.transpose(-1, -2)
            b_y = v_info_n
            b_op_embs = auged_op_embs.transpose(-2, -3)
            b_adj_op_inds = adj_op_inds.transpose(-1, -2)
            for i_layer, gcn in enumerate(self.b_gcns):
                # outskip is handled by specifying `skip_connection_index`
                if self.use_message:
                    b_y, b_message = gcn(b_y, b_adjs, b_adj_op_inds, b_op_embs,
                                         self_op_emb=self.self_op_emb)
                else:
                    b_y = gcn(b_y, b_adjs, b_adj_op_inds, b_op_embs, self_op_emb=self.self_op_emb)
                if self.b_use_bn:
                    shape_y = b_y.shape
                    b_y = self.b_bns[i_layer](b_y.reshape(shape_y[0], -1, shape_y[-1]))\
                              .reshape(shape_y)
                if i_layer != self.num_b_gcn_layers - 1:
                    b_y = F.relu(b_y)
                b_y = F.dropout(b_y, self.dropout, training=self.training)
                # b_y: (batch_size, num_cell_groups, num_nodes+1, b_gcn_out_dims[-1])

            # --- UpdateOpEmb ---
            if self.updateopemb_method == "concat_ofb":
                # TODO: use gather and scatter to do this to be more memory and computation efficient
                unsqueeze_y = y.unsqueeze(-2).unsqueeze(1).repeat([1, 2, 1, self._num_all_nodes, 1])
                unsqueeze_b_y = b_y.unsqueeze(-3).unsqueeze(1).repeat(
                    [1, 2, self._num_all_nodes, 1, 1])
                in_embedding = torch.cat(
                    (
                        op_embs.detach() if self.updateopemb_detach_opemb else op_embs,
                        unsqueeze_y.detach() if self.updateopemb_detach_finfo else unsqueeze_y,
                        unsqueeze_b_y
                    ), dim=-1
                )
            elif self.updateopemb_method == "concat_ofb_message": # use_message==True
                in_embedding = torch.cat(
                    (
                        op_embs.detach() if self.updateopemb_detach_opemb else op_embs,
                        message.detach() if self.updateopemb_detach_finfo else message,
                        b_message.transpose(-2, -3)
                    ), dim=-1
                )
            else:
                raise Exception()
            update = self.updateop_embedder(in_embedding)
            update = update * opemb_update_mask
            op_embs = op_embs + self.updateopemb_scale * update

        ## --- output ---
        y = y[:, -1, :] # just use the final special node
        y = torch.reshape(y, [batch_size, -1]) # concat across cell groups, just reshape here
        return y


if __name__ == "__main__":
    ss = ENASSearchSpace()
    rollout = ss.random_sample()
    import ipdb
    ipdb.set_trace()
