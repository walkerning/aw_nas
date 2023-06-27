"""
Networks that take architectures as inputs.
"""

import abc
import logging
from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from aw_nas import utils
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.base import Component
from aw_nas.utils import DenseGraphConvolution, DenseGraphOpEdgeFlow

__all__ = ["PointwiseComparator"]

class ArchNetwork(Component):
    REGISTRY = "arch_network"

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class ArchEmbedder(Component, nn.Module):
    REGISTRY = "arch_embedder"

    def __init__(self, schedule_cfg):
        Component.__init__(self, schedule_cfg)
        nn.Module.__init__(self)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))


class LSTMArchEmbedder(ArchEmbedder):
    NAME = "lstm"

    def __init__(self, search_space,
                 op_embedding_size=48,
                 node_embedding_size=48,
                 hidden_size=96,
                 dropout_ratio=0.,
                 num_layers=1,
                 schedule_cfg=None):
        super(LSTMArchEmbedder, self).__init__(schedule_cfg)
        self.search_space = search_space

        self.op_embedding_size = op_embedding_size
        self.node_embedding_size = node_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        _n_node_input = self.search_space.num_init_nodes + self.search_space.num_steps - 1
        # unlike controller rl_network,
        # this module only support shared primitives for all cell groups
        self.op_emb = nn.Embedding(len(self.search_space.shared_primitives), self.op_embedding_size)
        self.node_emb = nn.Embedding(_n_node_input, self.node_embedding_size)

        self.rnn = nn.LSTM(input_size=self.op_embedding_size + self.node_embedding_size,
                           hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True, dropout=dropout_ratio)

        # calculate out dim
        self.out_dim = self.hidden_size

    def embed_and_transform_arch(self, archs):
        if isinstance(archs, (np.ndarray, list, tuple)):
            archs = np.array(archs)
            if archs.ndim == 3:
                archs = np.expand_dims(archs, 0)
            else:
                assert archs.ndim == 4
            archs = self.node_emb.weight.new(archs).long()

        # embedding nodes
        # (batch_size, num_cell_groups, num_node_inputs * num_steps, node_embedding_size)
        node_embs = self.node_emb(archs[:, :, 0, :])
        # embedding ops
        op_embs = self.op_emb(archs[:, :, 1, :])
        # re-arrange
        cat_emb = torch.cat([node_embs, op_embs], dim=-1)

        return torch.reshape(cat_emb, [cat_emb.shape[0], -1, cat_emb.shape[-1]])

    def forward(self, archs):
        emb = self.embed_and_transform_arch(archs)
        # TODO: dropout on embedding?
        out, _ = self.rnn(emb)
        # normalize the output following NAO
        out = F.normalize(out, 2, dim=-1)

        # average across decisions (time steps)
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        return out

# ---- GCNArchEmbedder ----
# try:
#     from pygcn.layers import GraphConvolution
# except ImportError as e:
#     from aw_nas.utils import logger as _logger
#     _logger.getChild("arch_network").warn(
#         ("Cannot import module pygcn: {}\n"
#          "Should install the pygcn package for graph convolution").format(e))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GCNArchEmbedder(ArchEmbedder):
    NAME = "gcn"

    def __init__(self, search_space,
                 op_dim=48, op_hid=48, gcn_out_dims=[128, 128],
                 gcn_kwargs=None,
                 dropout=0.,
                 schedule_cfg=None):
        super(GCNArchEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space

        # configs
        self.op_dim = op_dim
        self.op_hid = op_hid
        self.gcn_out_dims = gcn_out_dims
        self.dropout = dropout
        self._num_init_nodes = self.search_space.num_init_nodes
        self._num_node_inputs = self.search_space.num_node_inputs
        self._num_steps = self.search_space.num_steps
        self._num_nodes = self._num_steps + self._num_init_nodes

        # the embedding of the first two nodes
        # self.init_node_emb = nn.ModuleList(
        #     [nn.Embedding(
        #         self.search_space.num_init_nodes, self._num_node_inputs * self.op_dim)
        #      for _ in self.search_space.num_cell_groups]
        # )
        # share init node embedding for all cell groups
        self.init_node_emb = nn.Embedding(
            self._num_init_nodes, self._num_node_inputs * self.op_dim)

        self.op_emb = nn.Embedding(len(search_space.shared_primitives), self.op_dim)
        # concat the embedding [op0, op1, ...] for each node
        self.x_hidden = nn.Linear(self._num_node_inputs * self.op_dim, self.op_hid)

        # init graph convolutions
        self.gcns = []
        in_dim = self.op_hid
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphConvolution(in_dim, dim, **(gcn_kwargs or {})))
            in_dim = dim
        self.gcns = nn.ModuleList(self.gcns)
        self.num_gcn_layers = len(self.gcns)

        self.out_dim = self.search_space.num_cell_groups * in_dim

    def get_adj_sparse(self, arch):
        return self._get_adj_sparse(arch, self._num_init_nodes,
                                    self._num_node_inputs, self._num_nodes)

    def get_adj_dense(self, arch):
        return self._get_adj_dense(arch, self._num_init_nodes,
                                   self._num_node_inputs, self._num_nodes)

    def _get_adj_sparse(self, arch, num_init_nodes, num_node_inputs, num_nodes): #pylint: disable=no-self-use
        """
        :param arch: previous_nodes, e.g. [1, 0, 0, 1, 2, 0, 4, 4],
            0, 1 is the previous init nodes
        :param num_node:
        :return:
        """
        f_nodes = np.array(arch)
        t_nodes = np.repeat(np.array(range(num_init_nodes, num_nodes)), num_node_inputs)
        adj = sp.coo_matrix((np.ones(f_nodes.shape[0]), (t_nodes, f_nodes)),
                            shape=(num_nodes, num_nodes), dtype=np.float32)
        adj = adj.multiply(adj > 0)
        # build symmetric adjacency matrix for undirected graph
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def _get_adj_dense(self, arch, num_init_nodes, num_node_inputs, num_nodes): #pylint: disable=no-self-use
        """
        get dense adjecent matrix, could be batched
        :param arch: previous_nodes, e.g. [1, 0, 0, 1, 2, 0, 4, 4],
            0, 1 is the previous init nodes
        :param num_node:
        :return:
        """
        f_nodes = np.array(arch)
        _ndim = f_nodes.ndim
        if _ndim == 1:
            f_nodes = np.expand_dims(arch, 0)
        else:
            assert _ndim == 2
        batch_size = f_nodes.shape[0]
        t_nodes = np.tile(
            np.repeat(np.array(range(num_init_nodes, num_nodes)), num_node_inputs)[None, :],
            [batch_size, 1]
        )
        batch_inds = np.tile(np.arange(batch_size)[:, None], [1, t_nodes.shape[1]])
        indexes = np.stack((batch_inds, t_nodes, f_nodes))
        indexes = indexes.reshape([3, -1])
        indexes, edge_counts = np.unique(indexes, return_counts=True, axis=1)
        adj = torch.zeros(batch_size, num_nodes, num_nodes)
        adj[indexes] += torch.tensor(edge_counts, dtype=torch.float32)
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
                assert archs.ndim == 4

        # get adjacent matrix
        # sparse
        # archs[:, :, 0, :]: (batch_size, num_cell_groups, num_node_inputs * num_steps)
        b_size, n_cg, _, n_edge = archs.shape
        adjs = self.get_adj_dense(archs[:, :, 0, :].reshape([-1, n_edge]))
        adjs = adjs.reshape([b_size, n_cg, adjs.shape[1], adjs.shape[2]]).to(
            self.op_emb.weight.device)
        # (batch_size, num_cell_groups, num_nodes, num_nodes)

        # embedding ops
        op_inds = torch.tensor(archs[:, :, 1, :]).to(self.op_emb.weight.device)
        op_embs = self.op_emb(op_inds)
        # (batch_size, num_cell_groups, num_node_inputs * num_steps, op_dim)

        shape = op_embs.shape
        # concat two input op embedding for each node, use reshape to replace split+cat
        # inter_node_embs = [t.unsqueeze(3) for t in torch.split(
        #     op_embs.reshape([
        #         shape[0], shape[1], self._num_steps,
        #         self._num_node_inputs, shape[3]]),
        #     1, dim=3)]
        # inter_node_embs = torch.cat(inter_node_embs, dim=-1)
        inter_node_embs = op_embs.reshape([
            shape[0], shape[1], self._num_steps, self._num_node_inputs * shape[3]])
        # (batch_size, num_cell_groups, num_steps, num_node_inputs * self.op_dim)

        # embedding of all nodes
        unsqueezed_init_emb = self.init_node_emb\
                                  .weight\
                                  .unsqueeze(0)\
                                  .unsqueeze(0)\
                                  .repeat([shape[0], shape[1], 1, 1])
        node_embs = torch.cat((unsqueezed_init_emb, inter_node_embs), dim=2)
        # (batch_size, num_cell_groups, num_nodes, num_node_inputs * self.op_dim)

        x = self.x_hidden(node_embs)
        # (batch_size, num_cell_groups, num_nodes, op_hid)
        return adjs, x

    def forward(self, archs):
        # adjs: (batch_size, num_cell_groups, num_nodes, num_nodes)
        # x: (batch_size, num_cell_groups, num_nodes, op_hid)
        adjs, x = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, num_cell_groups, num_nodes, gcn_out_dims[-1])
        y = y[:, :, 2:, :] # do not keep the init node embedding
        y = torch.mean(y, dim=2) # average across nodes (bs, nc, god)
        y = torch.reshape(y, [y.shape[0], -1]) # concat across cell groups, just reshape here
        return y

# ---- END: GCNArchEmbedder ----

# ---- BEGIN: GCNFlowArchEmbedder ----
class GCNFlowArchEmbedder(ArchEmbedder):
    NAME = "cellss-flow"

    def __init__(self, search_space,
                 node_dim=48, op_dim=48, hidden_dim=48,
                 gcn_out_dims=[128, 128],
                 other_node_zero=False,
                 gcn_kwargs=None,
                 dropout=0.,
                 normalize=False,
                 use_bn=False,
                 other_node_independent=False,
                 share_self_op_emb=False,
                 final_concat=False,
                 schedule_cfg=None):
        super(GCNFlowArchEmbedder, self).__init__(schedule_cfg)

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

        self._num_init_nodes = self.search_space.num_init_nodes
        self._num_node_inputs = self.search_space.num_node_inputs
        self._num_steps = self.search_space.num_steps
        self._num_nodes = self._num_steps + self._num_init_nodes
        self._num_cg = self.search_space.num_cell_groups

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
            self.none_index = len(self.search_space.shared_primitives)
            self.none_index = 0
            self.add_none_index = True
            self.num_ops += 1

        self.op_emb = []
        for idx in range(self.num_ops):
            if idx == self.none_index:
                emb = nn.Parameter(torch.zeros(self.op_dim), requires_grad=False)
            else:
                emb = nn.Parameter(torch.Tensor(self.op_dim).normal_())
            setattr(self, "op_embedding_{}".format(idx), emb)
            self.op_emb.append(emb)
        if self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(self.op_dim).normal_())
        else:
            self.self_op_emb = None

        self.x_hidden = nn.Linear(self.node_dim, self.hidden_dim)

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hidden_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphOpEdgeFlow(
                in_dim, dim, self.op_dim, **(gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self._num_nodes * self._num_cg))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        if not self.final_concat:
            self.out_dim = self._num_cg * in_dim
        else:
            self.out_dim = self._num_cg * in_dim * self._num_steps

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
        indexes = ori_indexes.reshape([3, -1])
        indexes, edge_counts = np.unique(indexes, return_counts=True, axis=1)
        adj = torch.zeros(batch_size, num_nodes, num_nodes)
        adj[indexes] += torch.tensor(edge_counts, dtype=torch.float32)
        adj_op_inds_lst = [
            torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.long) * none_index
            for _ in range(num_node_inputs)]
        ori_indexes_lst = np.split(
            ori_indexes.reshape(
                3, ori_indexes.shape[1],
                ori_indexes.shape[-1]//num_node_inputs, num_node_inputs),
            range(1, num_node_inputs), axis=-1)
        ops_lst = np.split(ops.reshape(ops.shape[0], ops.shape[1] // num_node_inputs, num_node_inputs), range(1, num_node_inputs), axis=-1)
        for adj_op_inds, inds, op in zip(adj_op_inds_lst, ori_indexes_lst, ops_lst):
            adj_op_inds[inds] = torch.tensor(op)

        if _ndim == 1:
            adj = adj[0]
            adj_op_inds_lst = [adj_op_inds[0] for adj_op_inds in adj_op_inds_lst]
            # adj_op_inds = adj_op_inds[0]
        return adj, adj_op_inds_lst

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
        b_size, n_cg, _, n_edge = archs.shape
        adjs, adj_op_inds_lst = self.get_adj_dense(archs.reshape(b_size * n_cg, 2, n_edge))
        adjs = adjs.reshape([b_size, n_cg, adjs.shape[1], adjs.shape[2]]).to(
            self.init_node_emb.device)
        adj_op_inds_lst = [adj_op_inds.reshape([b_size, n_cg, adj_op_inds.shape[1],
                                                adj_op_inds.shape[2]]).to(
                                                    self.init_node_emb.device)
                           for adj_op_inds in adj_op_inds_lst]
        # (batch_size, num_cell_groups, num_nodes, num_nodes)

        # embedding of init nodes
        # TODO: output op should have a embedding maybe? (especially for hierarchical purpose)
        if self.other_node_independent:
            node_embs = self.init_node_emb.unsqueeze(0).repeat(b_size, 1, 1, 1)
        else:
            node_embs = torch.cat(
                (self.init_node_emb.unsqueeze(0).repeat(b_size, 1, 1, 1),
                 self.other_node_emb.unsqueeze(0).repeat(b_size, 1, self._num_steps, 1)),
                dim=2)
        # (batch_size, num_cell_groups, num_nodes, self.node_dim)

        x = self.x_hidden(node_embs)
        # (batch_size, num_cell_groups, num_nodes, op_hid)
        return adjs, adj_op_inds_lst, x

    def forward(self, archs, return_all=False):
        # adjs: (batch_size, num_cell_groups, num_nodes, num_nodes)
        # adj_op_inds: (batch_size, num_cell_groups, num_nodes, num_nodes)
        # x: (batch_size, num_cell_groups, num_nodes, op_hid)
        adjs, adj_op_inds_lst, x = self.embed_and_transform_arch(archs)
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, adj_op_inds_lst, torch.stack(self.op_emb), self.none_index,
                    self_op_emb=self.self_op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, num_cell_groups, num_nodes, gcn_out_dims[-1])
        if return_all:
            y = F.normalize(y, 2, dim=-1) if self.normalize else y
            return y
        y = y[:, :, 2:, :] # do not keep the init node embedding
        if self.normalize:
            y = F.normalize(y, 2, dim=-1)
        if not self.final_concat:
            y = torch.mean(y, dim=2) # average across nodes (bs, nc, god)
        else:
            # concat across all internal nodes (bs, nc, num_steps * god)
            y = torch.reshape(y, [y.shape[0], y.shape[1], -1])
        if self.normalize:
            y = F.normalize(y, 2, dim=-1)
        y = torch.reshape(y, [y.shape[0], -1]) # concat across cell groups, just reshape here
        return y
# ---- END: GCNFlowArchEmbedder ----


class PointwiseComparator(ArchNetwork, nn.Module):
    """
    Compatible to NN regression-based predictor of architecture performance.
    """
    NAME = "pointwise_comparator"

    def __init__(self, search_space,
                 arch_embedder_type="lstm", arch_embedder_cfg=None,
                 mlp_hiddens=(200, 200, 200), mlp_dropout=0.1,
                 optimizer={
                     "type": "Adam",
                     "lr": 0.001
                 }, scheduler=None,
                 compare_loss_type="margin_linear",
                 compare_margin=0.01,
                 margin_l2=False,
                 use_incorrect_list_only=False,
                 tanh_score=None,
                 max_grad_norm=None,
                 schedule_cfg=None):
        # [optional] arch reconstruction loss (arch_decoder_type/cfg)
        super(PointwiseComparator, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

        # configs
        expect(compare_loss_type in {"binary_cross_entropy", "margin_linear"},
               "comparing loss type {} not supported".format(compare_loss_type),
               ConfigException)
        self.compare_loss_type = compare_loss_type
        self.compare_margin = compare_margin
        self.margin_l2 = margin_l2
        self.max_grad_norm = max_grad_norm
        # for update_argsort listwise only
        self.use_incorrect_list_only = use_incorrect_list_only
        self.tanh_score = tanh_score

        self.search_space = search_space
        ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
        self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim
        # construct MLP from embedding to score
        self.mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout)
        
        # init optimizer and scheduler
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

        # used for reinit optimizer and lr scheduler
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

    def reinit_optimizer(self, only_mlp: bool = False):
        parameters = self.mlp.parameters() if only_mlp else self.parameters()
        self.optimizer = utils.init_optimizer(parameters, self.optimizer_cfg)
    
    def reinit_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    @staticmethod
    def construct_mlp(dim: int, mlp_hiddens: Tuple[int], mlp_dropout: float, out_dim: int = 1) -> nn.Module:
        mlp = []
        for hidden_size in mlp_hiddens:
            mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace = False),
                nn.Dropout(p = mlp_dropout))
            )
            dim = hidden_size
        mlp.append(nn.Linear(dim, out_dim))
        mlp = nn.Sequential(*mlp)
        return mlp

    def predict_rollouts(self, rollouts, **kwargs):
        archs = [r.arch for r in rollouts]
        return self.predict(archs, **kwargs)

    def predict(self, arch, sigmoid=True, tanh=False):
        score = self.mlp(self.arch_embedder(arch)).squeeze(-1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score
    
    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()

    def update_predict_rollouts(self, rollouts, labels):
        archs = [r.arch for r in rollouts]
        return self.update_predict(archs, labels)

    def update_predict_list(self, predict_lst):
        # use MSE regression loss to step
        archs = [item[0] for item in predict_lst]
        labels = [item[1] for item in predict_lst]
        return self.update_predict(archs, labels)

    def update_predict(self, archs, labels):
        mse_loss = self.cal_predict_loss(archs, labels)
        return self.update_step(mse_loss)

    def cal_predict_loss(self, archs, labels):
        return self._cal_predict_loss(archs, labels, self.mlp)
    
    def _cal_predict_loss(self, archs, labels, mlp):
        scores = torch.sigmoid(mlp(self.arch_embedder(archs)))
        mse_loss = F.mse_loss(
            scores.squeeze(), scores.new(labels))
        return mse_loss

    def compare(self, arch_1, arch_2):
        # pointwise score and comparen
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        return torch.sigmoid(s_2 - s_1)

    def update_compare_rollouts(self, compare_rollouts, better_labels):
        arch_1, arch_2 = zip(*[(r.rollout_1.arch, r.rollout_2.arch) for r in compare_rollouts])
        return self.update_compare(arch_1, arch_2, better_labels)

    def update_compare_list(self, compare_lst):
        # use binary classification loss to step
        arch_1, arch_2, better_labels = zip(*compare_lst)
        return self.update_compare(arch_1, arch_2, better_labels)

    def update_compare_eq(self, arch_1, arch_2, better_eq_labels, margin=None):
        assert self.compare_loss_type == "margin_linear"
        # in range (0, 1) to make the `compare_margin` meaningful
        # s_1 = self.predict(arch_1)
        # s_2 = self.predict(arch_2)
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        better_pm = s_1.new(np.array(better_eq_labels, dtype=np.float32))
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        pair_loss = torch.mean(
            torch.where(
                better_pm == 0,
                torch.max(zero_, (s_2 - s_1).abs() - margin / 2),
                torch.max(zero_, margin - better_pm * (s_2 - s_1))
            ))
        self.optimizer.zero_grad()
        pair_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        # return pair_loss.item(), s_1, s_2
        return pair_loss.item()

    def update_compare(self, arch_1, arch_2, better_labels, margin=None):
        pair_loss = self.cal_compare_loss(arch_1, arch_2, better_labels, margin)
        return self.update_step(pair_loss)

    def cal_compare_loss(self, arch_1, arch_2, better_labels, margin = None):
        return self._cal_compare_loss(arch_1, arch_2, better_labels, self.mlp, margin)

    def _cal_compare_loss(self, arch_1, arch_2, better_labels, mlp, margin = None):
        if self.compare_loss_type == "binary_cross_entropy":
            # compare_score = self.compare(arch_1, arch_2)
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            compare_score = torch.sigmoid(s_2 - s_1)
            pair_loss = F.binary_cross_entropy(
                    compare_score, compare_score.new(better_labels))
        elif self.compare_loss_type == "margin_linear":
            # in range (0, 1) to make the `compare_margin` meaningful
            # s_1 = self.predict(arch_1)
            # s_2 = self.predict(arch_2)
            s_1 = mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = mlp(self.arch_embedder(arch_2)).squeeze()
            better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
            zero_ = s_1.new([0.])
            margin = [self.compare_margin] if margin is None else margin
            margin = s_1.new(margin)
            if not self.margin_l2:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
            else:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
                        ** 2 / np.maximum(1., margin))
        return pair_loss

    def argsort(self, archs, batch_size=None):
        pass

    def update_argsort(self, archs, idxes, first_n=None, accumulate_only=False, is_sorted=False):
        archs = np.array(archs)
        bs, len_ = archs.shape[:2]
        if idxes is not None:
            idxes = np.array(idxes)
            assert idxes.ndim == 2 and idxes.shape[0] == bs and idxes.shape[1] == len_
            # if idxes.ndim == 1:
            #     idxes = idxes[None, :]
        else:
            assert is_sorted
        flat_archs = archs.reshape([-1] + list(archs.shape[2:]))
        if self.tanh_score is not None:
            scores = self.tanh_score * self.predict(flat_archs, sigmoid=False, tanh=True)
        else:
            scores = self.predict(flat_archs, sigmoid=False)

        scores = scores.reshape((bs, len_))
        exp_score = (scores - scores.max(dim=-1, keepdim=True)[0].detach()).exp()
        if not is_sorted:
            exp_score_rank = exp_score[np.arange(0, bs)[:, None], idxes]
        else:
            exp_score_rank = exp_score
        EPS = 1e-12
        exp_score_rank = torch.max(exp_score_rank, torch.tensor(EPS).to(exp_score_rank.device))

        if self.use_incorrect_list_only:
            correct_idxes = torch.all(torch.argsort(
                exp_score_rank, dim=-1, descending=True) \
                                      == exp_score_rank.new(np.arange(len_)).to(torch.long),
                                      dim=-1)
            do_not_train_idxes = (exp_score_rank[:, -1] / exp_score_rank[:, 0] < 1e-9) & correct_idxes
            keep_list_idxes = 1 - do_not_train_idxes
            exp_score_rank = exp_score_rank[keep_list_idxes]
            actual_bs = torch.sum(keep_list_idxes).item()
            logging.debug("actual bs: {}".format(actual_bs))
        else:
            actual_bs = bs

        inds = (np.tile(np.arange(actual_bs)[:, None], [1, len_]),
                np.tile(np.arange(len_)[::-1][None, :], [actual_bs, 1]))
        normalize = torch.cumsum(exp_score_rank[inds], dim=1)[inds]

        if first_n is not None:
            exp_score_rank = exp_score_rank[:, :first_n]
            normalize = normalize[:, :first_n]

        normalize = torch.clamp(normalize, min=1.e-10)
        exp_score_rank = torch.clamp(exp_score_rank, min=1.e-11)
        loss = torch.mean(torch.mean(torch.log(normalize + EPS) - torch.log(exp_score_rank + EPS),
                                     dim=1))

        logging.debug("exp score maxmin: {} {}".format(exp_score_rank.min(), exp_score_rank.max()))
        logging.debug("normalize maxmin: {} {}".format(normalize.min(), normalize.max()))
        logging.debug("loss: {}".format(loss))
        if not accumulate_only:
            self.optimizer.zero_grad()
        loss.backward()
        if not accumulate_only:
            self._clip_grads()
            self.optimizer.step()
        return loss.item()

    # def argsort_list(self, archs, batch_size=None):
    #     # TODO
    #     pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    def on_epoch_start(self, epoch):
        super(PointwiseComparator, self).on_epoch_start(epoch)
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            self.logger.info("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


class DynamicEnsemblePointwiseComparator(PointwiseComparator):
    r"""
    Dynamic ensemble pointwise comparator.

    Args:
        search_space: The search space.
        auxiliary_head_num (int): Number of the low-fidelity experts.
        use_uniform_confidence (bool): Whether use uniform confidence. Default: `False`.
    """
    NAME = "dynamic_ensemble_pointwise_comparator"

    def __init__(self, search_space, 
            auxiliary_head_num: int,
            use_uniform_confidence: bool = False,
            arch_embedder_type: str = "lstm", arch_embedder_cfg = None,
            mlp_hiddens: Tuple[int] = (200, 200, 200), mlp_dropout: float = 0.1,
            optimizer: Dict[str, Union[str, float]] = {
                     "type": "Adam",
                     "lr": 0.001}, 
            scheduler: Optional = None,
            compare_loss_type: str = "margin_linear",
            compare_margin: float = 0.01,
            margin_l2: bool = False,
            use_incorrect_list_only: bool = False,
            tanh_score: bool = None,
            max_grad_norm: float = None,
            schedule_cfg: Optional = None) -> None:
        # [optional] arch reconstruction loss (arch_decoder_type/cfg)
        
        super(DynamicEnsemblePointwiseComparator, self).__init__(
            search_space, arch_embedder_type, arch_embedder_cfg,
            mlp_hiddens, mlp_dropout,
            optimizer, scheduler,
            compare_loss_type, compare_margin,
            margin_l2, use_incorrect_list_only,
            tanh_score, max_grad_norm, schedule_cfg)

        self.auxiliary_head_num = auxiliary_head_num
        self.use_uniform_confidence = use_uniform_confidence

        if self.use_uniform_confidence:
            self.confidence = nn.Parameter(torch.randn((1, self.auxiliary_head_num), requires_grad = True))
            nn.init.constant_(self.confidence, 1. / self.auxiliary_head_num)
        else:
            ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
            self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))
            dim = self.embedding_dim = self.arch_embedder.out_dim
            # construct MLP from architecture embedding to prediction confidence score
            self.confidence_mlp = self.construct_mlp(dim, mlp_hiddens, mlp_dropout, auxiliary_head_num)

        self.module_lst = nn.ModuleList([
            PointwiseComparator(
                search_space,
                arch_embedder_type,
                arch_embedder_cfg,
                mlp_hiddens,
                mlp_dropout,
                optimizer,
                scheduler,
                compare_loss_type,
                compare_margin,
                margin_l2,
                use_incorrect_list_only,
                tanh_score,
                max_grad_norm,
                schedule_cfg
            ) for i in range(self.auxiliary_head_num)
        ])

        # init optimizer and scheduler
        self.reinit_optimizer(only_mlp = False)
        self.reinit_scheduler()
    
    def init_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)
    
    def init_scheduler(self):
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def mtl_update_compare(self, auxiliary_datas, margin = None):
        loss = 0.
        for model, (arch_1, arch_2, better_lst) in zip(self.module_lst, auxiliary_datas):
            s_1 = model.predict(arch_1, False, False)
            s_2 = model.predict(arch_2, False, False)
            loss += self._compare_loss(s_1, s_2, better_lst, margin)
        return self.update_step(loss)

    def update_compare(self, arch_1, arch_2, better_labels, margin = None) -> float:
        score_1 = self.predict(arch_1, False, False)
        score_2 = self.predict(arch_2, False, False)
        loss = self._compare_loss(score_1, score_2, better_labels, margin)
        return self.update_step(loss)

    def mtl_predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tuple[Tensor, List[Tensor]]:
        score = self.predict(arch, sigmoid, tanh)
        lf_score_lst = [model.predict(arch, sigmoid, tanh) for model in self.module_lst]
        return score, lf_score_lst

    def separate_score_predict(self, arch) -> Tensor:
        score_lst = torch.cat([
            model.predict(arch, False, False).unsqueeze(1) 
            for model in self.module_lst],
        1)
        return score_lst

    def weighted_score(self, arch) -> Tensor:
        score_lst = self.separate_score_predict(arch)
        confidence_ratio = self._confidence_ratio(arch)
        score = (confidence_ratio * score_lst)
        return score

    def predict(self, arch, sigmoid: bool = True, tanh: bool = False) -> Tensor:
        weighted_score = self.weighted_score(arch)
        score = weighted_score.sum(1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def _confidence_ratio(self, archs) -> Tensor:
        if self.use_uniform_confidence:
            confidence_ratio = torch.softmax(self.confidence, 1)
        else:
            arch_embeddings = self.arch_embedder(archs)
            confidence_ratio = torch.softmax(self.confidence_mlp(arch_embeddings), 1) # softmax or sigmoid, maybe ablation
        return confidence_ratio

    def _compare_loss(self, s_1, s_2, better_labels, margin = None):
        s_1 = s_1.squeeze()
        s_2 = s_2.squeeze()
        better_pm = 2 * s_1.new(np.array(better_labels, dtype = np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        if not self.margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        return pair_loss

    def update_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return loss.item()


class PairwiseComparator(ArchNetwork, nn.Module):
    """
    Convert the pointwise regression problem to a pairwise classfication problem.
    Do not support predict call.
    """
    NAME = "pairwise_comparator"

    def __init__(self, search_space,
                 arch_embedder_type="lstm", arch_embedder_cfg=None,
                 mlp_hiddens=(200, 200, 200), mlp_dropout=0.1,
                 optimizer={
                     "type": "Adam",
                     "lr": 0.001
                 }, scheduler=None,
                 compare_loss_type="margin_linear",
                 compare_margin=0.01,
                 pairing_method="concat",
                 diff_only=False,
                 train_use_sigmoid=False,
                 sorting_residue_worse_thresh=100,
                 sorting_residue_better_thresh=100,
                 max_grad_norm=None,
                 schedule_cfg=None):
        # [optional] arch reconstruction loss (arch_decoder_type/cfg)
        super(PairwiseComparator, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

        # configs
        expect(compare_loss_type in {"binary_cross_entropy", "margin_linear"},
               "comparing loss type {} not supported".format(compare_loss_type),
               ConfigException)
        self.compare_loss_type = compare_loss_type
        self.compare_margin = compare_margin
        expect(pairing_method in {"concat", "diff"},
               "pairing method {} not supported".format(pairing_method),
               ConfigException)
        self.pairing_method = pairing_method
        self.sorting_residue_worse_thresh = sorting_residue_worse_thresh
        self.sorting_residue_better_thresh = sorting_residue_better_thresh
        self.max_grad_norm = max_grad_norm
        self.train_use_sigmoid = train_use_sigmoid
        self.diff_only = diff_only

        self.search_space = search_space
        ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
        self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim \
                                   if (diff_only and pairing_method == "diff") \
                                   else 2 * self.arch_embedder.out_dim
        # construct MLP from embedding to score
        self.mlp = []
        for hidden_size in mlp_hiddens:
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=mlp_dropout)))
            dim = hidden_size
        self.mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*self.mlp)

        # init optimizer and scheduler
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

    def compare(self, arch_1, arch_2):
        emb_1 = self.arch_embedder(arch_1)
        emb_2 = self.arch_embedder(arch_2)
        if self.pairing_method == "concat":
            emb = torch.cat((emb_1, emb_2), dim=-1)
            score = torch.sigmoid(self.mlp(emb))
            emb = torch.cat((emb_2, emb_1), dim=-1)
            score += 1 - torch.sigmoid(self.mlp(emb))
            score /= 2
        elif self.pairing_method == "diff":
            if not self.diff_only:
                emb = torch.cat((emb_1, emb_2 - emb_1), dim=-1)
                score = torch.sigmoid(self.mlp(emb))
                emb = torch.cat((emb_2, emb_1 - emb_2), dim=-1)
                score += 1 - torch.sigmoid(self.mlp(emb))
                score /= 2
            else:
                emb = emb_2 - emb_1
                score = torch.sigmoid(self.mlp(emb))
        return score.squeeze(-1)

    def _cmp_for_sorted(self, x, y):
        score = self.compare([x], [y]).item()
        return 2 * (score < 0.5) - 1

    def update_compare_rollouts(self, compare_rollouts, better_labels):
        arch_1, arch_2 = zip(*[(r.rollout_1.arch, r.rollout_2.arch) for r in compare_rollouts])
        return self.update_compare(arch_1, arch_2, better_labels)

    def update_compare_list(self, compare_lst):
        # use binary classification loss to step
        arch_1, arch_2, better_labels = zip(*compare_lst)
        return self.update_compare(arch_1, arch_2, better_labels)

    def _get_loss(self, mlp_out, better_labels):
        if self.compare_loss_type == "binary_cross_entropy":
            score = torch.sigmoid(mlp_out)
            pair_loss = F.binary_cross_entropy(
                score,
                score.new(better_labels))
        elif self.compare_loss_type == "margin_linear":
            if self.train_use_sigmoid:
                score = torch.sigmoid(mlp_out)
            else:
                score = mlp_out
            better_pm = 2 * score.new(np.array(better_labels, dtype=np.float32)) - 1
            zero_ = score.new([0.])
            if self.train_use_sigmoid:
                pair_loss = torch.mean(torch.max(zero_, self.compare_margin - \
                                                 better_pm * (2 * score - 1)))
            else:
                pair_loss = torch.mean(torch.max(zero_, self.compare_margin - \
                                                 better_pm * score))
        return pair_loss

    def update_compare(self, arch_1, arch_2, better_labels):
        emb_1 = self.arch_embedder(arch_1)
        emb_2 = self.arch_embedder(arch_2)
        better_labels = np.array(better_labels)
        if self.pairing_method == "concat":
            emb = torch.cat((emb_1, emb_2), dim=-1)
            mlp_out = self.mlp(emb).squeeze()
            pair_loss = self._get_loss(mlp_out, better_labels)
            emb = torch.cat((emb_2, emb_1), dim=-1)
            mlp_out = self.mlp(emb).squeeze()
            pair_loss += self._get_loss(mlp_out, 1 - better_labels)
        elif self.pairing_method == "diff":
            if not self.diff_only:
                emb = torch.cat((emb_1, emb_2 - emb_1), dim=-1)
                mlp_out = self.mlp(emb).squeeze()
                pair_loss = self._get_loss(mlp_out, better_labels)
                emb = torch.cat((emb_2, emb_1 - emb_2), dim=-1)
                mlp_out = self.mlp(emb).squeeze()
                pair_loss += self._get_loss(mlp_out, 1 - better_labels)
            else:
                emb = emb_2 - emb_1
                mlp_out = self.mlp(emb).squeeze()
                pair_loss = self._get_loss(mlp_out, better_labels)
        self.optimizer.zero_grad()
        pair_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return pair_loss.item()

    def compare_with_batchsize(self, arch, archs, batch_size):
        cur_ind = 0
        num_archs = len(archs)
        all_scores = np.zeros(0)
        while cur_ind < num_archs:
            end_ind = min(num_archs, cur_ind + batch_size)
            b_size = end_ind - cur_ind
            all_scores = np.concatenate((
                all_scores, self.compare([arch for _ in range(b_size)],
                                         archs[cur_ind:end_ind]).detach().cpu().numpy()))
            cur_ind = end_ind
        return all_scores

    def argsort_list(self, archs, batch_size, indexes=None):
        # q-sort with random pivots
        # the compares could be batched easily
        archs = np.array(archs)
        if indexes is None:
            indexes = np.arange(len(archs))
            # this random shuffle would influence the final sequence
            np.random.shuffle(indexes)
        # choose a pivot
        pivot_ind = indexes[0]
        scores = self.compare_with_batchsize(
            archs[pivot_ind], archs[indexes[1:]], batch_size=batch_size)
        # could use partition, but i think this is not the bottleneck
        sorted_inds = np.argsort(scores)
        sep_inds = np.where(scores[sorted_inds] > 0.5)[0]
        if len(sep_inds) == 0:
            sep_ind = len(indexes)
        else:
            # the first index that is better than archs[pivot_ind]
            sep_ind = sep_inds[0]
        # indexes of the archs that are worse than archs[pivot_ind]
        worse_inds = indexes[1:][sorted_inds[:sep_ind]]
        # indexes of the archs that are better than archs[pivot_ind]
        better_inds = indexes[1:][sorted_inds[sep_ind:]]
        if len(worse_inds) > self.sorting_residue_worse_thresh:
            # swap the first index with the middle index (possibly uniform seperate)
            worse_inds[0], worse_inds[len(worse_inds) // 2] \
                = worse_inds[len(worse_inds) // 2], worse_inds[0]
            worse_inds = self.argsort_list(archs, batch_size, indexes=worse_inds)
        if len(better_inds) > self.sorting_residue_better_thresh:
            # swap the first index with the middle index (possibly uniform seperate)
            better_inds[0], better_inds[len(better_inds) // 2] \
                = better_inds[len(better_inds) // 2], better_inds[0]
            better_inds = self.argsort_list(archs, batch_size, indexes=better_inds)
        return np.concatenate((worse_inds, [pivot_ind], better_inds))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def on_epoch_start(self, epoch):
        super(PairwiseComparator, self).on_epoch_start(epoch)
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            self.logger.info("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    imp_exception = e
    class RandomForest(ArchNetwork, nn.Module):
        NAME = "random_forest"
        def __new__(cls, *args, **kwargs):
            from aw_nas.utils import getLogger
            getLogger("arch_network").error(
                ("RandomForest arch network cannot be used: Cannot import module sklearn: {}".format(imp_exception)))
            raise Exception()

        def load(self, path):
            pass

        def save(self, path):
            pass
else:
    class RandomForest(ArchNetwork, nn.Module):
        NAME = "random_forest"

        def __init__(self, search_space,
                    arch_embedder_type="seq", arch_embedder_cfg=None,
                    scheduler=None, schedule_cfg=None):
            super(RandomForest, self).__init__(schedule_cfg)
            nn.Module.__init__(self)

            self.search_space = search_space
            ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
            self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))

            self.embedding_dim = self.arch_embedder.out_dim

        def update_predict(self, archs, scores):
            # All architectures should be fed in as one batch
            variance = np.var(scores)
            threshold = 1e-4 * variance
            self.regr = RandomForestRegressor(
                n_estimators=100, criterion="mse",
                max_features=0.5, min_impurity_decrease=threshold)
            self.regr.fit(self.arch_embedder(archs).cpu().numpy(), scores)
            return 0.

        def predict(self, archs):
            scores = self.regr.predict(self.arch_embedder(archs).cpu().numpy())
            return torch.from_numpy(scores).cuda()

        def save(self, path):
            torch.save(self.state_dict(), path)

        def load(self, path):
            self.load_state_dict(torch.load(path, map_loacation=torch.device("cpu")))

        def on_epoch_start(self, epoch):
            pass

        def on_epoch_end(self, epoch):
            pass


class AnyTimePointwiseComparator(PointwiseComparator):
    NAME = "any_time_pointwise_comparator"

    def __init__(self, search_space, seperate_mlp: bool = True,
            arch_embedder_type: str = "lstm", arch_embedder_cfg = None,
            mlp_hiddens: Tuple[int] = (200, 200, 200), mlp_dropout: float = 0.1,
            optimizer: dict = {
                     "type": "Adam",
                     "lr": 0.001
                 }, scheduler = None,
            compare_loss_type: str = "margin_linear",
            compare_margin: float = 0.01,
            margin_l2: bool = False,
            use_incorrect_list_only: bool = False,
            tanh_score: bool = None,
            max_grad_norm: float = None,
            schedule_cfg = None) -> None:
        super(AnyTimePointwiseComparator, self).__init__(
                search_space, arch_embedder_type, arch_embedder_cfg,
                mlp_hiddens, mlp_dropout, optimizer,
                scheduler, compare_loss_type, compare_margin,
                margin_l2, use_incorrect_list_only, tanh_score,
                max_grad_norm, schedule_cfg)

        assert "anytime" in arch_embedder_type, \
            "{} is not supported. Only support any time arch embedder.".format(arch_embedder_type)
        
        self.seperate_mlp = seperate_mlp
        if seperate_mlp:
            dim = self.embedding_dim
            self.pre_mlps = nn.ModuleList([
                    self.construct_mlp(dim, mlp_hiddens, mlp_dropout)
                    for i in range(self.arch_embedder.num_time_steps - 1)
            ])
        
        # init optimizer and scheduler
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

    def reinit_optimizer_scheduler(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def update_predict(self, archs: np.ndarray, labels_list: List[np.ndarray]) -> float:
        r"""
        Update the predictor with regression loss.
        Args:
            archs (np.ndarray): The architecture.
            labels_list (List[np.ndarray]): A list of labels at different time.
        Returns:
            loss (float): The calculated loss.
        """
        arch_embedding_list = self.arch_embedder(archs, any_time = True)
        
        if not self.seperate_mlp:
            scores_list = [
                torch.sigmoid(self.mlp(arch_embedding))
                for arch_embedding in arch_embedding_list
            ]
        
        else:
            scores_list = [
                torch.sigmoid(mlp(arch_emb))
                for mlp, arch_emb in zip(self.pre_mlps, arch_embedding_list[:-1])
            ]
        
        scores_list.append(torch.sigmoid(self.mlp(arch_embedding_list[-1])))

        mse_loss = sum([
            F.mse_loss(scores.squeeze(), scores.new(labels)) 
            for scores, labels in zip(scores_list, labels_list)
        ])
        self.optimizer.zero_grad()
        mse_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return mse_loss.item()

    def predict(self, arch, sigmoid=True, tanh=False, anytime = False):
        activate_func = torch.sigmoid if sigmoid else torch.tanh
        if not anytime:
            score = self.mlp(self.arch_embedder(arch)).squeeze(-1)
            score = activate_func(score)
            return score
        else:
            arch_embedding_list = self.arch_embedder(arch, any_time = True)
        
            if not self.seperate_mlp:
                scores_list = [
                    activate_func(self.mlp(arch_embedding)).squeeze(-1)
                    for arch_embedding in arch_embedding_list
                ]
        
            else:
                scores_list = [
                    activate_func(mlp(arch_emb)).squeeze(-1)
                    for mlp, arch_emb in zip(self.pre_mlps, arch_embedding_list[:-1])
                ]
                scores_list.append(activate_func(self.mlp(arch_embedding_list[-1])).squeeze(-1))
            return scores_list

          
class MultiPredictionPointwiseComparator(PointwiseComparator):
    r"""
    For ablation research of TA-GATES.
    """
    NAME = "multi_prediction_pointwise_comparator"
    
    @staticmethod
    def construct_mlp(dim: int, mlp_hiddens: Tuple[int], mlp_dropout: float) -> nn.Module:
        mlp = []
        for hidden_size in mlp_hiddens:
            mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace = False),
                nn.Dropout(p = mlp_dropout))
            )
            dim = hidden_size
        mlp.append(nn.Linear(dim, 2))
        mlp = nn.Sequential(*mlp)
        return mlp

    def update_predict(self, archs, labels_list):
        all_scores = torch.sigmoid(self.mlp(self.arch_embedder(archs)))
        mse_loss = sum([F.mse_loss(scores.squeeze(), scores.new(labels)) 
            for scores, labels in zip(all_scores.transpose(1, 0), labels_list)])
        self.optimizer.zero_grad()
        mse_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return mse_loss.item()

    def predict(self, arch, sigmoid: bool = True, tanh: bool = False):
        scores = self.mlp(self.arch_embedder(arch))
        if sigmoid:
            scores = torch.sigmoid(scores)
        elif tanh:
            scores = torch.tanh(scores)
            
        return [score.squeeze(-1) for score in scores.transpose(1, 0)]
