# pylint: disable=invalid-name,missing-docstring
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aw_nas.utils.exception import expect
from aw_nas.utils import DenseGraphSimpleOpEdgeFlow
from aw_nas.evaluator.arch_network import ArchEmbedder


class DenseRobGATESEmbedder(ArchEmbedder):
    NAME = "denserob-gates"

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
        super(DenseRobGATESEmbedder, self).__init__(schedule_cfg)

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

        self.vertices = self.search_space._num_nodes
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.primitives.index("none")
        expect(self.none_op_ind == 0,
               "DenseGraphSimpleOpEdgeFlow assume `none` to be the first primitive, "
               "if it exists. Or the codes need to be changed to mask the none op")
        self.num_cell_groups = self.search_space.num_cell_groups
        self.num_init_nodes = self.search_space.num_init_nodes

        self.input_node_emb = nn.Parameter(
            torch.FloatTensor(
                self.num_cell_groups, self.num_init_nodes, self.node_embedding_dim
            ).normal_(),
            requires_grad=True,
        )
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
        self.out_dim = in_dim * self.num_cell_groups

    def embed_and_transform_arch(self, archs):
        adjs = self.op_emb.weight.new(archs).long()
        op_embs = self.op_emb(
            adjs
        )  # (batch_size, num_cells, vertices, vertices, op_emb_dim)
        b_size = op_embs.shape[0]
        node_embs = torch.cat(
            (
                self.input_node_emb.unsqueeze(0).repeat([b_size, 1, 1, 1]),
                self.other_node_emb.unsqueeze(0).repeat(
                    [
                        b_size,
                        self.num_cell_groups,
                        self.vertices - self.num_init_nodes,
                        1,
                    ]
                ),
            ),
            dim=2,
        )
        x = self.x_hidden(node_embs)
        # x: (batch_size, num_cells, vertices, hid_dim)
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
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(
                    shape_y
                )
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            # shouldn't be used, since denserob ss concat all the nodes
            y = y[:, -1, :]
        else:
            y = y[
                :, :, self.num_init_nodes :, :
            ]  # do not keep the inputs node embedding
            # average across nodes, then concat across cell (bs, god*numcell)
            y = torch.reshape(torch.mean(y, dim=2), (y.shape[0], -1))
        return y


class DenseRobLSTMEmbedder(ArchEmbedder):
    NAME = "denserob-lstm"

    def __init__(
        self,
        search_space,
        embedding_size=48,
        hidden_size=96,
        dropout_ratio=0.0,
        num_layers=1,
        schedule_cfg=None,
    ):
        super(DenseRobLSTMEmbedder, self).__init__(schedule_cfg)
        self.search_space = search_space
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(len(self.search_space.primitives), self.embedding_size)
        self.rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout_ratio,
        )
        # calculate out dim
        self.out_dim = self.hidden_size

    def embed_and_transform_arch(self, archs):
        if isinstance(archs, (np.ndarray, list, tuple)):
            archs = np.array(archs)
            if archs.ndim == 3:
                archs = np.expand_dims(archs, 0)
            else:
                assert archs.ndim == 4
            archs = self.emb.weight.new(archs).long()
        emb = self.emb(archs)
        return torch.reshape(emb, [emb.shape[0], -1, emb.shape[-1]])

    def forward(self, archs):
        emb = self.embed_and_transform_arch(archs)
        out, _ = self.rnn(emb)
        # normalize the output following NAO
        out = F.normalize(out, 2, dim=-1)
        # average across decisions (time steps)
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        return out
