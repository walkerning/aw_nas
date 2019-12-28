"""
Networks that take architectures as inputs.
"""

import abc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import utils
from aw_nas.base import Component

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
        self.load_state_dict(torch.load(path))


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
        self._one_param = next(self.parameters())

        # calculate out dim
        self.out_dim = self.hidden_size

    def embed_and_transform_arch(self, archs):
        if isinstance(archs, (np.ndarray, list, tuple)):
            archs = np.array(archs)
            if archs.ndim == 3:
                archs = np.expand_dims(archs, 0)
            else:
                assert archs.ndim == 4
            archs = torch.tensor(archs).to(self._one_param.device)

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
        # FIXME: do not know why
        out = F.normalize(out, 2, dim=-1)

        # average across decisions (time steps)
        out = torch.mean(out, dim=1)
        # FIXME: normalzie again, why?
        out = F.normalize(out, 2, dim=-1)
        return out


class GCNArchEmbedder(ArchEmbedder):
    NAME = "gcn"

    def __init__(self, search_space, schedule_cfg=None):
        super(GCNArchEmbedder, self).__init__(schedule_cfg)
        self.search_space = search_space
        # calculate out dim
        self.out_dim = 100

    def forward(self, arch):
        # TODO
        return 1


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
                 schedule_cfg=None):
        # [optional] arch reconstruction loss (arch_decoder_type/cfg)
        super(PointwiseComparator, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

        self.search_space = search_space
        ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
        self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim
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
        self._one_param = next(self.parameters())

    def predict(self, arch):
        score = torch.sigmoid(self.mlp(self.arch_embedder(arch)))
        return score

    def update_predict(self, predict_lst):
        # use MSE regression loss to step
        scores = self.mlp(self.arch_embedder([item[0] for item in predict_lst]))
        mse_loss = F.mse_loss(
            scores.squeeze(),
            torch.tensor([item[1] for item in predict_lst]).to(self._one_param.device))
        mse_loss.backward()
        self.optimizer.step()
        return mse_loss.item()

    def compare(self, arch_1, arch_2):
        # pointwise score and comparen
        s_1 = self.mlp(self.arch_embedder(arch_1))
        s_2 = self.mlp(self.arch_embedder(arch_2))
        return torch.sigmoid(s_2 - s_1)

    def update_compare(self, compare_lst):
        # use binary classification loss to step
        arch_1, arch_2, better_labels = zip(*compare_lst)
        compare_score = self.compare(arch_1, arch_2)
        # criterion = nn.BCELoss()
        # pair_loss = criterion(compare_score, better_labels)
        pair_loss = F.binary_cross_entropy(
            compare_score.squeeze(),
            torch.tensor(better_labels).to(self._one_param.device))
        pair_loss.backward()
        self.optimizer.step()
        return pair_loss.item()

    def save(self, path):
        self.arch_embedder.save("{}-embedder".format(path))
        torch.save(self.mlp, "{}-mlp".format(path))

    def load(self, path):
        self.arch_embedder.load("{}-embedder".format(path))
        self.mlp = torch.load("{}-mlp".format(path))

    def on_epoch_start(self, epoch):
        super(PointwiseComparator, self).on_epoch_start(epoch)
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            self.logger.info("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

