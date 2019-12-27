"""
Networks that take architectures as inputs.
"""

import abc

import torch
from torch import nn

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

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class LSTMArchEmbedder(ArchEmbedder):
    NAME = "lstm"

    def __init__(self, search_space, schedule_cfg=None):
        super(LSTMArchEmbedder, self).__init__(schedule_cfg)
        self.search_space = search_space
        # calculate out dim
        self.out_dim = 100

    def forward(self, arch):
        # TODO
        return 1


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


class PointwiseComparator(ArchNetwork):
    """
    Compatible to regression-based predictor of architecture performance.
    """
    NAME = "pointwise_comparator"

    def __init__(self, search_space,
                 arch_embedder_type="lstm", arch_embedder_cfg=None,
                 schedule_cfg=None):
        # [optional] arch reconstruction loss (arch_decoder_type/cfg)
        super(PointwiseComparator, self).__init__(schedule_cfg)

        self.search_space = search_space
        ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
        self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))

        self.embedding_dim = self.arch_embedder.out_dim
        # TODO: construct MLP from embedding to score
        self.mlp = None

    def predict(self, arch):
        score = self.mlp(self.arch_embedder(arch))
        return score

    def compare(self, arch_1, arch_2):
        # pointwise score and compare
        # s_1 = self.mlp(self.arch_embedder(arch_1))
        # s_2 = self.mlp(self.arch_embedder(arch_2))
        s_1 = 2
        s_2 = 3
        return s_2 > s_1

    def update_predict(self, predict_lst):
        # TODO: use regression loss to step
        pass

    def update_compare(self, compare_lst):
        # TODO: use binary classification loss to step
        pass

    def save(self, path):
        self.arch_embedder.save("{}-embedder".format(path))
        torch.save(self.mlp, "{}-mlp".format(path))

    def load(self, path):
        self.arch_embedder.load("{}-embedder".format(path))
        self.mlp = torch.load("{}-mlp".format(path))

