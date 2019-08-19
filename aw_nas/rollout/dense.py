# -*- coding: utf-8 -*-
"""
Definitions of mutation-based rollout upon densenet-backbone.
"""

import numpy as np

from aw_nas import utils
from aw_nas.rollout.base import Rollout
from aw_nas.rollout.mutation import MutationRollout
from aw_nas.common import SearchSpace
from aw_nas.utils.exception import expect, ConfigException


class DenseSearchSpace(SearchSpace):
    """
    Parameters:
      num_dense_blocks (int): number of blocks, there are no inter-block connections.
      first_ratio (float): channel num ratio of the stem layer w.r.t the average
        growth rate.
      stem_channel (int): channel num of the stem layer.
        Note that one and only one of `first_ratio` and `stem_channel` should be specified.
      bc_mode (bool): whether or not to use bottleneck layers and reduction.
      bc_ratio (int): channel num ratio of the bottleneck layers.
      reduction (float): reduction ratio at transition layer.
    """
    NAME = "cnn_dense"

    def __init__(
            self,
            num_dense_blocks=3,
            first_ratio=2,
            stem_channel=None,
            bc_mode=True,
            bc_ratio=4,
            reduction=0.5,
    ):
        expect((first_ratio is None) + (stem_channel is None) == 1,
               "One and only one of `first_ratio` and `stem_channel` should be specified.",
               ConfigException)

        self.num_dense_blocks = num_dense_blocks
        self.first_ratio = first_ratio
        self.stem_channel = stem_channel
        self.bc_mode = bc_mode
        self.bc_ratio = bc_ratio
        self.reduction = reduction

        self.genotype_type_name = "DenseGenotype"
        self.block_names = sum(
            [["stem"]] + \
            [["block_{}".format(i), "transition_{}".format(i)] if i != self.num_dense_blocks - 1
             else ["block_{}".format(i)] for i in range(self.num_dense_blocks)],
            [])
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, [])

    def __getstate__(self):
        state = super(DenseSearchSpace, self).__getstate__().copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        super(DenseSearchSpace, self).__setstate__(state)
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, [])

    def genotype(self, arch):
        """Convert arch (controller representation) to genotype (semantic representation)"""
        assert len(arch) == self.num_dense_blocks
        if self.first_ratio is not None:
            mean_growth = np.mean(sum(arch, []))
            extend_arch = [int(mean_growth * self.first_ratio)]
        else:
            extend_arch = [self.stem_channel]
        last_channel = extend_arch[0]
        for i_dense, n_cs in enumerate(arch):
            extend_arch.append(n_cs)
            last_channel = int(last_channel + np.sum(n_cs))
            if i_dense != self.num_dense_blocks - 1:
                last_channel = int(self.reduction * last_channel)
                extend_arch.append(last_channel)
        return self.genotype_type(**dict(zip(self.block_names, extend_arch)))

    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""
        genotype_list = list(genotype._asdict().values())
        return DenseDiscreteRollout(
            [genotype_list[i] for i in range(1, 2 * self.num_dense_blocks, 2)],
            {}, self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        raise NotImplementedError()

    def random_sample(self):
        raise NotImplementedError()

    def distance(self, arch1, arch2):
        raise NotImplementedError()


class DenseDiscreteRollout(Rollout):
    @classmethod
    def random_sample_arch(cls, *args, **kwargs):
        raise NotImplementedError()


class DenseMutationRollout(MutationRollout):
    @classmethod
    def random_sample(cls, population, parent_index, num_mutations=1, primitive_prob=0.5):
        """
        Random sample a MutationRollout with mutations.

        Duplication is checked for multiple mutations.
        """
        search_space = population.search_space
        base_arch = search_space.rollout_from_genotype(
            population.get_model(parent_index).genotype).arch

        mutations = []
        # TODO


class DenseMutation(object):
    WIDER = 0
    DEEPER = 1

    def __init__(self, search_space, mutation_type,
                 block_idx, miniblock_idx, modified=None):
        self.search_space = search_space
        self.mutation_type = mutation_type
        self.block_idx = block_idx
        self.miniblock_idx = miniblock_idx
        self.modified = modified

    def apply(self, arch):
        if self.mutation_type == DenseMutation.WIDER:
            # if self.modified <= arch[self.block_idx][self.miniblock_idx] 
            # foxfi: if modified is not bigger than the original channel?
            arch[self.block_idx][self.miniblock_idx] = self.modified
        else: # deeper
            arch[self.block_idx].insert(
                self.miniblock_idx, arch[self.block_idx][self.miniblock_idx])

    def __repr__(self):
        return "DenseMutation(block={}, miniblock={}, {})".format(
            self.block_idx,
            self.miniblock_idx,
            "wider {}".format(self.modified) if self.mutation_type == DenseMutation.WIDER\
            else "deeper"
        )
