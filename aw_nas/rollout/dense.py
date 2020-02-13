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
            transition_channels=None,
            dynamic_transition=False
    ):
        super(DenseSearchSpace, self).__init__()

        expect((first_ratio is None) + (stem_channel is None) == 1,
               "One and only one of `first_ratio` and `stem_channel` should be specified.",
               ConfigException)
        expect(dynamic_transition or (transition_channels is None) + (reduction is None) == 1,
               "One and only one of `transition_channels` and `reduction` should be specified.",
               ConfigException)
        if dynamic_transition:
            expect(reduction is None and transition_channels is None,
                   "When `dynamic_transition` is true, note that `reduction` and "
                   "`transition_channels` should not be specified",
                   ConfigException)

        self.num_dense_blocks = num_dense_blocks
        self.first_ratio = first_ratio
        self.stem_channel = stem_channel
        self.bc_mode = bc_mode
        self.bc_ratio = bc_ratio
        self.reduction = reduction
        self.transition_channels = transition_channels
        self.dynamic_transition = dynamic_transition

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
        if self.dynamic_transition:
            assert len(arch) == 2 and len(arch[0]) == self.num_dense_blocks and \
                len(arch[1]) == self.num_dense_blocks - 1
            arch, trans_arch = arch
        else:
            assert len(arch) == self.num_dense_blocks
        if self.first_ratio is not None:
            mean_growth = np.mean(sum(arch, []))
            extend_arch = [int(mean_growth * self.first_ratio)]
        else:
            extend_arch = [self.stem_channel]
        last_channel = extend_arch[0]
        for i_dense, n_cs in enumerate(arch):
            extend_arch.append(list(n_cs))
            last_channel = int(last_channel + np.sum(n_cs))
            if i_dense != self.num_dense_blocks - 1:
                if self.dynamic_transition:
                    last_channel = trans_arch[i_dense]
                else:
                    if self.transition_channels:
                        last_channel = self.transition_channels[i_dense]
                    else:
                        last_channel = int(self.reduction * last_channel)
                extend_arch.append(last_channel)
        return self.genotype_type(**dict(zip(self.block_names, extend_arch)))

    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""
        genotype_list = list(genotype._asdict().values())
        if self.dynamic_transition:
            return DenseDiscreteRollout(
                [
                    [genotype_list[i] for i in range(1, 2 * self.num_dense_blocks, 2)],
                    [genotype_list[i] for i in range(2, 2 * self.num_dense_blocks, 2)],
                ],
                {}, self)
        return DenseDiscreteRollout(
            [genotype_list[i] for i in range(1, 2 * self.num_dense_blocks, 2)],
            {}, self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        raise NotImplementedError()

    def random_sample(self):
        raise NotImplementedError()

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    def relative_conv_flops(self, arch):
        if self.dynamic_transition:
            assert len(arch) == 2 and len(arch[0]) == self.num_dense_blocks and \
                len(arch[1]) == self.num_dense_blocks - 1
            arch, trans_arch = arch
        else:
            assert len(arch) == self.num_dense_blocks

        flops = 0
        last_channel = 3
        # stem
        if self.first_ratio is not None:
            mean_growth = np.mean(sum(arch, []))
            out_c = int(mean_growth * self.first_ratio)
        else:
            out_c = self.stem_channel
        flops += 3 * 3 * last_channel * out_c # stem kernel size = 3
        last_channel = out_c
        mult = 1.
        for i_dense, n_cs in enumerate(arch):
            for growth in n_cs:
                if self.bc_mode:
                    out_c = int(self.bc_ratio * growth)
                    flops += mult * 1 * 1 * last_channel * out_c # bc kernel size = 1
                    input_c = out_c
                else:
                    input_c = last_channel
                flops += mult * 3 * 3 * input_c * growth # kernel size = 3
                last_channel += growth
            if i_dense != self.num_dense_blocks - 1:
                # avg pool 2x2
                if self.dynamic_transition:
                    out_c = trans_arch[i_dense]
                else:
                    if self.transition_channels:
                        out_c = self.transition_channels[i_dense]
                    else:
                        out_c = int(self.reduction * last_channel)
                flops += mult * 1 * 1 * last_channel * out_c
                last_channel = out_c
                mult *= 0.25
        return flops * 2

    @classmethod
    def supported_rollout_types(cls):
        return ["dense_discrete", "dense_mutation"]

class DenseDiscreteRollout(Rollout):
    NAME = "dense_discrete"

    def relative_conv_flops(self):
        return self.search_space.relative_conv_flops(self.arch)

    @classmethod
    def random_sample_arch(cls, *args, **kwargs):
        raise NotImplementedError()


class DenseMutationRollout(MutationRollout):
    NAME = "dense_mutation"

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
    TRANSITION = 2

    def __init__(self, search_space, mutation_type,
                 block_idx, miniblock_idx, modified=None):
        self.search_space = search_space
        self.mutation_type = mutation_type
        self.block_idx = block_idx
        self.miniblock_idx = miniblock_idx
        self.modified = modified

    def apply(self, arch):
        if self.search_space.dynamic_transition:
            if self.mutation_type == DenseMutation.WIDER:
                # if self.modified <= arch[self.block_idx][self.miniblock_idx]
                # foxfi: if modified is not bigger than the original channel?
                arch[0][self.block_idx][self.miniblock_idx] = self.modified
            elif self.mutation_type == DenseMutation.DEEPER: # deeper
                arch[0][self.block_idx].insert(
                    self.miniblock_idx, arch[self.block_idx][self.miniblock_idx])
            else: # transition
                # NOTE: miniblock_idx is ignored
                arch[1][self.block_idx] = self.modified
        else:
            if self.mutation_type == DenseMutation.WIDER:
                # if self.modified <= arch[self.block_idx][self.miniblock_idx]
                # foxfi: if modified is not bigger than the original channel?
                arch[self.block_idx][self.miniblock_idx] = self.modified
            elif self.mutation_type == DenseMutation.DEEPER: # deeper
                arch[self.block_idx].insert(
                    self.miniblock_idx, arch[self.block_idx][self.miniblock_idx])
            else: # transition
                raise Exception("search space with `dynamic_transition`==False "
                                "cannot have transition mutation type")

    def __repr__(self):
        return "DenseMutation(block={}, miniblock={}, {})".format(
            self.block_idx,
            self.miniblock_idx,
            "wider {}".format(self.modified) if self.mutation_type == DenseMutation.WIDER\
            else "deeper"
        )
