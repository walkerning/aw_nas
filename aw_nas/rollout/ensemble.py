"""
Ensemble search space and rollout definition.
"""

import re
from collections import namedtuple

from aw_nas.common import SearchSpace, genotype_from_str
from aw_nas.rollout.base import BaseRollout
from aw_nas.utils.exception import expect, ConfigException

class EnsembleSearchSpace(SearchSpace):
    NAME = "ensemble"

    def __init__(self, inner_search_space_type=None, inner_search_space_cfg=None, ensemble_size=3):
        super(EnsembleSearchSpace, self).__init__()

        expect(inner_search_space_type is not None and inner_search_space_cfg is not None,
               ConfigException,
               "should specify search space type and cfg")

        self.inner_search_space = SearchSpace.get_class_(inner_search_space_type)(
            **inner_search_space_cfg)

        self.ensemble_size = ensemble_size
        self.genotype_type_name = "EnsembleGenotype"
        self.genotype_type = namedtuple(
            self.genotype_type_name,
            ["arch_{}".format(i) for i in range(self.ensemble_size)])

        self.genotype_str_pattern = r"EnsembleGenotype\({}\)".format(
            ", ".join(["arch_{}=(.+)".format(i) for i in range(self.ensemble_size)])
        )

    def genotype_from_str(self, genotype_str):
        matched = re.match(self.genotype_str_pattern, genotype_str)
        inner_genotypes = [
            genotype_from_str(matched.group(i + 1), self.inner_search_space)
            for i in range(self.ensemble_size)]
        return self.genotype_type(*inner_genotypes)

    def random_sample(self):
        return EnsembleRollout([self.inner_search_space.random_sample()
                                for _ in range(self.ensemble_size)], search_space=self)

    def genotype(self, rollout): #pylint:disable=arguments-differ
        return self.genotype_type(*[r.genotype for r in rollout.rollout_list])

    def rollout_from_genotype(self, genotype):
        return EnsembleRollout([self.inner_search_space.rollout_from_genotype(g) for g in genotype],
                               search_space=self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        file_list = []
        for i in range(self.ensemble_size):
            file_list += self.inner_search_space.plot_arch(
                genotypes[i],
                filename=filename + "_{}".format(i),
                label="Arch {} {}".format(i, label), **kwargs)
        return file_list

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    @classmethod
    def supported_rollout_types(cls):
        return ["ensemble"]

    def mutate(self, rollout, **mutate_kwargs):
        """
        Mutate a rollout to a neighbor rollout in the search space.
        Called by mutation-based controllers, e.g., EvoController.
        """
        return EnsembleRollout([r.mutate(**mutate_kwargs) for r in rollout.rollout_list],
                               search_space=self)


class EnsembleRollout(BaseRollout):
    NAME = "ensemble"
    supported_components = [("trainer", "simple"), ("controller", "evo"), ("evaluator", "mepa")]

    def __init__(self, rollout_list, search_space, candidate_net=None):
        super(EnsembleRollout, self).__init__()

        self.rollout_list = rollout_list
        self.search_space = search_space
        self.candidate_net = candidate_net

        self._genotype = None # calc when need

    def __eq__(self, other):
        return self.genotype == other.genotype

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self)
        return self._genotype

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net


    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(self.genotype,
                                           filename,
                                           label=label)

