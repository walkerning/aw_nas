"""
General rollout and search space
"""

import numpy as np

from collections import namedtuple

from aw_nas import utils
from aw_nas.rollout.base import Rollout
from aw_nas.common import SearchSpace, genotype_from_str
from aw_nas.utils.exception import expect, ConfigException


def GeneralGenotype(args, **kwargs):
    if len(kwargs) > 0:
        raise ValueError("GeneralGenotype supports only position arguments.")
    return list(args)

class GeneralSearchSpace(SearchSpace):
    NAME = "general"

    def __init__(self, primitives, schedule_cfg=None):
        super(GeneralSearchSpace, self).__init__(schedule_cfg)
        self.genotype_type_name = "GeneralGenotype"

        self.genotype_type = GeneralGenotype

    def random_sample(self):
        raise NotImplementedError("It has not been implemented to sample in general search space.")

    def genotype(self, arch):
        """
        -- arch: list of primitive
        -- primitive:
            -- op: str, params: List([number of str])
        
        """
        return self.genotype_type(arch)

    def rollout_from_genotype(self, genotype):
        if isinstance(genotype, str):
            genotype = genotype_from_str(genotype, self)
        return GeneralRollout(genotype, {}, self)
    
    @classmethod
    def supported_rollout_types(cls):
        return ["general"]

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    def plot_arch(self, genotypes, filename, label="", edge_labels=None, plot_format="pdf"):
        raise NotImplementedError()

class GeneralRollout(Rollout):
    NAME = "general"

    def __init__(self, arch, info, search_space, candidate_net=None):
        super(GeneralRollout, self).__init__(arch, info, search_space, candidate_net)

    def genotype_list(self):
        return self.genotype

    @classmethod
    def random_sample_arch(self):
        raise NotImplementedError("It has not been implemented to sample in general search space.")

    