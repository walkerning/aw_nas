# -*- coding: utf-8 -*-

from aw_nas.controller.population import BaseMutationSampler

class DenseMutationSampler(BaseMutationSampler):
    NAME = "dense_rl"

    def __init__(self, search_space, population, device, schedule_cfg=None):
        pass

    def sample_mutation(self, model_index, num_mutations=1):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
