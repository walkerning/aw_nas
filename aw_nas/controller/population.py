# -*- coding: utf-8 -*-
"""
Population-based controllers
"""

import os
import abc

import six
import yaml
import numpy as np

from aw_nas import Component, utils
from aw_nas.rollout.mutation import Population, MutationRollout
from aw_nas.common import assert_rollout_type
from aw_nas.controller.base import BaseController
from aw_nas.utils.exception import expect, ConfigException


class BaseMutationSampler(Component):
    REGISTRY = "mutation_sampler"

    def __init__(self, search_space, population, device, schedule_cfg=None):
        super(BaseMutationSampler, self).__init__(schedule_cfg)
        self.search_space = search_space
        self.population = population
        self.device = device

    @abc.abstractmethod
    def sample_mutation(self, model_index, num_mutations=1):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class RandomMutationSampler(BaseMutationSampler):
    NAME = "random"

    def __init__(self, search_space, population, device,
                 mutate_primitive_prob=0.5, schedule_cfg=None):
        super(RandomMutationSampler, self).__init__(search_space, population, device, schedule_cfg)

        self.mutate_primitive_prob = mutate_primitive_prob

    def sample_mutation(self, model_index, num_mutations=1):
        return MutationRollout.random_sample(self.population, model_index,
                                             num_mutations=num_mutations,
                                             primitive_prob=self.mutate_primitive_prob)

    def save(self, path):
        pass

    def load(self, path):
        pass


class PopulationController(BaseController):
    NAME = "population"

    def __init__(self, search_space, device, rollout_type="mutation", mode="eval",
                 score_func="choose('acc')",
                 population_dirs=[], result_population_dir=None,
                 num_mutations_per_child=1,
                 # choose parent
                 parent_pool_size=25,
                 mutation_sampler_type="random", mutation_sampler_cfg=None):
        super(PopulationController, self).__init__(search_space, rollout_type, mode)

        self.device = device
        expect(population_dirs, "Config `population_dirs` should not be empty.",
               ConfigException)
        expect(result_population_dir, "Config `result_population_dir` must be given.",
               ConfigException)
        self.result_population_dir = result_population_dir
        self.num_mutations_per_child = num_mutations_per_child
        self.parent_pool_size = parent_pool_size

        self.population = Population.init_from_dirs(population_dirs, self.search_space)
        self.score_func = self._get_score_func(score_func)
        self.indexes, self.scores = self._init_indexes_and_scores(self.population, self.score_func)

        ms_cls = BaseMutationSampler.get_class_(mutation_sampler_type)
        self.mutation_sampler = ms_cls(self.search_space, self.population, self.device,
                                       **(mutation_sampler_cfg or {}))

    @classmethod
    def _init_indexes_and_scores(cls, population, score_func):
        scores = [-1] * population.next_index
        indexes = []
        for ind, model_record in six.iteritems(population.model_records):
            scores[ind] = score_func(model_record)
            indexes.append(ind)
        return sorted(indexes), scores

    @classmethod
    def _get_score_func(cls, score_func_str):
        return eval("cls._score_func_{}".format(score_func_str))

    @staticmethod
    def _score_func_choose(name, weight=1.):
        def choose_perf(model_record):
            return weight * model_record.perfs[name]
        return choose_perf

    @staticmethod
    def _score_func_weighted_sum(names, weights):
        def weighted_perf(model_record):
            score = 0.
            for name, weight in zip(names, weights):
                score += model_record.perfs[name] * weight
            return score
        return weighted_perf

    def _choose_parent(self, pool_size=None):
        pool_size = pool_size if pool_size is not None else self.parent_pool_size
        select_indexes = np.random.choice(self.indexes, size=pool_size, replace=False)
        max_ = -np.inf
        max_ind = None
        for ind in select_indexes:
            if self.scores[ind] > max_:
                max_ = self.scores[ind]
                max_ind = ind
        return max_ind

    # ---- APIs ----
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def sample(self, n, batch_size=1):
        assert batch_size == 1, "batch_size must equal 1 for population controller"
        rollouts = []
        conflict = 0
        remain = n
        while remain:
            assert conflict < 4 * n, "Mutation conflicts occur too frequently, there might be bugs?"
            parent_index = self._choose_parent()
            rollout = self.mutation_sampler.sample_mutation(parent_index,
                                                            self.num_mutations_per_child)
            if self.population.contain_rollout(rollout):
                conflict += 1
            else:
                rollouts.append(rollout)
                remain -= 1
        return rollouts

    def save(self, path):
        # write new model records to disk
        _ = self.population.save(self.result_population_dir)

        # write the indexes of the model records in the current population to checkpoint
        path = utils.makedir(path)
        with open(os.path.join(path, "indexes.yaml"), "w") as w_f:
            yaml.safe_dump([int(ind) for ind in self.indexes], stream=w_f)

        # save mutation sampler state
        self.mutation_sampler.save(os.path.join(path, "mutation_sampler"))

    def load(self, path):
        # load checkpoint (indexes of the model records in the current population)
        with open(os.path.join(path, "indexes.yaml"), "r") as r_f:
            self.indexes = yaml.safe_load(r_f)
        # load mutation sampler state
        self.mutation_sampler.load(path)

    def step(self, rollouts, optimizer, perf_name):
        """
        Update the performance of the rollouts into the population and assign index.
        Call `sampler.step`.
        Remove some individuals from the population according to specific conditions
        """
        indexes = []
        scores = []
        for rollout in rollouts:
            indexes.append(int(self.population.add_model(rollout.model_record)))
            scores.append(self.score_func(rollout.model_record))
        self.scores = self.scores + [-1] * (max(indexes) - len(self.scores) + 1)
        self.indexes += indexes
        for ind, score in zip(indexes, scores):
            self.scores[ind] = score
            print(score)

        self.logger.info(
            "Add {} model records to population (Current best score: {}):\n".format(
                len(rollouts), max(self.scores)) + \
            "\n\t".join([str(ind) + ": " + str(rollout.model_record)
                         for ind, rollout in zip(indexes, rollouts)]))
        # TODO: aging population?

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("mutation")]

    # ---- Override some components functionality: dispatch to mutation_sampler ----
    def on_epoch_start(self, epoch):
        super(PopulationController, self).on_epoch_start(epoch)
        self.mutation_sampler.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        super(PopulationController, self).on_epoch_end(epoch)
        self.mutation_sampler.on_epoch_end(epoch)

    def setup_writer(self, writer):
        super(PopulationController, self).setup_writer(writer)
        self.mutation_sampler.setup_writer(writer.get_sub_writer("mutation_sampler"))

    @classmethod
    def get_default_config_str(cls):
        # Override. As there are sub-component in PopulationController
        all_str = super(PopulationController, cls).get_default_config_str()
        # Possible mutation_sampler configs
        all_str += utils.component_sample_config_str("mutation_sampler", prefix="#   ") + "\n"
        return all_str
