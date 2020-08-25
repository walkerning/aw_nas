# -*- coding: utf-8 -*-
"""
Evolutionary controller.
"""

import collections

import numpy as np
import torch

from aw_nas.common import BaseRollout, genotype_from_str
from aw_nas.controller.base import BaseController
from aw_nas.utils.exception import expect, ConfigException

class EvoController(BaseController):
    """
    A single-objective (reward) evolutionary controller.

    Arguments:
      parent_pool_size: tournament size
      eval_sample_strategy: `population` - return best N archs in the population,
                            `all` - return best N archs ever seen (including those been removed)
      elimination_strategy: `regularized` - return oldest archs (regularized evolution),
                            `perf` - return worst-performing archs
      mutate_kwargs: dict of keyword arguments that will be passed to `search_space.mutate`
    """
    NAME = "evo"

    def __init__(self, search_space, device, rollout_type=None, mode="eval",
                 population_size=100, parent_pool_size=10, mutate_kwargs={},
                 eval_sample_strategy="population", elimination_strategy="regularized",
                 schedule_cfg=None):
        super(EvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        expect(eval_sample_strategy in {"population", "all"},
               "Invalid `eval_sample_strategy` {}, choices: {}".format(
                   eval_sample_strategy, ["population", "all"]),
               ConfigException)
        expect(elimination_strategy in {"regularized", "perf"},
               "Invalid `elimination_strategy` {}, choices: {}".format(
                   elimination_strategy, ["regularized", "perf"]),
               ConfigException)

        self.population_size = population_size
        self.parent_pool_size = parent_pool_size
        self.mutate_kwargs = mutate_kwargs
        self.eval_sample_strategy = eval_sample_strategy
        self.elimination_strategy = elimination_strategy
        self.population = collections.OrderedDict()

        # keep track of all seen rollouts and scores
        self._gt_rollouts = []
        self._gt_scores = []

    def set_init_population(self, rollout_list, perf_name):
        # clear the current population
        self.population = collections.OrderedDict()
        for r in rollout_list:
            self.population[r.genotype] = r.get_perf(perf_name)

    def sample(self, n, batch_size=1):
        assert batch_size == 1, "`batch_size` is not meaningful for Evolutionary controller"
        if self.mode == "eval" and len(self.population) > 0:
            if self.eval_sample_strategy == "population":
                # return n archs with best rewards in the population
                genotypes, scores = zip(*list(self.population.items()))
                best_inds = np.argpartition(scores, -n)[-n:]
                rollouts = [self.search_space.rollout_from_genotype(genotypes[ind])\
                            .set_perf(scores[ind], "reward")
                            for ind in best_inds]
            elif self.eval_sample_strategy == "all":
                # return n archs with best rewards ever seen
                best_inds = np.argpartition(self._gt_scores, -n)[-n:]
                rollouts = [self._gt_rollouts[ind] for ind in best_inds]
            # if the population size or number of seen rollout is not large enough,
            # fill with random sample
            rollouts += [self.search_space.random_sample() for _ in range(n - len(rollouts))]
            return rollouts

        # if population size does no reach preset `self.population_size`, just random sample
        if len(self.population) < self.population_size:
            self.logger.info("Population not full, random sample {}".format(n))
            return [self.search_space.random_sample() for _ in range(n)]

        rollouts = []
        population_items = list(self.population.items())
        population_size = len(self.population)
        for _ in range(n):
            # random select tournament with size `self.parent_pool_size`
            choices = np.random.choice(np.arange(population_size),
                                       size=self.parent_pool_size, replace=False)
            selected_pop = [population_items[i] for i in choices]
            parent_ind = np.argmax([item[1] for item in selected_pop])
            parent_geno = selected_pop[parent_ind][0]
            parent_rollout = self.search_space.rollout_from_genotype(parent_geno)
            new_rollout = self.search_space.mutate(parent_rollout, **self.mutate_kwargs)
            rollouts.append(new_rollout)
        return rollouts

    def step(self, rollouts, optimizer=None, perf_name="reward"):
        # update all rollouts into the population
        for rollout in rollouts:
            self.population[rollout.genotype] = rollout.get_perf(perf_name)

        if len(self.population) > self.population_size:
            if self.elimination_strategy == "regularized":
                to_eliminate_num = len(self.population) - self.population_size
                # eliminate according to age (Regularized Evolution)
                for key in list(self.population.keys())[:to_eliminate_num]:
                    self.population.pop(key)
            elif self.elimination_strategy == "perf":
                # eliminate according to perf.
                genotypes, scores = zip(*list(self.population.items()))
                eliminate_inds = np.argpartition(scores, to_eliminate_num)[:to_eliminate_num]
                for ind in eliminate_inds:
                    self.population.pop(genotypes[ind])

        # Also, save all seen rollouts into _gt_rollouts
        self._gt_rollouts = self._gt_rollouts + rollouts
        self._gt_scores = self._gt_scores + [r.get_perf(perf_name) for r in rollouts]

        return 0

    def save(self, path):
        state = {
            "epoch": self.epoch,
            "population": {str(k):v for k, v in self.population.items()},
            "gt_rollouts": [r.__getstate__() for r in self._gt_rollouts],
            "gt_scores": self._gt_scores
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device("cpu"))

        self.epoch = state["epoch"]
        self.population = {genotype_from_str(k, self.search_space): v
                           for k, v in state["population"].items()}
        self._gt_rollouts = []
        for r in state["gt_rollouts"]:
            rollout = self.search_space.random_sample()
            rollout.__setstate__(r)
            self._gt_rollouts.append(rollout)
        self._gt_scores = state["gt_scores"]

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def set_device(self, device):
        pass

    @classmethod
    def supported_rollout_types(cls):
        """
        Search space must implement `mutate` method to support evo controller.
        """
        return list(BaseRollout.all_classes_().keys())


class ParetoEvoController(BaseController):
    """
    A controller that samples new rollouts by mutating from points on the pareto front only.
    The first `init_population_size` archs are random sampled.
    """
    NAME = "pareto-evo"

    def __init__(self, search_space, device, rollout_type=None,
                 mode="eval", init_population_size=100, perf_names=["reward"],
                 mutate_kwargs={},
                 eval_sample_strategy="all",
                 schedule_cfg=None):
        super(ParetoEvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        expect(eval_sample_strategy in {"all", "n"},
               "Invalid `eval_sample_strategy` {}, choices: {}".format(
                   eval_sample_strategy, ["all", "n"]), ConfigException)
        self.init_population_size = init_population_size
        self.perf_names = perf_names
        self.mutate_kwargs = mutate_kwargs
        self.eval_sample_strategy = eval_sample_strategy

        # after initial random sampling, only pareto front points are saved in the population
        self.population = collections.OrderedDict()
        # whether or not sampling by mutation from pareto front has started
        self._start_pareto_sample = False

    def sample(self, n, batch_size=1):
        if self.mode == "eval":
            if self.eval_sample_strategy == "all":
                # return all archs on the pareto curve,
                # note that number of sampled rollouts does not necessarily equals `n`
                choices = self.population.items()
            elif self.eval_sample_strategy == "n" and len(self.population) > 0:
                # return only `n` random samples on the pareto curve
                choices = np.random.choice(zip(*list(self.population.items())), size=min(n, len(self.population)), replace=False)
            rollouts = []
            for geno, perfs in choices:
                rollout = self.search_space.rollout_from_genotype(geno)
                for name, perf in zip(self.perf_names, perfs):
                    rollout.set_perf(perf, name)
                rollouts += [rollout]
            rollouts += [self.search_space.random_sample() for _ in range(n - len(rollouts))]
            return rollouts

        if not self._start_pareto_sample and len(self.population) < self.init_population_size:
            return [self.search_space.random_sample() for _ in range(n)]

        rollouts = []
        population_items = list(self.population.items())
        population_size = len(self.population)
        for _ in range(n):
            choices = np.random.choice(np.arange(population_size), size=1, replace=False)
            parent_geno = population_items[choices[0]][0]
            parent_rollout = self.search_space.rollout_from_genotype(parent_geno)
            new_rollout = self.search_space.mutate(parent_rollout, **self.mutate_kwargs)
            rollouts.append(new_rollout)
        return rollouts

    def step(self, rollouts, optimizer=None, perf_name="reward"):
        """
        Note that `perf_name` argument will be ignored.
        Use `perf_names` in cfg file/`__init__` call to configure.
        """
        if not self._start_pareto_sample:
            # save all perfs in the population
            for rollout in rollouts:
                self.population[rollout.genotype] = np.array([
                    rollout.get_perf(perf_name) for perf_name in self.perf_names])

            if len(self.population) >= self.init_population_size:
                # finish random sample, start mutation from pareto front
                self._start_pareto_sample = True
                self._remove_non_pareto()
        else:
            # only save the pareto front in the population
            for rollout in rollouts:
                r_perf = np.array([
                    rollout.get_perf(perf_name) for perf_name in self.perf_names])
                for p_perf in self.population.values():
                    if np.all(r_perf < p_perf):
                        break
                else:
                    # if no existing is better than this rollout on all perfs, add it to population
                    self.population[rollout.genotype] = r_perf
        return 0

    def _remove_non_pareto(self):
        pop_keys = list(self.population.keys())
        pop_size = len(pop_keys)
        for ind1 in range(pop_size):
            key1 = pop_keys[ind1]
            if key1 not in self.population:
                continue
            for ind2 in range(ind1, pop_size):
                key2 = pop_keys[ind2]
                if key2 not in self.population:
                    continue
                diff_12 = self.population[key1] - self.population[key2]
                if np.all(diff_12 > 0):
                    # arch 1 is better than arch 2 on all perfs
                    self.population.pop(key2)
                elif np.all(diff_12 < 0):
                    # arch 2 is better than arch 1 on all perfs
                    self.population.pop(key1)
                    break

    def save(self, path):
        state = {
            "epoch": self.epoch,
            "population": {str(k): v for k, v in self.population.items()},
            "_start_pareto_sample": self._start_pareto_sample
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device("cpu"))
        self.epoch = state["epoch"]
        self.population = {genotype_from_str(k, self.search_space): v
                           for k, v in state["population"].items()}
        self._start_pareto_sample = state["_start_pareto_sample"]

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def set_device(self, device):
        pass

    @classmethod
    def supported_rollout_types(cls):
        """
        Search space must implement `mutate` method to support evo controller.
        """
        return list(BaseRollout.all_classes_().keys())
