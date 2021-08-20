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


__all__ = ["RandomSampleController", "EvoController", "ParetoEvoController"]


class RandomSampleController(BaseController):
    NAME = "random_sample"

    def __init__(self, search_space, device, rollout_type=None, mode="eval",
                 schedule_cfg=None):
        super(RandomSampleController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)
        self.device = device

    def set_device(self, device):
        self.device = device

    def sample(self, n, batch_size=1):
        rollouts = []
        for _ in range(n):
            rollouts.append(self.search_space.random_sample())
        return rollouts

    def step(self, rollouts, optimizer, perf_name):
        return 0.

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return list(BaseRollout.all_classes_().keys())


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
                 avoid_repeat=False,
                 avoid_mutate_repeat=False,
                 # if `avoid_repeat_worst_threshold` mutation cannot go out, raise/return
                 # controlled by `avoid_repeat_fallback`
                 avoid_repeat_worst_threshold=10,
                 avoid_mutate_repeat_worst_threshold=10,
                 avoid_repeat_fallback="return",
                 schedule_cfg=None):
        super(EvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        expect(eval_sample_strategy in {"population", "all", "population_random"},
               "Invalid `eval_sample_strategy` {}, choices: {}".format(
                   eval_sample_strategy, ["population", "all", "population_random"]),
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
        self.avoid_repeat = avoid_repeat
        self.avoid_mutate_repeat = avoid_mutate_repeat
        self.avoid_repeat_worst_threshold = avoid_repeat_worst_threshold
        self.avoid_mutate_repeat_worst_threshold = avoid_mutate_repeat_worst_threshold
        self.avoid_repeat_fallback = avoid_repeat_fallback
        expect(self.avoid_repeat_fallback in {"return", "raise"})

        self.population = collections.OrderedDict()

        # keep track of all seen rollouts and scores
        self._gt_rollouts = []
        self._gt_scores = []

    def reinit(self):
        """
        Clear controller population.

        e.g., Used by predictor-based controller to clean the inner controller's population
        between multiple iterations
        """
        self.logger.info("Reinit called. Clear the population.")
        self.set_init_population([], perf_name=None)
        self._gt_rollouts = []
        self._gt_scores = []

    def set_init_population(self, rollout_list, perf_name):
        # clear the current population
        self.population = collections.OrderedDict()
        for r in rollout_list:
            self.population[r.genotype] = r.get_perf(perf_name)

    def _avoid_repeat_fallback(self, is_mutate=False):
        resample_str_ = "mutate" if is_mutate else "reselect-and-mutate"
        trials = self.avoid_mutate_repeat_worst_threshold \
            if is_mutate else self.avoid_repeat_worst_threshold
        if self.avoid_repeat_fallback == "raise":
            raise Exception(
                "Cannot get a new rollout that is not in the population by {} {} trials.".format(
                    trials, resample_str_))

    def _mutate(self, rollout, **mutate_kwargs):
        if not self.avoid_mutate_repeat:
            return self.search_space.mutate(rollout, **mutate_kwargs)
        for _ in range(self.avoid_mutate_repeat_worst_threshold):
            new_rollout = self.search_space.mutate(rollout, **mutate_kwargs)
            if rollout.genotype not in self.population:
                break
        else:
            self._avoid_repeat_fallback(is_mutate=True)
        return new_rollout

    def _sample_one(self, population_items, population_size, prob=None):
        # random select tournament with size `self.parent_pool_size`
        choices = np.random.choice(np.arange(population_size),
                                   size=self.parent_pool_size, replace=False,
                                   p=prob)
        selected_pop = [population_items[i] for i in choices]
        parent_ind = np.argmax([item[1] for item in selected_pop])
        parent_geno = selected_pop[parent_ind][0]
        parent_rollout = self.search_space.rollout_from_genotype(parent_geno)
        new_rollout = self._mutate(parent_rollout, **self.mutate_kwargs)
        return new_rollout

    def sample(self, n, batch_size=1):
        assert batch_size == 1, "`batch_size` is not meaningful for Evolutionary controller"
        if self.mode == "eval" and self.population:
            if self.eval_sample_strategy == "population":
                # return n archs with best rewards in the population
                genotypes, scores = zip(*list(self.population.items()))
                best_inds = np.argpartition(scores, -n)[-n:]
                rollouts = [self.search_space.rollout_from_genotype(genotypes[ind])
                            .set_perf(scores[ind], "reward")
                            for ind in best_inds]
            elif self.eval_sample_strategy == "all":
                # return n archs with best rewards ever seen
                best_inds = np.argpartition(self._gt_scores, -n)[-n:]
                rollouts = [self._gt_rollouts[ind] for ind in best_inds]
            elif self.eval_sample_strategy == "population_random":
                size = min(len(self.population), n)
                genotypes = list(self.population.keys())
                all_inds = np.arange(len(self.population))
                inds = np.random.choice(all_inds, size=size)
                rollouts = [self.search_space.rollout_from_genotype(genotypes[ind])
                            for ind in inds]
            # if the population size or number of seen rollout is not large enough,
            # fill with random sample
            rollouts += [self.search_space.random_sample() for _ in range(n - len(rollouts))]
            return rollouts

        # if population size does no reach preset `self.population_size`, just random sample
        if len(self.population) < self.population_size:
            self.logger.debug("Population not full, random sample {}".format(n))
            return [self.search_space.random_sample() for _ in range(n)]

        population_items = list(self.population.items())
        population_size = len(self.population)
        rollouts = []
        for _ in range(n):
            if self.avoid_repeat:
                for _ in range(self.avoid_repeat_worst_threshold):
                    new_rollout = self._sample_one(population_items, population_size)
                    if new_rollout.genotype not in self.population:
                        break
                else:
                    self._avoid_repeat_fallback(is_mutate=False)
            else:
                new_rollout = self._sample_one(population_items, population_size)
            rollouts.append(new_rollout)
        return rollouts

    def step(self, rollouts, optimizer=None, perf_name="reward"):
        # update all rollouts into the population
        for rollout in rollouts:
            self.population[rollout.genotype] = rollout.get_perf(perf_name)

        if len(self.population) > self.population_size:
            to_eliminate_num = len(self.population) - self.population_size
            if self.elimination_strategy == "regularized":
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
            "population": {str(k): v for k, v in self.population.items()},
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
            if hasattr(rollout, "__setstate__"):
                rollout.__setstate__(r)
            else:
                rollout.__dict__.update(r)
            self._gt_rollouts.append(rollout)
        self._gt_scores = state["gt_scores"]

    def __getstate__(self):
        state = super(EvoController, self).__getstate__()
        state["population"] = {str(k): v for k, v in state["population"].items()}
        state["gt_rollouts"] = [r.__getstate__() for r in state["_gt_rollouts"]]
        return state

    def __setstate__(self, state):
        super(EvoController, self).__setstate__(state)
        self.population = {genotype_from_str(k, self.search_space): v
                           for k, v in state["population"].items()}
        self._gt_rollouts = []
        for r in state["gt_rollouts"]:
            rollout = self.search_space.random_sample()
            if hasattr(rollout, "__setstate__"):
                rollout.__setstate__(r)
            else:
                rollout.__dict__.update(r)
            self._gt_rollouts.append(rollout)

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
                 avoid_repeat=False,
                 avoid_mutate_repeat=False,
                 # if `avoid_repeat_worst_threshold` mutation cannot go out, raise/return
                 # controlled by `avoid_repeat_fallback`
                 avoid_repeat_worst_threshold=10,
                 avoid_mutate_repeat_worst_threshold=10,
                 avoid_repeat_fallback="return",
                 avoid_repeat_from="population",
                 num_eliminate=10,
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
        self.avoid_repeat = avoid_repeat
        self.avoid_mutate_repeat = avoid_mutate_repeat
        self.avoid_repeat_worst_threshold = avoid_repeat_worst_threshold
        self.avoid_mutate_repeat_worst_threshold = avoid_mutate_repeat_worst_threshold
        self.avoid_repeat_fallback = avoid_repeat_fallback
        self.avoid_repeat_from = avoid_repeat_from
        self.num_eliminate = num_eliminate
        expect(self.avoid_repeat_fallback in {"return", "raise"})
        expect(self.avoid_repeat_from in {"population", "gt_population"})

        # after initial random sampling, only pareto front points are saved in the population
        self.population = collections.OrderedDict()
        self.gt_population = collections.OrderedDict()

        # whether or not sampling by mutation from pareto front has started
        self._start_pareto_sample = False

    def _avoid_repeat_fallback(self, is_mutate=False):
        resample_str_ = "mutate" if is_mutate else "reselect-and-mutate"
        trials = self.avoid_mutate_repeat_worst_threshold \
            if is_mutate else self.avoid_repeat_worst_threshold
        if self.avoid_repeat_fallback == "raise":
            raise Exception(
                "Cannot get a new rollout that is not in the population by {} {} trials.".format(
                    trials, resample_str_))

    def _mutate(self, rollout, **mutate_kwargs):
        if not self.avoid_mutate_repeat:
            return self.search_space.mutate(rollout, **mutate_kwargs)
        for _ in range(self.avoid_mutate_repeat_worst_threshold):
            new_rollout = self.search_space.mutate(rollout, **mutate_kwargs)
            if rollout.genotype not in self.population:
                break
        else:
            self._avoid_repeat_fallback(is_mutate=True)
        return new_rollout

    def _sample_one(self, population_items, population_size, prob=None):
        choices = np.random.choice(np.arange(population_size), size=1,
                                   replace=False, p=prob)
        parent_geno = population_items[choices[0]][0]
        parent_rollout = self.search_space.rollout_from_genotype(parent_geno)
        new_rollout = self._mutate(parent_rollout, **self.mutate_kwargs)
        return new_rollout

    def sample(self, n, batch_size=1):
        if self.mode == "eval":
            self.pareto_frontier = self.find_pareto_opt(self.population)
            if self.eval_sample_strategy == "all":
                # return all archs on the pareto curve,
                # note that number of sampled rollouts does not necessarily equals `n`
                choices = self.pareto_frontier.items()
            elif self.eval_sample_strategy == "n" and self.pareto_frontier:
                # return only `n` random samples on the pareto curve
                choices = np.random.choice(zip(*list(self.pareto_frontier.items())),
                                           size=min(n, len(self.pareto_frontier)), replace=False)
            rollouts = [self.search_space.rollout_from_genotype(geno)
                        .set_perfs(dict(zip(self.perf_names, perfs)))
                        for geno, perfs in choices]
            # fill to n rollouts by random samples, note that these rollouts are not evaluated
            rollouts += [self.search_space.random_sample()
                         for _ in range(n - len(rollouts))]
            return rollouts

        if not self._start_pareto_sample and len(self.population) < self.init_population_size:
            return [self.search_space.random_sample() for _ in range(n)]

        rollouts = []
        population_items = list(self.population.items())
        prob = None

        # FIXME: This logic is messed up... fix it!!!!!!
        # (FURTHUR) And use a data structure to maintain the pareto front?
        # if len(self.population) > self.init_population_size:
        #     pareto_frontier = self.find_pareto_opt()
        #     distances = self._distance_from_pareto(pareto_frontier)
        #     to_eliminate_num = len(self.population) - self.init_population_size
        #     indices = np.argpartition(distances, to_eliminate_num)[:to_eliminate_num]
        #     indices = np.argsort(distances)
        #     indices = indices[:self.init_population_size -
        #                       min(int(self.init_population_size / 5), self.num_eliminate)]
        #     population_items = [population_items[i] for i in indices]
        #     exp = np.exp(-distances[indices])
        #     prob = exp / exp.sum()
        population_size = len(population_items)

        for _ in range(n):
            if self.avoid_repeat:
                for _ in range(self.avoid_repeat_worst_threshold):
                    new_rollout = self._sample_one(population_items,
                                                   population_size, prob)
                    pool = self.population if self.avoid_repeat_from == \
                        "population" else self.gt_population
                    if new_rollout.genotype not in pool:
                        break
                else:
                    self._avoid_repeat_fallback(is_mutate=False)
            else:
                new_rollout = self._sample_one(population_items,
                                               population_size, prob)
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
                self.gt_population[rollout.genotype] = \
                    self.population[rollout.genotype]
            if len(self.population) >= self.init_population_size:
                # finish random sample, start mutation from pareto front
                self._start_pareto_sample = True
        else:
            # only save the pareto front in the population
            for rollout in rollouts:
                r_perf = np.array([
                    rollout.get_perf(perf_name) for perf_name in self.perf_names])
                self.gt_population[rollout.genotype] = r_perf
                for p_perf in self.population.values():
                    if np.all(r_perf < p_perf):
                        break
                else:
                    # if no existing is better than this rollout on all perfs, add it to population
                    self.population[rollout.genotype] = r_perf
        return 0

    def _euclidean_distance(self, points_a, points_b):
        """
        Calculate the distance between N vectors and M vectors respectively
        The dimension of each vector is K.

        A: shape (N, K)
        B: shape (M, K)

        A * B^T = C, c_ij = \sum_k {a_ik * b_jk}, i.e. the dot of the ith
        vector in A and the jth vector in B.

        A_mode_sq: shape (N, 1), i.e. the square of the mode of each vector in A.
        B_mode_sq: shape (M, 1), i.e. the square of the mode of each vector in B.

        Then repeat them to the same shape:
        A': shape(N, M), a'_ij = mode(vec_a_i) ** 2
        B': shape(M, N), b'_ij = mode(vec_b_j) ** 2

        A' + B' - 2 * A * B^T = D:
        d_ij = mode(vec_a_i) ** 2 + mode(vec_b_j) ** 2 - 2 * \sum_k {a_ik * b_jk}
             = mode(vec_a_i - vec_b_j) ** 2
        """
        assert len(points_a.shape) == 2
        assert len(points_b.shape) == 2

        transpose_b = points_b.T
        dot = np.dot(points_a, transpose_b)

        a_mode_sq = np.tile(
            (points_a ** 2).sum(-1, keepdims=True), (1, points_b.shape[0]))
        b_mode_sq = np.tile((transpose_b ** 2).sum(0, keepdims=True),
                            (points_a.shape[0], 1))

        distance = np.sqrt(a_mode_sq + b_mode_sq - 2 * dot)
        return distance

    def _distance_from_pareto(self, pareto):
        pareto = np.array(sorted(pareto.values(), key=lambda x: tuple(x)))
        perfs = np.array(list(self.population.values()))
        distances = self._euclidean_distance(perfs, pareto).min(-1)
        return distances

    @classmethod
    def find_pareto_opt(cls, population):
        pop_keys = list(population.keys())
        pop_size = len(pop_keys)
        population = {k: v for k, v in population.items()}
        for ind1 in range(pop_size):
            key1 = pop_keys[ind1]
            if key1 not in population:
                continue
            for ind2 in range(ind1, pop_size):
                key2 = pop_keys[ind2]
                if key2 not in population:
                    continue
                diff_12 = population[key1] - population[key2]
                if np.all(diff_12 > 0):
                    # arch 1 is better than arch 2 on all perfs
                    population.pop(key2)
                elif np.all(diff_12 < 0):
                    # arch 2 is better than arch 1 on all perfs
                    population.pop(key1)
                    break
        return population

    def save(self, path):
        state = {
            "epoch": self.epoch,
            "population": {str(k): v for k, v in self.population.items()},
            "gt_population": {str(k): v for k, v in
                              self.gt_population.items()},
            "_start_pareto_sample": self._start_pareto_sample
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device("cpu"))
        self.epoch = state["epoch"]
        self.population = {genotype_from_str(k, self.search_space): v
                           for k, v in state["population"].items()}
        self.gt_population = {genotype_from_str(k, self.search_space): v
                              for k, v in state["gt_population"].items()}
        self._start_pareto_sample = state["_start_pareto_sample"]

    def __getstate__(self):
        state = super(ParetoEvoController, self).__getstate__()
        state["population"] = {str(k): v for k, v in state["population"].items()}
        state["gt_population"] = {str(k): v for k, v in state["gt_population"].items()}
        return state

    def __setstate__(self, state):
        super(ParetoEvoController, self).__setstate__(state)
        self.population = {genotype_from_str(k, self.search_space): v
                           for k, v in state["population"].items()}
        self.gt_population = {genotype_from_str(k, self.search_space): v
                              for k, v in state["gt_population"].items()}

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
