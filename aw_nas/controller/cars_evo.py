# pylint: disable=invalid-name
import random
import collections

import numpy as np
import torch

from aw_nas.common import genotype_from_str
from aw_nas.controller.evo import ParetoEvoController
from aw_nas.utils.exception import expect, ConfigException


# ---- Code from VEGA https://github.com/huawei-noah/vega/ ----
def Dominates(x, y):
    """Check if x dominates y.

    :param x: a sample
    :type x: array
    :param y: a sample
    :type y: array
    """
    return np.all(x <= y) & np.any(x < y)


def NonDominatedSorting(pop):
    """Perform non-dominated sorting.

    :param pop: the current population
    :type pop: array
    """
    _, npop = pop.shape
    rank = np.zeros(npop)
    dominatedCount = np.zeros(npop)
    dominatedSet = [[] for i in range(npop)]
    F = [[]]
    for i in range(npop):
        for j in range(i + 1, npop):
            p = pop[:, i]
            q = pop[:, j]
            if Dominates(p, q):
                dominatedSet[i].append(j)
                dominatedCount[j] += 1
            if Dominates(q, p):
                dominatedSet[j].append(i)
                dominatedCount[i] += 1
        if dominatedCount[i] == 0:
            rank[i] = 1
            F[0].append(i)
    k = 0
    while 1:
        Q = []
        for i in F[k]:
            p = pop[:, i]
            for j in dominatedSet[i]:
                dominatedCount[j] -= 1
                if dominatedCount[j] == 0:
                    Q.append(j)
                    rank[j] = k + 1
        if not Q:
            break
        F.append(Q)
        k += 1
    return F


def CARS_NSGA(target, objs, N):
    """pNSGA-III (CARS-NSGA).

    :param target: the first objective, e.g. accuracy
    :type target: array
    :param objs: the other objective, e.g. FLOPs, number of parameteres
    :type objs: array
    :param N: number of population
    :type N: int
    :return: The selected samples
    :rtype: array
    """
    selected = np.zeros(target.shape[0])
    Fs = []
    for obj in objs:
        Fs.append(NonDominatedSorting(np.vstack((1 / target, obj))))
        Fs.append(NonDominatedSorting(np.vstack((1 / target, 1 / obj))))
    stage = 0
    while np.sum(selected) < N:
        current_front = []
        for F in Fs:
            if len(F) > stage:
                current_front.append(F[stage])
        current_front = [np.array(c) for c in current_front]
        current_front = np.hstack(current_front)
        current_front = list(set(current_front))
        if np.sum(selected) + len(current_front) <= N:
            for i in current_front:
                selected[i] = 1
        else:
            not_selected_indices = np.arange(len(selected))[selected == 0]
            crt_front = [
                index for index in current_front if index in not_selected_indices
            ]
            num_to_select = N - np.sum(selected).astype(np.int32)
            current_front = (
                crt_front
                if len(crt_front) <= num_to_select
                else random.sample(crt_front, num_to_select)
            )
            for i in current_front:
                selected[i] = 1
        stage += 1
    return np.where(selected == 1)[0]


# ---- End code from VEGA https://github.com/huawei-noah/vega/ ----


class CarsParetoEvoController(ParetoEvoController):
    """
    A controller that samples new rollouts by mutating from points on the pareto front only.
    The first `population_size` archs are random sampled.
    """

    NAME = "cars"

    def __init__(
        self,
        search_space,
        device,
        rollout_type=None,
        mode="eval",
        population_size=100,
        perf_names=["reward", "param_size"],
        mutate_kwargs={},
        crossover_kwargs={
            "per_cell_group": False,
            "per_node": True,
            "per_connection": False,
            "sep_node_op": False,
            "prob_1": 0.5,
        },
        eval_sample_strategy="n",
        avoid_repeat=True,
        avoid_mutate_repeat=False,
        # if `avoid_repeat_worst_threshold` mutation cannot go out, raise/return
        # controlled by `avoid_repeat_fallback`
        avoid_repeat_worst_threshold=10,
        avoid_mutate_repeat_worst_threshold=10,
        avoid_repeat_fallback="return",
        avoid_repeat_from="population",
        prefill_population=True,
        schedule_cfg=None,
    ):
        super(CarsParetoEvoController, self).__init__(
            search_space,
            device=device,
            rollout_type=rollout_type,
            mode=mode,
            schedule_cfg=schedule_cfg,
        )

        expect(
            eval_sample_strategy in {"all", "n", "half_random"},
            "Invalid `eval_sample_strategy` {}, choices: {}".format(
                eval_sample_strategy, ["all", "n", "half_random"]
            ),
            ConfigException,
        )
        self.population_size = population_size
        self.perf_names = perf_names
        self.mutate_kwargs = mutate_kwargs
        self.crossover_kwargs = crossover_kwargs
        self.has_crossover = hasattr(self.search_space, "crossover")
        self.eval_sample_strategy = eval_sample_strategy
        self.avoid_repeat = avoid_repeat
        self.avoid_mutate_repeat = avoid_mutate_repeat
        self.avoid_repeat_worst_threshold = avoid_repeat_worst_threshold
        self.avoid_mutate_repeat_worst_threshold = avoid_mutate_repeat_worst_threshold
        self.avoid_repeat_fallback = avoid_repeat_fallback
        self.avoid_repeat_from = avoid_repeat_from
        # pre fill population with random sample
        self.prefill_population = prefill_population

        expect(self.avoid_repeat_fallback in {"return", "raise"})
        expect(self.avoid_repeat_from in {"population", "gt_population"})

        # after initial random sampling, only pareto front points are saved in the population
        self.population = collections.OrderedDict()
        if self.prefill_population:
            for _ in range(self.population_size):
                rollout = self.search_space.random_sample()
                # fill population with fake rewards
                self.population[rollout.genotype] = [
                    0.0 for _ in range(len(self.perf_names))
                ]

    def _sample_one(self, population_items, population_size, prob=None):
        rand = np.random.rand()
        if rand >= 0.5:
            new_rollout = self.search_space.random_sample()
        else:
            if self.has_crossover and rand > 0.25:
                parent_ind = np.random.randint(0, population_size)
                parent_ind_2 = np.random.randint(0, population_size)
                parent_rollout = self.search_space.rollout_from_genotype(
                    population_items[parent_ind][0]
                )
                parent_rollout_2 = self.search_space.rollout_from_genotype(
                    population_items[parent_ind_2][0]
                )
                # `crossover` only implement for cnn search space now
                new_rollout = self.search_space.crossover(
                    parent_rollout, parent_rollout_2, **self.crossover_kwargs
                )
            else:
                parent_ind = np.random.randint(0, population_size)
                parent_geno = population_items[parent_ind][0]
                parent_rollout = self.search_space.rollout_from_genotype(parent_geno)
                new_rollout = self._mutate(parent_rollout, **self.mutate_kwargs)
        return new_rollout

    def sample(self, n, batch_size=1):
        if self.mode == "eval":
            if self.population:
                if self.eval_sample_strategy == "all":
                    # return all archs on the pareto curve,
                    # note that number of sampled rollouts does not necessarily equals `n`
                    choices = list(self.population.items())
                elif self.eval_sample_strategy == "n":
                    # return only `n` random samples on the pareto curve
                    items = list(self.population.items())
                    ind_choices = np.random.choice(
                        np.arange(len(items)),
                        size=min(n, len(self.population)),
                        replace=False,
                    )
                    choices = [items[ind] for ind in ind_choices]
                elif self.eval_sample_strategy == "half_random":
                    rand = np.random.rand()
                    pop_n = (n // 2) if rand >= 0.5 else (n + 1) // 2
                    items = list(self.population.items())
                    ind_choices = np.random.choice(
                        np.arange(len(items)),
                        size=min(pop_n, len(self.population)),
                        replace=False,
                    )
                    choices = [items[ind] for ind in ind_choices]
                rollouts = [
                    self.search_space.rollout_from_genotype(geno).set_perfs(
                        dict(zip(self.perf_names, perfs))
                    )
                    for geno, perfs in choices
                ]
            else:
                rollouts = []
            if n > len(rollouts):
                # fill to n rollouts by random samples, note that these rollouts are not evaluated
                rollouts += [
                    self.search_space.random_sample() for _ in range(n - len(rollouts))
                ]
            return rollouts

        if len(self.population) < self.population_size:
            # population is still not full, random sample
            return [self.search_space.random_sample() for _ in range(n)]

        assert n > self.population_size, (
            "Cars need to return the population for controller/population update, "
            "set `controller_sample` to be at least larger than `population_size`"
        )

        rollouts = [
            self.search_space.rollout_from_genotype(geno) for geno in self.population
        ]
        population_items = list(self.population.items())
        prob = None
        population_size = len(population_items)

        for _ in range(n - population_size):
            if self.avoid_repeat:
                for _ in range(self.avoid_repeat_worst_threshold):
                    new_rollout = self._sample_one(
                        population_items, population_size, prob
                    )
                    pool = (
                        self.population
                        if self.avoid_repeat_from == "population"
                        else self.gt_population
                    )
                    if new_rollout.genotype not in pool:
                        break
                else:
                    self._avoid_repeat_fallback(is_mutate=False)
            else:
                new_rollout = self._sample_one(population_items, population_size, prob)
            rollouts.append(new_rollout)
        return rollouts

    def step(self, rollouts, optimizer=None, perf_name="reward"):
        """
        Note that `perf_name` argument will be ignored.
        Use `perf_names` in cfg file/`__init__` call to configure.
        """
        if len(self.perf_names) > 1:
            # only save the pareto front in the population
            # the first perf is the fitness
            fitness_list = np.array(
                [rollout.get_perf(self.perf_names[0]) for rollout in rollouts]
            )
            other_obj_lists = [
                np.array([rollout.get_perf(perf_name) for rollout in rollouts])
                for perf_name in self.perf_names[1:]
            ]
            keep = CARS_NSGA(fitness_list, other_obj_lists, self.population_size)
        else:
            # only use fitness
            fitness_list = np.array(
                [rollout.get_perf(self.perf_names[0]) for rollout in rollouts]
            )
            keep = np.argpartition(fitness_list, -self.population_size)[
                -self.population_size :
            ]

        self.population = collections.OrderedDict()
        for ind in keep:
            rollout = rollouts[ind]
            self.population[rollout.genotype] = np.array(
                [rollout.get_perf(perf_name) for perf_name in self.perf_names]
            )
        return 0

    def save(self, path):
        state = {
            "epoch": self.epoch,
            "population": {str(k): v for k, v in self.population.items()},
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device("cpu"))
        self.epoch = state["epoch"]
        self.population = {
            genotype_from_str(k, self.search_space): v
            for k, v in state["population"].items()
        }

    def __getstate__(self):
        state = super(CarsParetoEvoController, self).__getstate__()
        state["population"] = {str(k): v for k, v in state["population"].items()}
        return state

    def __setstate__(self, state):
        super(CarsParetoEvoController, self).__setstate__(state)
        self.population = {
            genotype_from_str(k, self.search_space): v
            for k, v in state["population"].items()
        }
