# -*- coding: utf-8 -*-
"""
ofa controllers
"""

from collections import OrderedDict

import numpy as np
import torch

from aw_nas import assert_rollout_type
from aw_nas.controller.base import BaseController
from aw_nas.rollout.ofa import MNasNetOFARollout, genotype_from_str

class OFAEvoController(BaseController):
    NAME = "ofa-evo"

    def __init__(self, search_space, device, rollout_type="ofa", mode="eval",
                 population_nums=100, parent_pool_size=10, mutation_prob=1.0,
                 schedule_cfg=None):
        super(OFAEvoController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        self.mutation_prob = mutation_prob
        self.parent_pool_size = parent_pool_size
        self.cur_perf = None
        self.cur_solution = self.search_space.random_sample()
        self.population_nums = population_nums
        self.population = OrderedDict()
        self.gt_rollouts = []
        self.gt_scores = []

    def set_init_population(self, rollout_list, perf_name):
        # clear the current population
        self.population = OrderedDict()
        for r in rollout_list:
            self.population[r.genotype] = r.get_perf(perf_name)

    def mutate(self, rollout):
        arch = rollout.arch
        new_arch = {
            "depth": [1, ],
            "width": [[1], ],
            "kernel": [[3], ]
        }
        layers = sum(arch["depth"][1:], 0)
        layer_mutation_prob = self.mutation_prob / layers
        depth_mutation_prob = self.mutation_prob / len(arch["depth"])
        for i, depth in enumerate(arch["depth"][1:], 1):
            width = arch["width"][i]
            kernel = arch["kernel"][i]
            new_depth = depth
            if np.random.random() < depth_mutation_prob:
                new_depth = np.random.choice(self.search_space.depth_choice)
            new_arch["depth"] += [new_depth]
            new_arch["width"] += [[]]
            new_arch["kernel"] += [[]]
            for w, k in zip(width, kernel):
                new_w = w
                new_k = k
                if np.random.random() < layer_mutation_prob:
                    new_w = np.random.choice(self.search_space.width_choice)
                if np.random.random() < layer_mutation_prob:
                    new_k = np.random.choice(self.search_space.kernel_choice)
                new_arch["width"][-1] += [new_w]
                new_arch["kernel"][-1] += [new_k]
        return MNasNetOFARollout(new_arch, "", self.search_space)

    def sample(self, n, batch_size=None):
        if self.mode == "eval" and len(self.gt_rollouts) > n:
            # Return n archs seen(self.gt_rollouts) with best rewards
            self.logger.info("Return the best {} rollouts in the population".format(n))
            best_inds = np.argpartition(self.gt_scores, -n)[-n:]
            return [self.gt_rollouts[ind] for ind in best_inds]

        if len(self.population) < self.population_nums:
            self.logger.info("Population not full, random sample {}".format(n))
            return [self.search_space.random_sample() for _ in range(n)]

        rollouts = []
        for i in range(n):
            population_items = list(self.population.items())
            choices = np.random.choice(np.arange(len(self.population)),
                                   size=self.parent_pool_size, replace=False)
            selected_pop = [population_items[i] for i in choices]
            new_genos = sorted(selected_pop, key=lambda x: x[1], reverse=True)
            ss = self.search_space
            cur_rollout = self.search_space.rollout_from_genotype(new_genos[0][0])
            new_rollout = self.mutate(cur_rollout)
            rollouts.append(new_rollout)
        return rollouts

    def step(self, rollouts, optimizer, perf_name):
        # best_rollout = rollouts[0]
        # for r in rollouts:
        #     if r.get_perf(perf_name) > best_rollout.get_perf(perf_name):
        #         best_rollout = r
        for best_rollout in rollouts:
            self.population[best_rollout.genotype] = best_rollout.get_perf(perf_name)
        if len(self.population) > self.population_nums:
            for key in list(self.population.keys())[:len(self.population) - self.population_nums]:
                self.population.pop(key)

        # Save the best rollouts
        num_rollouts_to_keep = 200
        self.gt_rollouts = self.gt_rollouts + rollouts
        self.gt_scores = self.gt_scores + [r.get_perf(perf_name) for r in rollouts]
        if len(self.gt_rollouts) >= num_rollouts_to_keep:
            best_inds = np.argpartition(
                self.gt_scores, -num_rollouts_to_keep)[-num_rollouts_to_keep:]
            self.gt_rollouts = [self.gt_rollouts[ind] for ind in best_inds]
            self.gt_scores = [self.gt_scores[ind] for ind in best_inds]

        return 0

    @classmethod
    def supported_rollout_types(cls):
        return ["ofa"]

    # ---- APIs that is not necessary ----
    def set_device(self, device):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        saved_state = {
                "epoch": self.epoch,
                "cur_perf": self.cur_perf,
                "cur_solution": self.cur_solution.__getstate__(),
                "population": {str(k):v for k, v in self.population.items()},
                "gt_rollouts": [r.__getstate__() for r in self.gt_rollouts],
                }
        torch.save(saved_state, path)

    def load(self, path):
        state = torch.load(path, "cpu")
        self.cur_solution.__setstate__(state["cur_solution"])
        self.population = {genotype_from_str(k, self.search_space): v for k, v in state["population"].items()}
        self.gt_rollouts = []
        for r in state["gt_rollouts"]:
            rollout = self.search_space.random_sample()
            rollout.__setstate__(r)
            self.gt_rollouts += [rollout]
        self.cur_perf = state["cur_perf"]
        self.gt_scores = state["gt_scores"]
        self.epoch = state["epoch"]


class OFAController(BaseController):
    NAME = "ofa"


    def __init__(self, search_space, device, rollout_type="ofa",
                 schedule_cfg=None):
        super(OFAController, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)

        self.device = device
        self.search_space = search_space

    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def forward(self, n=1): #pylint: disable=arguments-differ
        return self.sample(n=n)

    def sample(self, n=1, batch_size=1):
        rollouts = []
        for _ in range(n):
            rollouts.append(self.search_space.random_sample())
        return rollouts

    def save(self, path):
        """Save the parameters to disk."""
        pass

    def load(self, path):
        """Load the parameters from disk."""
        pass

    def step(self):
        pass

    def summary(self, *args, **kargs):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("ofa")]
