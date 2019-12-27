"""
Controller that sample compare rollout
"""

import abc

from aw_nas import Component, assert_rollout_type
from aw_nas.common import CompareRollout
from aw_nas.controller.base import BaseController


class BaseCompareBaseSelector(Component):
    """
    Choose comparator base
    """
    REGISTRY = "compare_base_selector"


class BaseCompareCompetitorSelector(Component):
    """
    Choose comparator competitor
    """
    REGISTRY = "compare_competitor_selector"


class RandomBaseSelector(BaseCompareBaseSelector):
    NAME = "random"
    def __init__(self, search_space, device):
        self.search_space = search_space

    def select(self):
        return self.search_space.random_sample()


class RandomCompetitorSelector(BaseCompareCompetitorSelector):
    NAME = "random"

    def __init__(self, search_space, device):
        self.search_space = search_space

    def select(self, base):
        while 1:
            r = self.search_space.random_sample()
            if r != base:
                return r


class CompareController(BaseController):
    NAME = "compare"

    def __init__(self, search_space, device, rollout_type="compare", mode="eval",
                 base_selector_type="random", base_selector_cfg=None,
                 competitor_selector_type="random", competitor_selector_cfg=None):
        super(CompareController, self).__init__(search_space, rollout_type, mode)

        self.device = device
        bs_cls = BaseCompareBaseSelector.get_class_(base_selector_type)
        self.base_selector = bs_cls(self.search_space, self.device, **(base_selector_cfg or {}))
        comp_cls = BaseCompareCompetitorSelector.get_class_(competitor_selector_type)
        self.competitor_selector = comp_cls(self.search_space, self.device,
                                            **(competitor_selector_cfg or {}))

    # ---- APIs ----
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def forward(self, n=1):
        return self.sample(n=n)

    def sample(self, n, batch_size=None):
        assert batch_size is None
        rollouts = []
        for _ in range(n):
            # choose base rollout
            base_r = self.base_selector.select()
            comp_r = self.competitor_selector.select(base=base_r)
            rollout = CompareRollout(rollout_1=base_r, rollout_2=comp_r)
            rollouts.append(rollout)
        return rollouts

    def step(self, rollouts, optimizer):
        # TODO:
        # Need to think about how can the arch network shared among the evaluator and controller
        # Option 1: let arch_network be a highest-level component, construct in the main function
        #   check if the evaluator or controller need the arch network component
        #   (e.g. by a class method, return )
        #   optimize in evaluator
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        # TODO: summary the selected rollouts' stats
        pass

    def save(self, path):
        self.base_selector.save("{}_base_selector".format(path))
        self.competitor_selector.save("{}_competitor_selector".format(path))

    def load(self, path):
        self.base_selector.load("{}_base_selector".format(path))
        self.competitor_selector.load("{}_competitor_selector".format(path))

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("compare")]
