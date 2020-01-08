# -*- coding: utf-8 -*-

import numpy as np

from aw_nas import assert_rollout_type
from aw_nas.controller.base import BaseController

class ArchNetworkQueryController(BaseController):
    NAME = "archnetwork_query"

    def __init__(self, search_space, device, rollout_type="discrete", mode="eval",
                 query_ratio=10.):
        super(ArchNetworkQueryController, self).__init__(search_space, rollout_type, mode)

        self.query_ratio = query_ratio

    # ---- APIs ----
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def forward(self, n, arch_network_evaluator):
        return self.sample(n=n, arch_network_evaluator=arch_network_evaluator)

    def sample(self, n, arch_network_evaluator):
        n_sample = int(n * self.query_ratio)
        rollouts = [self.search_space.random.sample() for _ in range(n_sample)]
        rollouts = arch_network_evaluator.evaluate_rollouts(rollouts)
        scores = [r.perf["reward"] for r in rollouts]
        inds = np.argsort(scores)[::-1][:n]
        rollouts = [rollouts[ind] for ind in inds]
        return rollouts

    def step(self, rollouts, optimizer):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        # TODO: summary the selected rollouts' stats
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("discrete")]
