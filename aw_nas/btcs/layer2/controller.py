"""
2-layer controller.
"""

from aw_nas.controller.base import BaseController

class Layer2Controller(BaseController):
    NAME = "layer2"

    def __init__(self, search_space, rollout_type, mode="eval",
                 macro_controller_type="random_sample",
                 macro_controller_cfg={},
                 micro_controller_type="random_sample",
                 micro_controller_cfg={},
                 schedule_cfg=None):
        super(Layer2Controller, self).__init__(schedule_cfg)

        # let's do something

    def set_device(self, device):
        pass

    def sample(self, n, batch_size):
        pass

    def step(self, rollouts, optimizer, perf_name):
        pass

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["layer2"]


# TODO: macro controller
class MacroStagewiseDiffController(BaseController):
    NAME = "macro-stagewise-diff"

    def __init__(self, search_space, rollout_type, mode="eval",
                 # TODO: differentiable configurations
                 schedule_cfg=None):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["macro-stagewise"]


# TODO: micro controller
class MicroDenseDiffController(BaseController):
    NAME = "micro-dense-diff"

    def __init__(self, search_space, rollout_type, mode="eval",
                 # TODO: differentiable configurations
                 schedule_cfg=None):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["micro-dense"]
