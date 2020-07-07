import numpy as np

from aw_nas.objective.base import BaseObjective
from aw_nas.hardware.base import BaseHardwareObjectiveModel


class HardwareObjective(BaseObjective):
    NAME = "hardware"

    def __init__(self,
                 search_space,
                 hardware_obj_type,
                 hardware_obj_cfg={},
                 hardware_model_path='',
                 perf_names=["latency"],
                 schedule_cfg=None):
        super().__init__(search_space, schedule_cfg=schedule_cfg)

        self.hwobj_model = BaseHardwareObjectiveModel.get_class_(
            hardware_obj_type)(**hardware_obj_cfg)
        self._perf_names = perf_names
        if hardware_model_path:
            self.hwobj_model.load(hardware_model_path)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return self._perf_names

    def get_perfs(self, inputs, outputs, targets, cand_net):
        perfs = self.hwobj_model.predict(cand_net.rollout)
        return perfs

    def get_reward(self, inputs, outputs, targets, cand_net):
        return 0.

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        return 0.
