from aw_nas.objective.base import BaseObjective
from aw_nas.hardware.base import BaseHardwarePerformanceModel


class HardwareObjective(BaseObjective):
    NAME = "hardware"

    def __init__(self,
                 search_space,
                 hardware_perfmodel_type,
                 hardware_perfmodel_cfg=None,
                 perf_names=("latency", ),
                 hardware_model_paths=None,
                 hardware_obj_type=None,
                 schedule_cfg=None):
        super().__init__(search_space, schedule_cfg=schedule_cfg)

        if hardware_obj_type is not None:
            raise TypeError("`hardware_obj_type` has been removed, use "
                            "`hardware_perfmodel_type` instead")

        self.hardware_perfmodels = [
            BaseHardwarePerformanceModel.get_class_(hardware_perfmodel_type)(
                search_space,
                perf_name=perf_name,
                **hardware_perfmodel_cfg
            )
            for perf_name in perf_names
        ]
        self._perf_names = perf_names
        if hardware_model_paths:
            for path, perfmodel in zip(hardware_model_paths, self.hardware_perfmodels):
                perfmodel.load(path)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return self._perf_names

    def get_perfs(self, inputs, outputs, targets, cand_net):
        perfs = [perfmodel.predict(cand_net.rollout) for perfmodel in self.hardware_perfmodels]
        return perfs

    def get_reward(self, inputs, outputs, targets, cand_net):
        return 0.

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        return 0.
