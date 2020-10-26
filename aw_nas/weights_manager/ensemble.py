from aw_nas.dataset import AVAIL_DATA_TYPES
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.utils.exception import ConfigException, expect

class EnsembleWeightsManager(BaseWeightsManager):
    NAME = "ensemble"

    def __init__(self, search_space, device, rollout_type,
                 inner_weights_manager_type=None, inner_weights_manager_cfg=None,
                 schedule_cfg=None):
        super(EnsembleWeightsManager, self).__init__(
            search_space, device, rollout_type, schedule_cfg)

        expect(inner_weights_manager_type is not None and inner_weights_manager_cfg is not None,
               ConfigException,
               "Should specificy `inner_weights_manager_type` and `inner_weights_manager_type`")

        self.inner_wm = BaseWeightsManager.get_class_(inner_weights_manager_type)(
            **inner_weights_manager_cfg)

    def set_device(self, device):
        self.inner_wm.set_device(device)

    def assemble_candidate(self, rollout):
        return EnsembleCandidateNet([self.inner_wm.assemble_candidate(r)
                                     for r in rollout.rollout_list])

    def step_current_gradients(self, optimizer):
        if hasattr(self.inner_wm, "step_current_gradients"):
            self.inner_wm.step_current_gradients(optimizer)

    def step(self, gradients, optimizer):
        # TODO: remove "inner_wm". But for now, `step` method is not called when `mepa_sample=1`
        # leave it for now
        return self.inner_wm.step(gradients, optimizer)

    def save(self, path):
        return self.inner_wm.save(path)

    def load(self, path):
        return self.inner_wm.load(path)

    @classmethod
    def supported_rollout_types(cls):
        return ["ensemble"]

    @classmethod
    def supported_data_types(cls):
        return AVAIL_DATA_TYPES


class EnsembleCandidateNet(CandidateNet):
    def __init__(self, candidate_nets, eval_no_grad=True):
        super(EnsembleCandidateNet, self).__init__(eval_no_grad=eval_no_grad)

        self.cand_nets = candidate_nets
        self.ensemble_size = len(self.cand_nets)

    def forward(self, *args, **kwargs): #pylint: disable=arguments-differ
        logits_list = []
        for cand_net in self.cand_nets:
            logits_list.append(cand_net.forward(*args, **kwargs))
        # TODO: more ensemble ways, e.g., stacking
        return sum(logits_list) / self.ensemble_size

    def _forward_with_params(self, *args, **kwargs): #pylint: disable=arguments-differ
        raise NotImplementedError()

    def get_device(self):
        return self.cand_nets[0].get_device()
