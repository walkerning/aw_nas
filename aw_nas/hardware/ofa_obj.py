
from aw_nas.hardware.base import MixinProfilingSearchSpace, BaseHardwareObjectiveModel
from aw_nas.utils.common_utils import make_divisible


class OFAHardwareObjectiveModel(BaseHardwareObjectiveModel):
    def __init__(self, schedule_cfg):
        super(OFAHardwareObjectiveModel, self).__init__(schedule_cfg)

    def predict(self, rollout):
        """
        Rollout that is not enough to predict performance, should be combined with base_channels
        """
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class OFAMixinProfilingSearchSpace(MixinProfilingSearchSpace):
    def __init__(self, seach_space, base_channels, mult_ratio, strides, primitive_type, fixed_primitives=None):
        super(OFAMixinProfilingSearchSpace, self).__init__()
        self.search_space = search_space
        self.base_channels = base_channels
        self.mult_ratio = mult_ratio
        self.channels = [c * self.mult_ratio for c in self.base_channels]
        self.strides = strides
        self.primitive_type = primitive_type

    def generate_profiling_primitives(self, sample=100):
        rollouts = [self.search_space.random_sample() for _ in range(sample)]
        primitives = []
        for rollout in rollouts:
            primitives += self._rollout_to_primitive(rollout)
        if self.fixed_primitives is not None:
            primitives += self.fixed_primitives
        primitives = list(set(primitives))
        return primitives

    def _rollout_to_primitive(self, rollout):
        primitives = []
        for i, (depth, s, c_in, c_out) in enumerate(
            zip(
                rollout.depth, 
                self.strides, 
                self.channels[:-1], 
                self.channels[1:])
            ):
            for j, width, kernel in zip(
                range(d), 
                rollout.width[i], 
                rollout.kernel[i]
                )
                if i > 0:
                    c_in = c_out
                    s = 1
                primitives.append((c_in, c_out, s, width[j], kernel[j], self.primitive_type))
        return primitives

    

