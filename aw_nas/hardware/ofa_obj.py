
import numpy as np
import pickle
from collections import namedtuple

from aw_nas.hardware.base import MixinProfilingSearchSpace, BaseHardwareObjectiveModel
from aw_nas.utils.common_utils import make_divisible

OFAPrim = namedtuple('OFAPrim', ['prim_type', 'input_ch', 'output_ch', 'stride', 'spatial_size', 'kernel'])

def ofa_rollout_to_primitive(rollout, strides, channels, primitive_type):
    primitives = []
    for i, (depth, s, c_in, c_out) in enumerate(
        zip(
            rollout.depth, 
            strides, 
            channels[:-1], 
            channels[1:])
        ):
        for j, width, kernel in zip(
            range(d), 
            rollout.width[i], 
            rollout.kernel[i]
            )
            if i > 0:
                c_in = c_out
                s = 1
            primitives.append(OFAPrim(primitive_type, c_in, c_out, s, width[j], kernel[j]))
    return primitives

class OFAHardwareObjectiveModel(BaseHardwareObjectiveModel):
    def __init__(self, prof_prims, prof_prims_cfg, schedule_cfg):
        super(OFAHardwareObjectiveModel, self).__init__(schedule_cfg)
        assert 'base_channels' in prof_prims_cfg
        assert 'mult_ratio' in prof_prims_cfg
        assert 'strides' in prof_prims_cfg
        self.default_prim_type = prof_prims_cfg.get('default_prim_type', 'default')
        self.performances = prof_prims_cfg.get('performances', ['latency'])

        self.prof_prims = prof_prims
        self.prof_prims_cfg = prof_prims_cfg
        self.channels = [c * prof_prims_cfg['mult_ratio'] for c in prof_prims['base_channels']]
        self.strides = prof_prims_cfg['strides']

        self.Perf = namedtuple('Performances', self.performances)

        self._orgnize_table()

        # TODO: @z_c
        self._perf_model = None

    def _orgnize_table(self):
        self._table = {self.default_prim_type: {}}
        for prim in self.prof_prims:
            prim_type = prim['prim_type']
            if prim_type not in self._table:
                self._table[prim_type] = {}
            sub_table = self._table[prim_type]

            perf = self.Perf(*[prim[f] for f in self.performances])
            if prim_type == self.default_prim_type:
                prim = OFAPrim(*[prim[f] for f in OFAPrim._fields])
            else:
                prim = tuple([prim[k] for k in prim if k not in self.performances])
            sub_table[prim] = perf


    def predict(self, rollout):
        primtives = ofa_rollout_to_primitive(rollout, self.strides, self.channels, self.default_prim_type)
        perfs = []
        for prim in primtives:
            perf = self._table[prim.prim_type].get(prim)
            if perf is None:
                perf = self._perf_model(prim) # TODO: latency model
            perfs.append(perf)
        # assert that each of performances is float
        perfs = np.array(perfs)
        return perfs.sum(axis=0)

    def save(self, path):
        pickle.dump({'table': self._table, 'model': self._perf_model.dumps()}, path)

    def load(self, path):
        m = pickle.load(path)
        self._table = m['table']
        self._perf_model.loads(m['model'])

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
            primitives += rollout_to_primitive(rollout, self.strides, self.channels, self.primitive_type)
        if self.fixed_primitives is not None:
            primitives += self.fixed_primitives
        primitives = list(set(primitives))
        return primitives

