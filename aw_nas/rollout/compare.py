"""
Compare rollout.
"""

import collections

from aw_nas.rollout.base import BaseRollout

class CompareRollout(BaseRollout):
    NAME = "compare"

    def __init__(self, rollout_1, rollout_2):
        super(CompareRollout, self).__init__()

        assert rollout_1.search_space == rollout_2.search_space
        self.rollout_1 = rollout_1
        self.rollout_2 = rollout_2
        self.search_space = rollout_1.search_space

        self.perf = collections.OrderedDict()

    def plot_arch(self, *args, **kwargs): #pylint: disable=arguments-differ
        return self.rollout_2.plot_arch(*args, **kwargs)

    def set_candidate_net(self, cand_net):
        raise Exception()

    def genotype_list(self):
        return list(self.genotype._asdict().items())

    @property
    def genotype(self):
        return self.rollout_2.genotype

    def __repr__(self):
        return ("CompareRollout(search_space={sn},\n\tr1={r1},\n\tr2={r2}, "
                "perf={perf})").format(
                    sn=self.search_space.NAME, r1=self.rollout_1,
                    r2=self.rollout_2, perf=self.perf)
