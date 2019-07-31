"""
Rollout base class and discrete/differentiable rollouts.
"""

import abc
import collections
import six
import numpy as np

from aw_nas import utils
from aw_nas.utils import logger as _logger
from aw_nas.utils import RegistryMeta, softmax

__all__ = ["BaseRollout", "Rollout", "DifferentiableRollout"]

@six.add_metaclass(RegistryMeta)
class BaseRollout(object):
    """
    Rollout is the interface object that is passed through all the components.
    """
    REGISTRY = "rollout"

    def __init__(self):
        self.perf = collections.OrderedDict()
        self._logger = None

    @property
    def logger(self):
        # logger should be a mixin class. but i'll leave it as it is...
        if self._logger is None:
            self._logger = _logger.getChild(self.__class__.__name__)
        return self._logger

    @abc.abstractmethod
    def set_candidate_net(self, c_net):
        pass

    @abc.abstractmethod
    def plot_arch(self, filename, label="", edge_labels=None):
        pass

    def get_perf(self, name="reward"):
        return self.perf.get(name, None)

    def set_perf(self, value, name="reward"):
        self.perf[name] = value

    def set_perfs(self, perfs):
        for n, v in perfs.items():
            self.set_perf(v, name=n)

class Rollout(BaseRollout):
    """Discrete rollout"""
    NAME = "discrete"

    def __init__(self, arch, info, search_space, candidate_net=None):
        super(Rollout, self).__init__()

        self.arch = arch
        self.info = info
        self.search_space = search_space
        self.candidate_net = candidate_net

        self._genotype = None # calc when need

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(self.genotype_list(),
                                           filename,
                                           label=label,
                                           edge_labels=edge_labels)

    def genotype_list(self):
        return list(self.genotype._asdict().items())

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

    def __repr__(self):
        _info = ",".join(self.info.keys()) if isinstance(self.info, dict) else self.info
        return ("Rollout(search_space={sn}, arch={arch}, info=<{info}>, "
                "candidate_net={cn}, perf={perf})").format(sn=self.search_space.NAME,
                                                           arch=self.arch,
                                                           info=_info,
                                                           cn=self.candidate_net,
                                                           perf=self.perf)

    @classmethod
    def random_sample_arch(cls, num_cell_groups, num_steps,
                           num_init_nodes, num_node_inputs, num_primitives):
        """
        random sample
        """
        arch = []
        for i_cg in range(num_cell_groups):
            num_prims = num_primitives if isinstance(num_primitives, int) else num_primitives[i_cg]
            nodes = []
            ops = []
            _num_step = num_steps if isinstance(num_steps, int) else num_steps[i_cg]
            for i_out in range(_num_step):
                nodes += list(np.random.randint(
                    0, high=i_out + num_init_nodes, size=num_node_inputs))
                ops += list(np.random.randint(
                    0, high=num_prims, size=num_node_inputs))
            arch.append((nodes, ops))
        return arch

class DifferentiableRollout(BaseRollout):
    """Rollout based on differentiable relaxation"""
    NAME = "differentiable"
    def __init__(self, arch, sampled, logits, search_space, candidate_net=None):
        super(DifferentiableRollout, self).__init__()

        self.arch = arch
        self.sampled = sampled # softmax-relaxed sample
        self.logits = logits
        self.search_space = search_space
        self.candidate_net = candidate_net

        self._genotype = None # calc when need
        self._discretized_arch = None # calc when need
        self._edge_probs = None # calc when need

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def plot_arch(self, filename, label="", edge_labels=None):
        if edge_labels is None:
            edge_labels = self.discretized_arch_and_prob[1]
        return self.search_space.plot_arch(self.genotype_list(),
                                           filename,
                                           label=label,
                                           edge_labels=edge_labels)

    def genotype_list(self):
        return list(self.genotype._asdict().items())

    def parse(self, weights):
        """parse and get the discertized arch"""
        archs = []
        edge_probs = []
        for i_cg, (cg_weight, cg_logits) in enumerate(zip(weights, self.logits)):
            cg_probs = softmax(cg_logits)
            start = 0
            n = self.search_space.num_init_nodes
            arch = [[], []]
            edge_prob = []
            num_steps = self.search_space.get_num_steps(i_cg)
            for _ in range(num_steps):
                end = start + n
                w = cg_weight[start:end]
                probs = cg_probs[start:end]
                edges = sorted(range(n), key=lambda node_id: -max(w[node_id])) #pylint: disable=cell-var-from-loop
                edges = edges[:self.search_space.num_node_inputs]
                arch[0] += edges # from nodes
                op_lst = [np.argmax(w[edge]) for edge in edges] # ops
                edge_prob += ["{:.3f}".format(probs[edge][op_id]) \
                              for edge, op_id in zip(edges, op_lst)]
                arch[1] += op_lst
                n += 1
                start = end
            archs.append(arch)
            edge_probs.append(edge_prob)
        return archs, edge_probs

    @property
    def discretized_arch_and_prob(self):
        if self._discretized_arch is None:
            if self.arch[0].ndimension() == 2:
                self._discretized_arch, self._edge_probs = self.parse(self.sampled)
            else:
                assert self.arch[0].ndimension() == 3
                self.logger.warning("Rollout batch size > 1, use logits instead of samples"
                                    "to parse the discretized arch.")
                # if multiple arch samples per step is used, (2nd dim of sampled/arch is
                # batch_size dim). use softmax(logits) to parse discretized arch
                self._discretized_arch, self._edge_probs = \
                                        self.parse(utils.softmax(utils.get_numpy(self.logits)))
        return self._discretized_arch, self._edge_probs

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.discretized_arch_and_prob[0])
        return self._genotype

    def __repr__(self):
        return ("DifferentiableRollout(search_space={sn}, arch={arch}, "
                "candidate_net={cn}, perf={perf})").format(sn=self.search_space.NAME,
                                                           arch=self.arch,
                                                           cn=self.candidate_net,
                                                           perf=self.perf)
