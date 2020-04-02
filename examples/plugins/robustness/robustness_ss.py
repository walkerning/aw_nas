"""
Search space for robustness NAS
"""

import numpy as np

from collections import namedtuple
from aw_nas.common import SearchSpace
from aw_nas.rollout.base import BaseRollout
from aw_nas.utils.exception import expect, ConfigException


class DenseRobRollout(BaseRollout):
    NAME = "dense_rob"

    def __init__(self, arch, search_space, candidate_net=None):
        super(DenseRobRollout, self).__init__()

        self.arch = arch
        self.search_space = search_space
        self.candidate_net = candidate_net

        self._genotype = None # calc when need

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def plot_arch(self, filename, label="", edge_labels=None):
        raise NotImplementedError()

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

class DenseRobSearchSpace(SearchSpace):
    """
    Macro-arch: how many cell groups, how to arrange them, how cells connect with each other -> cnn
    Micro-arch: -> nb201
    """

    NAME = "dense_rob"

    def __init__(self, 
                 ## macro arch
                 num_cell_groups=2,
                 num_init_nodes=1,
                 cell_layout=None,
                 reduce_cell_groups=None,
                 ## micro arch
                 num_steps=4,
                 # concat for output
                 concat_op="concat",
                 concat_nodes=None,
                 loose_end=False,
                 primitives=("none", "skip_connect", "sep_conv_3x3")):
        super(DenseRobSearchSpace, self).__init__()

        ## micro-arch
        self.primitives = primitives
        self.num_op_choices = len(primitives)
        self.num_steps = num_steps
        self.num_init_nodes = num_init_nodes
        self._num_nodes = self.num_steps + self.num_init_nodes
        _num_input_conns = self.num_init_nodes * (self.num_init_nodes - 1) // 2
        self.idx = np.tril_indices(self._num_nodes, k=-1)
        self.idx = (self.idx[0][_num_input_conns:], self.idx[1][_num_input_conns:])
        self.num_possible_edges = np.sum(self.num_init_nodes + np.arange(self.num_steps))
        assert len(self.idx[0]) == self.num_possible_edges

        self.concat_op = concat_op
        self.concat_nodes = concat_nodes
        self.loose_end = loose_end
        if loose_end:
            expect(self.concat_nodes is None,
                   "When `loose_end` is given, `concat_nodes` will be automatically determined, "
                   "should not be explicitly specified.", ConfigException)

        ## overall structure/meta-arch: how to arrange cells
        # number of cell groups, different cell groups has different structure
        self.num_cell_groups = num_cell_groups

        # cell layout
        expect(cell_layout is not None,
               "`cell_layout` need to be explicitly specified",
               ConfigException)
        expect(np.max(cell_layout) == self.num_cell_groups - 1,
               "Max of elements of `cell_layout` should equal `num_cell_groups-1`")
        self.cell_layout = cell_layout
        self.num_layers = len(cell_layout)

        self.reduce_cell_groups = reduce_cell_groups
        expect(reduce_cell_groups is not None,
               "`reduce_cell_groups` need to be explicitly specified",
               ConfigException)

        self.cell_group_names = ["{}_{}".format(
            "reduce" if i in self.reduce_cell_groups else "normal", i)\
                                 for i in range(self.num_cell_groups)]

        # check concat node
        if self.concat_nodes is not None:
            if isinstance(self.concat_nodes[0], int):
                # same concat node specification for every cell groups
                _concat_nodes = [self.concat_nodes] * self.num_cell_groups
            for i_cg in range(self.num_cell_groups):
                _num_steps = self.get_num_steps(i_cg)
                expect(np.max(_concat_nodes[i_cg]) < num_init_nodes + _num_steps,
                       "`concat_nodes` {} should be in range(0, {}+{}) for cell group {}"\
                       .format(_concat_nodes[i_cg], num_init_nodes, _num_steps, i_cg))

        self.genotype_type_name = "DenseRobGenotype"
        self.genotype_type = namedtuple(self.genotype_type_name, self.cell_group_names)

        # init nodes(meta arch: connect from the last `num_init_nodes` cell's output)
        self.num_init_nodes = num_init_nodes

    def _random_sample_cell_arch(self):
        arch = np.zeros((self._num_nodes, self._num_nodes))
        arch[self.idx] = np.random.randint(
            low=0, high=self.num_op_choices, size=self.num_possible_edges)
        return arch

    def _random_sample_arch(self):
        return [self._random_sample_cell_arch() for _ in range(self.num_cell_groups)]

    # ---- APIs ----
    def random_sample(self):
        """Random sample an architecture rollout from search space"""
        return DenseRobRollout(self._random_sample_arch(), search_space=self)

    def genotype(self, arch):
        """Convert arch (controller representation) to genotype (semantic representation)"""
        cell_strs = []
        for cell_arch in arch:
            node_strs = ["init_node~{}".format(self.num_init_nodes)]
            for i_node in range(self.num_init_nodes, self._num_nodes):
                node_strs.append("|" + "|".join(["{}~{}".format(
                    self.primitives[int(cell_arch[i_node, i_input])], i_input)
                                                 for i_input in range(0, i_node)]) + "|")
            cell_str = "+".join(node_strs)
            cell_strs.append(cell_str)
        return self.genotype_type(**dict(zip(self.cell_group_names, cell_strs)))

    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""
        cell_strs = list(genotype._asdict().values())
        arch = []
        for cell_str in cell_strs:
            node_strs = cell_str.strip().split("+")
            _geno_num_i_nodes = int(node_strs[0].split("~")[1])
            expect(_geno_num_i_nodes == self.num_init_nodes,
                   ("Search space configuration (`num_init_nodes={}` "
                    "differs from the genotype specification {})").format(
                        self.num_init_nodes, _geno_num_i_nodes))
            all_conn_ops = [conn_str.split("~")[0] for node_str in node_strs[1:]
                            for conn_str in node_str[1:-1].split("|")]
            all_conn_op_inds = [self.primitives.index(conn_op) for conn_op in all_conn_ops]
            cell_arch = np.zeros((self._num_nodes, self._num_nodes))
            cell_arch[self.idx] = all_conn_op_inds
            arch.append(cell_arch)
        return DenseRobRollout(arch, search_space=self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        # TODO: plot arch
        pass

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    @classmethod
    def supported_rollout_types(cls):
        return ["dense_rob"]
