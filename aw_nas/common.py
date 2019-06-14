"""
Definitions that will be used by multiple components
"""

import abc
import collections
import itertools
import six

import numpy as np

from aw_nas import Component
from aw_nas.utils import RegistryMeta, softmax
from aw_nas.utils.exception import expect, ConfigException

def assert_rollout_type(type_name):
    expect(type_name in BaseRollout.all_classes_(),
           "rollout type {} not registered yet".format(type_name))
    return type_name

@six.add_metaclass(RegistryMeta)
class BaseRollout(object):
    """
    Rollout is the interface object that is passed through all the components.
    """
    REGISTRY = "rollout"

    def __init__(self):
        self.perf = collections.OrderedDict()

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
        for _ in range(num_cell_groups):
            nodes = []
            ops = []
            for i_out in range(num_steps):
                nodes += list(np.random.randint(
                    0, high=i_out + num_init_nodes, size=num_node_inputs))
                ops += list(np.random.randint(
                    0, high=num_primitives, size=num_node_inputs))
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
        for cg_weight, cg_logits in zip(weights, self.logits):
            cg_probs = softmax(cg_logits)
            start = 0
            n = self.search_space.num_init_nodes
            arch = [[], []]
            edge_prob = []
            for _ in range(self.search_space.num_steps):
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
            self._discretized_arch, self._edge_probs = self.parse(self.sampled)
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


class SearchSpace(Component):
    REGISTRY = "search_space"

    def __init__(self):
        super(SearchSpace, self).__init__(schedule_cfg=None)
        self.genotype_type = None
        self.genotype_type_name = None
        self.cell_group_names = None

    # namedtuple defined not at the module top level is unpicklable
    # remove it from the states
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.genotype_type = collections.namedtuple(self.genotype_type_name, self.cell_group_names)

    @abc.abstractmethod
    def random_sample(self):
        pass

    @abc.abstractmethod
    def genotype(self, arch):
        pass

    @abc.abstractmethod
    def plot_arch(self, genotypes, filename, label, **kwargs):
        pass

    @abc.abstractmethod
    def distance(self, arch1, arch2):
        pass


class CNNSearchSpace(SearchSpace):
    """
    A cell-based CNN search space.

    This SearchSpace is cell-based, as used in ....@TODO: add paper refs.

    .. note::
    If cell group number (`num_cell_groups`) is 2, which is the default,
    the cell layout will be: cell 1 at 1/3 and 2/3 position; cell 0 at other
    position. e.g. if `num_layers`=8, the cell layout will be infered to
    [0, 0, 1, 0, 1, 0, 0, 0]). This default cell layout is used in many papers,
    where cell 1 is interpreted as reduction block , and cell 0 is normal
    block.

    .. warning::
    However, when `num_cell_groups` is not 2, there are no default cell
    layout, the cell layout needs to be specified using `cell_layout` key
    in the configuration. Also

    .. note::
    The implementation of `weights_manager` are tightly coupled
    with the implicit defined architecture rules:
    * Meta-arch: (Arch rules of the upper hierarchy)
        * There are 2 type of cell groups:
          normal cell group, reduce cell group.
          Default rules: If there are only two cell groups,
          cell group 0 of type normal, cell group 1 of type reduce.
          The reduce cells is by default placed at the position
          of 1/3 and 2/3.

          During the forward pass, for every reduce cell encountered,
          the spartial dimension decay by 2,
          and channel dimension increase by 2.
        * The last `num_init_nodes` cell's output is used
          as each cell's initial input nodes.

    * Arch rules of current hierarchy:
        * The final node's merge op is `depthwise_concat`,
          all the output of inner-cell nodes will be merged.
        * Every other node's merge op is `sum`.
    """

    NAME = "cnn"

    def __init__(self,
                 # meta arch
                 num_cell_groups=2,
                 num_init_nodes=2,
                 num_layers=8, cell_layout=None,
                 reduce_cell_groups=(1,),
                 # cell arch
                 num_steps=4, num_node_inputs=2,
                 shared_primitives=(
                     "none",
                     "max_pool_3x3",
                     "avg_pool_3x3",
                     "skip_connect",
                     "sep_conv_3x3",
                     "sep_conv_5x5",
                     "dil_conv_3x3",
                     "dil_conv_5x5")):

        super(CNNSearchSpace, self).__init__()

        ## inner-cell
        # number of nodes in every cell
        self.num_steps = num_steps

        # number of input nodes for every node in the cell
        self.num_node_inputs = num_node_inputs

        # primitive single-operand operations
        self.shared_primitives = shared_primitives
        self._num_primitives = len(self.shared_primitives)

        ## overall structure/meta-arch: how to arrange cells
        # number of cell groups, different cell groups has different structure
        self.num_cell_groups = num_cell_groups

        # number of cell layers
        # if (by default) `num_layers` == 8: the cells of 2 cell groups 0(normal), 1(reduce)
        # will be arranged in order [0, 0, 1, 0, 1, 0, 0, 0]
        self.num_layers = num_layers

        # cell layout
        expect(self.num_cell_groups == 2 or cell_layout is not None,
               "`cell_layout` need to be explicitly specified when `num_cell_groups` != 2.",
               ConfigException)
        if cell_layout is not None:
            expect(len(cell_layout) == self.num_layers,
                   "Length of `cell_layout` should equal `num_layers`")
            expect(np.max(cell_layout) == self.num_cell_groups - 1,
                   "Max of elements of `cell_layout` should equal `num_cell_groups-1`")
            self.cell_layout = cell_layout
        elif self.num_cell_groups == 2:
            # by default: cell 0: normal cel, cell 1: reduce cell
            self.cell_layout = [0] * self.num_layers
            self.cell_layout[self.num_layers//3] = 1
            self.cell_layout[(2 * self.num_layers)//3] = 1

        expect(self.num_cell_groups == 2 or reduce_cell_groups is not None,
               "`reduce_cell_groups` need to be explicitly specified when `num_cell_groups` !=  2.",
               ConfigException)
        self.reduce_cell_groups = reduce_cell_groups
        if self.reduce_cell_groups is None:
            # by default, 2 cell groups, the cell group 1 is the reduce cell group
            self.reduce_cell_groups = (1,)

        self.cell_group_names = ["{}_{}".format(
            "reduce" if i in self.reduce_cell_groups else "normal", i)\
                                 for i in range(self.num_cell_groups)]

        self.genotype_type_name = "CNNGenotype"
        self.genotype_type = collections.namedtuple(self.genotype_type_name,
                                                    self.cell_group_names)

        # init nodes(meta arch: connect from the last `num_init_nodes` cell's output)
        self.num_init_nodes = num_init_nodes

    def random_sample(self):
        """
        Random sample a discrete architecture.
        """
        return Rollout(Rollout.random_sample_arch(self.num_cell_groups,
                                                  self.num_steps,
                                                  self.num_init_nodes,
                                                  self.num_node_inputs,
                                                  self._num_primitives),
                       info={}, search_space=self)

    def genotype(self, arch):
        """
        Get a human readable description of an discrete architecture.

        ..note:
            Due to the implementation of weights_manager, genotype must be ordered in the way
            that `to_node` monotonously increase, this is guaranteed due to the implicit
            assumption of the controller's sampling order.
        """

        expect(len(arch) == self.num_cell_groups)

        genotype_list = []
        for cg_arch in arch:
            genotype = []
            nodes, ops = cg_arch
            for i_out in range(self.num_steps):
                for i_in in range(self.num_node_inputs):
                    idx = i_out * self.num_node_inputs + i_in
                    genotype.append((self.shared_primitives[ops[idx]],
                                     int(nodes[idx]), int(i_out + self.num_init_nodes)))
            genotype_list.append(genotype)
        return self.genotype_type(**dict(zip(self.cell_group_names,
                                             genotype_list)))

    def plot_cell(self, genotype, filename,
                  label="", edge_labels=None):
        """Plot a cell to `filename` on disk."""

        from graphviz import Digraph

        if edge_labels is not None:
            expect(len(edge_labels) == len(genotype))

        graph = Digraph(
            format="png",
            # https://stackoverflow.com/questions/4714262/graphviz-dot-captions
            body=["label=\"{l}\"".format(l=label),
                  "labelloc=top", "labeljust=left"],
            edge_attr=dict(fontsize="20", fontname="times"),
            node_attr=dict(style="filled", shape="rect",
                           align="center", fontsize="20",
                           height="0.5", width="0.5",
                           penwidth="2", fontname="times"),
            engine="dot")
        graph.body.extend(["rankdir=LR"])

        node_names = ["c_{k-" + str(self.num_init_nodes - i_in) + "}"\
                      for i_in in range(self.num_init_nodes)]
        [graph.node(node_name, fillcolor="darkseagreen2") for node_name in node_names]

        for i in range(self.num_steps):
            graph.node(str(i), fillcolor="lightblue")
        node_names += [str(i) for i in range(self.num_steps)]

        for i, (op_type, from_, to_) in enumerate(genotype):
            edge_label = op_type
            if edge_labels is not None:
                edge_label = edge_label + "; " + str(edge_labels[i])

            graph.edge(node_names[from_], node_names[to_],
                       label=edge_label, fillcolor="gray")


        graph.node("c_{k}", fillcolor="palegoldenrod")
        for i in range(self.num_steps):
            graph.edge(str(i), "c_{k}", fillcolor="gray")

        graph.render(filename, view=False)

        return filename + ".png"

    def plot_arch(self, genotypes, filename, label="", edge_labels=None): #pylint: disable=arguments-differ
        """Plot an architecture to files on disk"""

        if edge_labels is None:
            edge_labels = [None] * self.num_cell_groups
        fnames = []
        for e_label, (cg_n, cg_geno) in zip(edge_labels, genotypes):
            fname = self.plot_cell(cg_geno, filename+"-"+cg_n,
                                   label=cg_n + " " + label,
                                   edge_labels=e_label)
            fnames.append((cg_n, fname))

        return fnames

    def distance(self, arch1, arch2):
        raise NotImplementedError()


class RNNSearchSpace(SearchSpace):
    NAME = "rnn"

    def __init__(self,
                 # meta arch
                 num_cell_groups=1,
                 num_init_nodes=1,
                 num_layers=1,
                 # cell arch
                 num_steps=8, num_node_inputs=1,
                 loose_end=False,
                 shared_primitives=(
                     "tanh",
                     "relu",
                     "sigmoid",
                     "identity")):
        super(RNNSearchSpace, self).__init__()

        ## inner-cell
        # number of nodes in every cell
        self.num_steps = num_steps

        # number of input nodes for every node in the cell
        self.num_node_inputs = num_node_inputs

        # primitive single-operand operations
        self.shared_primitives = shared_primitives
        self._num_primitives = len(self.shared_primitives)

        # whether or not to use loose end
        self.loose_end = loose_end

        ## overall structure/meta-arch: how to arrange cells
        self.num_cell_groups = num_cell_groups # this must be 1 for current rnn weights manager

        # init nodes(meta arch: connect from the last `num_init_nodes` cell's output)
        self.num_init_nodes = num_init_nodes # this must be 1 for current rnn weights manager

        self.num_layers = num_layers

        self.cell_group_names = ["cell"]

        self.genotype_type_name = "RNNGenotype"
        self.genotype_type = collections.namedtuple(self.genotype_type_name,
                                                    self.cell_group_names +\
                                                    [n + "_concat" for n in self.cell_group_names])

    def random_sample(self):
        """
        Random sample a discrete architecture.
        """
        return Rollout(Rollout.random_sample_arch(self.num_cell_groups, # =1
                                                  self.num_steps,
                                                  self.num_init_nodes, # =1
                                                  self.num_node_inputs, # =1
                                                  self._num_primitives),
                       info={}, search_space=self)

    def genotype(self, arch):
        """
        Get a human readable description of an discrete architecture.
        """

        expect(len(arch) == self.num_cell_groups) # =1

        genotype_list = []
        concat_list = []
        for cg_arch in arch:
            genotype = []
            nodes, ops = cg_arch
            used_end = set()
            for i_out in range(self.num_steps):
                for i_in in range(self.num_node_inputs):
                    idx = i_out * self.num_node_inputs + i_in
                    from_ = int(nodes[idx])
                    used_end.add(from_)
                    genotype.append((self.shared_primitives[ops[idx]],
                                     from_, int(i_out + self.num_init_nodes)))
            genotype_list.append(genotype)
            if self.loose_end:
                concat = [i for i in range(1, self.num_steps+1) if i not in used_end]
            else:
                concat = list(range(1, self.num_steps+1))
            concat_list.append(concat)
        kwargs = dict(itertools.chain(
            zip(self.cell_group_names, genotype_list),
            zip([n + "_concat" for n in self.cell_group_names], concat_list)
        ))
        return self.genotype_type(**kwargs)

    def plot_arch(self, genotypes, filename, label="", edge_labels=None): #pylint: disable=arguments-differ
        """Plot an architecture to files on disk"""
        expect(len(genotypes) == 2 and self.num_cell_groups == 1,
               "Current RNN search space only support one cell group")
        expect(self.num_init_nodes == 1, "Current RNN search space only support one init node")

        # only one cell group now!
        geno_, concat_ = genotypes
        geno_, concat_ = geno_[1], concat_[1]
        edge_labels = edge_labels[0] if edge_labels is not None else None
        filename = filename + "-" + genotypes[0][0]

        from graphviz import Digraph

        graph = Digraph(
            format="png",
            # https://stackoverflow.com/questions/4714262/graphviz-dot-captions
            body=["label=\"{l}\"".format(l=label),
                  "labelloc=top", "labeljust=left"],
            edge_attr=dict(fontsize="20", fontname="times"),
            node_attr=dict(style="filled", shape="rect",
                           align="center", fontsize="20",
                           height="0.5", width="0.5",
                           penwidth="2", fontname="times"),
            engine="dot")
        graph.body.extend(["rankdir=LR"])
        graph.node("x_{t}", fillcolor="darkseagreen2")
        graph.node("h_{t-1}", fillcolor="darkseagreen2")
        graph.node("0", fillcolor="lightblue")
        graph.edge("x_{t}", "0", fillcolor="gray")
        graph.edge("h_{t-1}", "0", fillcolor="gray")

        _steps = self.num_steps
        for i in range(1, 1 + _steps):
            graph.node(str(i), fillcolor="lightblue")

        for i, (op_type, from_, to_) in enumerate(geno_):
            edge_label = op_type
            if edge_labels is not None:
                edge_label = edge_label + "; " + str(edge_labels[i])

            graph.edge(str(from_), str(to_), label=edge_label, fillcolor="gray")

        graph.node("h_{t}", fillcolor="palegoldenrod")

        for i in concat_:
            graph.edge(str(i), "h_{t}", fillcolor="gray")

        graph.render(filename, view=False)
        return [(genotypes[0][0], filename + ".png")]

    def distance(self, arch1, arch2):
        raise NotImplementedError()


def get_search_space(cls, **cfg):
    return SearchSpace.get_class_(cls)(**cfg)

def plot_genotype(genotype, dest, cls, label="", edge_labels=None, **search_space_cfg):
    ss = get_search_space(cls, **search_space_cfg)
    if isinstance(genotype, str):
        genotype = eval("ss.genotype_type({})".format(genotype)) # pylint: disable=eval-used
        genotype = list(genotype._asdict().items())
    expect(isinstance(genotype, (list, tuple)))
    return ss.plot_arch(genotype, dest, label, edge_labels)
