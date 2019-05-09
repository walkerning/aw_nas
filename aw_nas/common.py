"""
Definitions that will be used by multiple components
"""

import abc
import collections
import six

import numpy as np

from aw_nas.utils import RegistryMeta


class Rollout(object):
    """
    Rollout is the interface object that is passed through all the components.
    """

    def __init__(self, arch, info, search_space, candidate_net=None):
        self.arch = arch
        self.info = info
        self.search_space = search_space
        self.candidate_net = candidate_net

        self.perf = {}

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def get_perf(self, name="perf"):
        return self.perf[name]

    def set_perf(self, value, name="perf"):
        self.perf[name] = value

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(self.arch,
                                           filename,
                                           label=label,
                                           edge_labels=edge_labels)

    def genotype(self):
        return self.search_space.genotype(self.arch)

    def __repr__(self):
        _info = ",".join(self.info.keys()) if isinstance(self.info, dict) else self.info
        return ("Rollout(search_space={sn}, info=<{info}>, "
                "candidate_net={cn}, perf={perf})").format(sn=self.search_space.NAME,
                                                           info=_info,
                                                           cn=self.candidate_net,
                                                           perf=self.perf)


class SearchSpaceException(Exception):
    pass


@six.add_metaclass(RegistryMeta)
class SearchSpace(object):
    REGISTRY = "search_space"

    @abc.abstractmethod
    def random_sample(self):
        pass

    @abc.abstractmethod
    def genotype(self, arch):
        pass

    @abc.abstractmethod
    def plot_arch(self, arch, filename, label, **kwargs):
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
        if cell_layout is not None:
            assert len(cell_layout) == self.num_layers
            assert np.max(cell_layout) == self.num_cell_groups - 1
            self.cell_layout = cell_layout
        elif self.num_cell_groups == 2:
            # by default: cell 0: normal cel, cell 1: reduce cell
            self.cell_layout = [0] * self.num_layers
            self.cell_layout[self.num_layers//3] = 1
            self.cell_layout[self.num_layers//3 * 2] = 1
        else:
            raise SearchSpaceException("`cell_layout` need to be explicitly "
                                       "specified when `num_cell_groups` != 2.")

        self.reduce_cell_groups = reduce_cell_groups
        if self.reduce_cell_groups is None:
            if self.num_cell_groups != 2:
                raise SearchSpaceException("`reduce_cell_groups` need to be explicitly "
                                           "specified when `num_cell_groups` !=  2.")
        else:
            # by default, 2 cell groups, the cell group 1 is the reduce cell group
            self.reduce_cell_groups = (1,)

        self.cell_group_names = ["{}_{}".format(
            "reduce" if i in self.reduce_cell_groups else "normal", i)\
                                 for i in range(self.num_cell_groups)]

        self.genotype_type = collections.namedtuple("CNNGenotype",
                                                    self.cell_group_names)

        # init nodes(meta arch: connect from the last `num_init_nodes` cell's output)
        self.num_init_nodes = num_init_nodes

    def random_sample(self):
        arch = []
        for _ in range(self.num_cell_groups):
            nodes = []
            ops = []
            for i_out in range(self.num_steps):
                nodes += list(np.random.randint(
                    0, high=i_out + self.num_init_nodes, size=2))
                ops += list(np.random.randint(
                    0, high=self._num_primitives, size=2))
            arch.append((nodes, ops))
        return arch

    def genotype(self, arch):
        """
        Get a human readable description of an architecture.
        """

        assert len(arch) == self.num_cell_groups

        genotype_list = []
        for cg_arch in arch:
            genotype = []
            nodes, ops = cg_arch
            for i_out in range(self.num_steps):
                for i_in in range(self.num_node_inputs):
                    idx = i_out * self.num_node_inputs + i_in
                    genotype.append((self.shared_primitives[ops[idx]],
                                     nodes[idx], i_out + self.num_init_nodes))
            genotype_list.append(genotype)
        return self.genotype_type(**dict(zip(self.cell_group_names,
                                             genotype_list)))

    def plot_cell(self, genotype, filename,
                  label="", edge_labels=None):
        """Plot a cell to `filename` on disk."""

        from graphviz import Digraph

        if edge_labels is not None:
            assert len(edge_labels) == len(genotype)

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
                edge_label = edge_label + "; " + edge_labels[i]

            graph.edge(node_names[from_], node_names[to_],
                       label=edge_label, fillcolor="gray")


        graph.node("c_{k}", fillcolor="palegoldenrod")
        for i in range(self.num_steps):
            graph.edge(str(i), "c_{k}", fillcolor="gray")

        graph.render(filename, view=False)

    def plot_arch(self, arch, filename, label="", edge_labels=None): #pylint: disable=arguments-differ
        """Plot an architecture to files on disk"""

        if edge_labels is None:
            edge_labels = [None] * self.num_cell_groups
        genotypes = self.genotype(arch)._asdict().items()
        for e_label, (cg_n, cg_geno) in zip(edge_labels, genotypes):
            self.plot_cell(cg_geno, filename+"-"+cg_n,
                           label=cg_n + " " + label,
                           edge_labels=e_label)

    def distance(self, arch1, arch2):
        raise NotImplementedError()


class RNNSearchSpace(object):
    # @TODO: rnn search space
    pass


def get_search_space(cls, **cfg):
    if cls == "cnn":
        return CNNSearchSpace(**cfg)
    if cls == "rnn":
        return RNNSearchSpace(**cfg)
    return None


#pylint: disable=invalid-name
if __name__ == "__main__":
    ss = get_search_space(cls="cnn")
    arch = ss.random_sample()
    rollout = Rollout(arch, info={}, search_space=ss)
    mock_edge_label = np.random.rand(ss.num_cell_groups,
                                     ss.num_steps*ss.num_node_inputs)
    mock_edge_label = np.vectorize("{:.3f}".format)(mock_edge_label)
    print("architecture: ", arch)
    rollout.plot_arch("./try_plot", label="try plot",
                      edge_labels=mock_edge_label.tolist())
#pylint: enable=invalid-name
