"""
Search space and some search space related utils.
"""

import re
import abc
import copy
import collections
import itertools

import numpy as np

from aw_nas import Component, utils
from aw_nas.utils.exception import expect, ConfigException


class SearchSpace(Component):
    REGISTRY = "search_space"

    def __init__(self, schedule_cfg=None):
        super(SearchSpace, self).__init__(schedule_cfg)

    @abc.abstractmethod
    def random_sample(self):
        pass

    @abc.abstractmethod
    def genotype(self, arch):
        """Convert arch (controller representation) to genotype (semantic representation)"""

    @abc.abstractmethod
    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""

    @abc.abstractmethod
    def plot_arch(self, genotypes, filename, label, **kwargs):
        pass

    @abc.abstractmethod
    def distance(self, arch1, arch2):
        pass

    @utils.abstractclassmethod
    def supported_rollout_types(cls):
        pass

    def mutate(self, rollout, **mutate_kwargs):
        """
        Mutate a rollout to a neighbor rollout in the search space.
        Called by mutation-based controllers, e.g., EvoController.
        """
        raise NotImplementedError()


class CellSearchSpace(SearchSpace):
    def __init__(self):
        super(CellSearchSpace, self).__init__()
        self.genotype_type = None
        self.genotype_type_name = None
        self.cell_group_names = None
        self._default_concats = None

    # namedtuple defined not at the module top level is unpicklable
    # remove it from the states
    def __getstate__(self):
        state = super(CellSearchSpace, self).__getstate__().copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        super(CellSearchSpace, self).__setstate__(state)
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name,
            self.cell_group_names + [n + "_concat" for n in self.cell_group_names],
            self._default_concats)

    @abc.abstractmethod
    def get_num_steps(self, cell_index):
        pass

    @classmethod
    def supported_rollout_types(cls):
        return ["discrete", "differentiable"]


class CNNSearchSpace(CellSearchSpace):
    """
    A cell-based CNN search space.

    This SearchSpace is cell-based, as used in ENAS, DARTS, FBNet and so on.
    Also support baseline architectures (see `examples/baselines/`) with linear structure,
    as long as the basic blocks are defined and registered as ops.

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
                 # concat for output
                 concat_op="concat",
                 concat_nodes=None,
                 loose_end=False,
                 shared_primitives=(
                     "none",
                     "max_pool_3x3",
                     "avg_pool_3x3",
                     "skip_connect",
                     "sep_conv_3x3",
                     "sep_conv_5x5",
                     "dil_conv_3x3",
                     "dil_conv_5x5"),
                 cell_shared_primitives=None,
                 derive_without_none_op=False):

        super(CNNSearchSpace, self).__init__()

        ## inner-cell
        # number of nodes in every cell
        self.num_steps = num_steps

        # number of input nodes for every node in the cell
        self.num_node_inputs = num_node_inputs

        # primitive single-operand operations
        self.shared_primitives = shared_primitives

        # for now, fixed `concat_op` will not be in the rollout
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

        # different layerwise primitives
        if cell_shared_primitives is not None:
            expect(len(cell_shared_primitives) == self.num_cell_groups,
                   "When `cell_shared_primitives` is specified, "
                   "it length should equal `num_cell_groups`", ConfigException)
            self.cell_shared_primitives = cell_shared_primitives
            self.logger.info("`cell_shared_primitives` specified, "
                             "will ignore `shared_primitives` configuration.")
            self.shared_primitives = None # avoid accidentally access this to cause subtle bugs
            self.cellwise_primitives = True
            self._num_primitives_list = [len(cps) for cps in self.cell_shared_primitives]
        else:
            self.cell_shared_primitives = [self.shared_primitives] * self.num_cell_groups
            self.cellwise_primitives = False
            self._num_primitives = len(self.shared_primitives)

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

        self.genotype_type_name = "CNNGenotype"
        if self.concat_nodes is None and not self.loose_end:
            # For backward compatiblity, when concat all steps as output
            # specificy default concats
            self._default_concats = [
                list(range(num_init_nodes, num_init_nodes + self.get_num_steps(i_cg)))
                for i_cg in range(self.num_cell_groups)
            ]
        else:
            # concat_nodes specified or loose end
            self._default_concats = []

        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name,
            self.cell_group_names + [n + "_concat" for n in self.cell_group_names],
            self._default_concats)

        # init nodes(meta arch: connect from the last `num_init_nodes` cell's output)
        self.num_init_nodes = num_init_nodes
        self.derive_without_none_op = derive_without_none_op

    def get_layer_num_steps(self, layer_index):
        return self.get_num_steps(self.cell_layout[layer_index])

    def get_num_steps(self, cell_index):
        return self.num_steps if isinstance(self.num_steps, int) else self.num_steps[cell_index]

    def random_sample(self):
        """
        Random sample a discrete architecture.
        """
        return Rollout(Rollout.random_sample_arch(
            self.num_cell_groups,
            self.num_steps,
            self.num_init_nodes,
            self.num_node_inputs,
            self._num_primitives if not self.cellwise_primitives else self._num_primitives_list),
                       info={}, search_space=self)

    def crossover(self, rollout_1, rollout_2, prob_1=0.5,
                  per_cell_group=True, per_node=True, per_connection=False, sep_node_op=False):
        dim_cell_group = self.num_cell_groups if per_cell_group else 1
        dim_node_op = 2 if sep_node_op else 1
        dim_conn = self.num_node_inputs if per_connection else 1
        assert isinstance(self.num_steps, int), \
            "For now, crossover only support using the same `num_steps` for all cell groups"
        use_rollout_2 = np.random.rand(
            dim_cell_group, dim_node_op, self.num_steps, dim_conn) > prob_1
        new_arch = np.where(
            use_rollout_2,
            np.array(rollout_1.arch).reshape(
                [self.num_cell_groups, 2, self.num_steps, self.num_node_inputs]),
            np.array(rollout_2.arch).reshape(
                [self.num_cell_groups, 2, self.num_steps, self.num_node_inputs])
        ).reshape([self.num_cell_groups, 2, -1])
        return Rollout(new_arch, info={}, search_space=self)

    def mutate(self, rollout, node_mutate_prob=0.5):
        new_arch = copy.deepcopy(rollout.arch)
        # randomly select a cell gorup to modify
        mutate_i_cg = np.random.randint(0, self.num_cell_groups)
        num_prims = self._num_primitives \
                    if not self.cellwise_primitives else self._num_primitives_list[mutate_i_cg]
        _num_step = self.get_num_steps(mutate_i_cg)
        if np.random.random() < node_mutate_prob:
            # mutate connection
            # if #cell init nodes is 1, no need to mutate the connection to node 1
            start = int(self.num_init_nodes == 1) * self.num_node_inputs
            node_mutate_idx = np.random.randint(start, len(new_arch[mutate_i_cg][0]))
            i_out = node_mutate_idx // self.num_node_inputs
            out_node = i_out + self.num_init_nodes
            offset = np.random.randint(1, out_node)
            new_arch[mutate_i_cg][0][node_mutate_idx] = \
                    (new_arch[mutate_i_cg][0][node_mutate_idx] + offset) % out_node
        else:
            # mutate op
            op_mutate_idx = np.random.randint(0, len(new_arch[mutate_i_cg][1]))
            offset = np.random.randint(1, num_prims)
            new_arch[mutate_i_cg][1][op_mutate_idx] = \
                    (new_arch[mutate_i_cg][1][op_mutate_idx] + offset) % num_prims
        return Rollout(new_arch, info={}, search_space=self)

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
        concat_list = []
        for i_cg, cg_arch in enumerate(arch):
            genotype = []
            used_end = set()
            nodes, ops = cg_arch
            num_steps = self.get_num_steps(i_cg)
            for i_out in range(num_steps):
                for i_in in range(self.num_node_inputs):
                    idx = i_out * self.num_node_inputs + i_in
                    from_ = int(nodes[idx])
                    used_end.add(from_)
                    genotype.append((self.cell_shared_primitives[i_cg][ops[idx]],
                                     from_, int(i_out + self.num_init_nodes)))
            genotype_list.append(genotype)
            if self.loose_end:
                concat = [i for i in range(self.num_init_nodes, num_steps + self.num_init_nodes)
                          if i not in used_end]
            else:
                concat = list(range(self.num_init_nodes, num_steps + self.num_init_nodes))
            concat_list.append(concat)
        genotype_list = utils.recur_apply(lambda x: x, genotype_list, depth=10, out_type=tuple)
        concat_list = utils.recur_apply(lambda x: x, concat_list, depth=10, out_type=tuple)
        kwargs = dict(itertools.chain(
            zip(self.cell_group_names, genotype_list),
            zip([n + "_concat" for n in self.cell_group_names], concat_list)
        ))
        return self.genotype_type(**kwargs)

    def rollout_from_genotype(self, genotype):
        genotype_list = list(genotype._asdict().values())
        assert len(genotype_list) == 2 * self.num_cell_groups
        conn_list = genotype_list[:self.num_cell_groups]

        arch = []
        for i_cg, cell_geno in enumerate(conn_list):
            nodes, ops = [], []
            num_steps = self.get_num_steps(i_cg)
            for i_out in range(num_steps):
                for i_in in range(self.num_node_inputs):
                    conn = cell_geno[i_out * self.num_node_inputs + i_in]
                    ops.append(self.cell_shared_primitives[i_cg].index(conn[0]))
                    nodes.append(conn[1])
            cg_arch = [nodes, ops]
            arch.append(cg_arch)
        return Rollout(arch, {}, self)

    def plot_cell(self, genotype_concat, filename, cell_index,
                  label="", edge_labels=None, plot_format="pdf"):
        """Plot a cell to `filename` on disk."""

        genotype, concat = genotype_concat

        from graphviz import Digraph

        if edge_labels is not None:
            expect(len(edge_labels) == len(genotype))

        graph = Digraph(
            format=plot_format,
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

        num_steps = self.get_num_steps(cell_index)
        for i in range(num_steps):
            graph.node(str(i), fillcolor="lightblue")
        node_names += [str(i) for i in range(num_steps)]

        for i, (op_type, from_, to_) in enumerate(genotype):
            if op_type == "none":
                continue
            edge_label = op_type
            if edge_labels is not None:
                edge_label = edge_label + "; " + str(edge_labels[i])

            graph.edge(node_names[from_], node_names[to_],
                       label=edge_label, fillcolor="gray")


        graph.node("c_{k}", fillcolor="palegoldenrod")
        for node in concat:
            if node < self.num_init_nodes:
                from_ = "c_{k-" + str(self.num_init_nodes - node) + "}"
            else:
                from_ = str(node - self.num_init_nodes)
            graph.edge(from_, "c_{k}", fillcolor="gray")

        graph.render(filename, view=False)

        return filename + ".{}".format(plot_format)

    def plot_arch(self, genotypes, filename, label="", edge_labels=None, plot_format="pdf"): #pylint: disable=arguments-differ
        """Plot an architecture to files on disk"""
        genotypes = list(genotypes._asdict().items())
        if edge_labels is None:
            edge_labels = [None] * self.num_cell_groups
        fnames = []
        for i_cg, (e_label, (cg_n, cg_geno)) in enumerate(zip(edge_labels,
                                                              genotypes[:self.num_cell_groups])):
            fname = self.plot_cell((cg_geno, genotypes[self.num_cell_groups + i_cg][1]),
                                   filename+"-"+cg_n,
                                   cell_index=i_cg,
                                   label=cg_n + " " + label,
                                   edge_labels=e_label,
                                   plot_format=plot_format)
            fnames.append((cg_n, fname))

        return fnames

    def distance(self, arch1, arch2):
        raise NotImplementedError()


class RNNSearchSpace(CellSearchSpace):
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
        assert isinstance(self.num_steps, int), \
            "RNN Searchspace not support multple cell group for now"

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

        # RNN do not support cell wise shared primitive for now,
        # acutally, only `num_cell_group=1` is supported now
        self.cell_shared_primitives = [self.shared_primitives] * self.num_cell_groups
        self.cellwise_primitives = False

        self.cell_group_names = ["cell"]

        self._default_concats = []
        self.genotype_type_name = "RNNGenotype"
        self.genotype_type = collections.namedtuple(self.genotype_type_name,
                                                    self.cell_group_names +\
                                                    [n + "_concat" for n in self.cell_group_names])

    def get_layer_num_steps(self, layer_index):
        pass

    def get_num_steps(self, cell_index):
        return self.num_steps if isinstance(self.num_steps, int) else self.num_steps[cell_index]

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

    def rollout_from_genotype(self, genotype):
        genotype_list = list(genotype._asdict().values())
        assert len(genotype_list) == 2 * self.num_cell_groups
        arch = []
        for i in range(self.num_cell_groups):
            cell_geno = genotype_list[2*i]
            nodes, ops = [], []
            for i_out in range(self.num_steps):
                for i_in in range(self.num_node_inputs):
                    conn = cell_geno[i_out * self.num_node_inputs + i_in]
                    ops.append(self.shared_primitives.index(conn[0]))
                    nodes.append(conn[1])
            cg_arch = [nodes, ops]
            arch.append(cg_arch)
        return Rollout(arch, {}, self)

    def plot_arch(self, genotypes, filename,
                  label="", edge_labels=None, plot_format="pdf"): #pylint: disable=arguments-differ
        """Plot an architecture to files on disk"""
        genotypes = list(genotypes._asdict().items())
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
            format=plot_format,
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
        # _steps = self.get_num_steps(cell_index)
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
        return [(genotypes[0][0], filename + ".{}".format(plot_format))]

    def distance(self, arch1, arch2):
        raise NotImplementedError()


# --- search space/rollout/genotype utils ---
def get_search_space(cls, **cfg):
    return SearchSpace.get_class_(cls)(**cfg)

def genotype_from_str(genotype_str, search_space):
    #pylint: disable=eval-used,bare-except
    genotype_str = str(genotype_str)
    if hasattr(search_space, "genotype_from_str"):
        # If search space
        return search_space.genotype_from_str(genotype_str)
    try:
        return eval("search_space.genotype_type({})".format(genotype_str))
    except:
        genotype_str = re.search(r".+?Genotype\((.+)\)", genotype_str).group(1)
        return eval("search_space.genotype_type({})".format(genotype_str))

def rollout_from_genotype_str(genotype_str, search_space):
    return search_space.rollout_from_genotype(genotype_from_str(genotype_str, search_space))

def plot_genotype(genotype, dest, cls, label="",
                  edge_labels=None, plot_format="pdf", **search_space_cfg):
    #pylint: disable=eval-used,bare-except
    ss = get_search_space(cls, **search_space_cfg)
    if isinstance(genotype, str):
        try:
            genotype = eval("ss.genotype_type({})".format(genotype))
        except:
            genotype = re.search(r".+?Genotype\((.+)\)", genotype).group(1)
            genotype = eval("ss.genotype_type({})".format(genotype))
        # genotype = list(genotype._asdict().items())
    # expect(isinstance(genotype, (list, tuple)))
    return ss.plot_arch(genotype, dest, label, edge_labels, plot_format=plot_format)

def group_and_sort_by_to_node(cell_geno):
    group_dct = collections.defaultdict(list)
    for conn in cell_geno:
        group_dct[conn[2]].append(conn)
    return sorted(group_dct.items(), key=lambda item: item[0])

def get_genotype_substr(genotypes):
    try:
        return re.search(r".+?Genotype\((.+)\)", genotypes).group(1)
    except Exception:
        return genotypes

class ConfigTemplate(dict):
    def create_cfg(self, genotype):
        cfg = copy.deepcopy(self)
        cfg["final_model_cfg"]["genotypes"] = get_genotype_substr(str(genotype))
        return dict(cfg)

# import all the rollouts here
from aw_nas.rollout import ( # pylint:disable=unused-import
    assert_rollout_type,
    BaseRollout,
    Rollout,
    DifferentiableRollout,
    MutationRollout,
    CompareRollout
)

from aw_nas.rollout.dense import (
    DenseSearchSpace,
    DenseDiscreteRollout,
    DenseMutationRollout
)
