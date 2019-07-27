"""
Definitions of mutation-based rollout and population.
"""

import os
import copy
import glob
import collections

import yaml
import numpy as np

from aw_nas.utils import expect
from aw_nas.base import Component
from aw_nas.rollout.base import BaseRollout
from aw_nas.common import get_genotype_substr

class ConfigTemplate(dict):
    def create_cfg(self, genotype):
        cfg = copy.deepcopy(self)
        cfg["final_model_cfg"]["genotypes"] = get_genotype_substr(str(genotype))
        return cfg

class ModelRecord(object):
    def __init__(self, genotype, config, info_path, checkpoint_path, finished=False,
                 confidence=None, perfs=None):
        self.genotype = genotype
        # the config which is used to train this model
        self.config = config
        self.info_path = info_path
        self.checkpoint_path = checkpoint_path
        self.finished = finished
        self.confidence = confidence
        self.perfs = perfs

class Population(Component):
    """
    Model population.
    """

    def __init__(self, search_space, model_records, cfg_template, next_index=None):
        super(Component, self).__init__()
        self.search_space = search_space
        self._model_records = model_records
        self._size = len(model_records) # _size will be adjusted along with self._model_records
        self.cfg_template = cfg_template
        if next_index is None:
            self._next_index = np.max(list(model_records.keys())) + 1
        else:
            self._next_index = next_index

    @property
    def model_records(self):
        return self._model_records

    @property
    def next_index(self):
        return self._next_index

    @property
    def size(self):
        # return len(self.model_records)
        return self._size

    def get_model(self, index):
        return self.model_records[index]

    def add_model(self, model_record, index=None):
        index = self._next_index if index is None else index
        self.model_records[index] = model_record
        self._next_index += 1
        self._size += 1
        return index

    def remove_age(self, args):
        """
        Remove according to age (index).
        """
        # TODO

    def remove_perf(self, args):
        """
        Remove according to performance.
        """
        # TODO

    def __repr__(self):
        return "{}(size={}, search_space={}, next_index={})".format(
            self.__class__.__name__, self.size, self.search_space, self.next_index)

    @classmethod
    def init_from_dirs(self, dirs, search_space, cfg_template_file=None):
        """
        Init population from directories.

        Args:
          dirs: [directory paths]
          search_space: SearchSpace
          cfg_template_file: if not specified, default: "template.yaml" under `dirs[0]`
        Returns: Population

        Under each directory, there should be multiple meta-info (yaml) files named as `<number>.yaml`,
        each of them specificy the meta information for a model in the population, with `<number>`
        represent its index. Note there should not be duplicate index, if there are duplicate index,
        rename or soft-link the files.

        In each meta-info file, the possible meta informations are:
        * genotype
        * train_config
        * checkpoint_path
        * (optional) confidence
        * perfs: a dict of performance name to performance value

        "template.yaml" under the first dir will be used as the template training config for
        new candidate model
        """
        assert dirs, "No dirs specified!"
        if cfg_template_file is None:
            cfg_template_file = os.path.join(dirs[0], "template.yaml")
        with open(cfg_template_file, "r") as cf:
            cfg_template = ConfigTemplate(yaml.safe_load(cf))
        self.logger.info("Read the template config from %s", cfg_template_file)
        model_records = collections.OrderedDict()
        for i, dir_ in enumerate(dirs):
            meta_files = glob.glob(os.path.join(dir_, "*.yaml"))
            for fname in meta_files:
                index = int(os.path.basename(fname).rsplit(".", 1)[0])
                expect(index not in model_records,
                       "There are duplicate index: {}. rename or soft-link the files".format(index))
                with open(fname, "r") as mf:
                    meta_info = yaml.safe_load(mf)
                model_records[index] = ModelRecord(
                    meta_info["genotypes"], meta_info["config"], os.path.abspath(fname),
                    meta_info["checkpoint_path"], finished=True,
                    confidence=meta_info.get("confidence", None), perfs=meta_info["perfs"])
        return Population(search_space, model_records, cfg_template)

class CellMutation(object):
    NODE = 0
    PRIMITIVE = 1

    def __init__(self, search_space, mutation_type, cell, step, connection, modified=None):
        assert mutation_type in {CellMutation.PRIMITIVE, CellMutation.NODE}, "invalid mutation_type"
        self.search_space = search_space
        self.mutation_type = mutation_type
        self.cell = cell
        self.step = step
        self.connection = connection
        self.modified = modified
        self.node = modified if self.mutation_type == CellMutation.NODE else None
        self.primitive = modified if self.mutation_type == CellMutation.PRIMITIVE else None

    def apply(self, arch):
        """
        Apply this mutation on the `arch` in place.
        """
        arch[self.cell][self.mutation_type][self.search_space.num_node_inputs * self.step + self.connection] = self.modified
        return arch

    def __repr__(self):
        return "Mutation({}, {}, {}, {}, {})".format(
            self.cell,
            self.step,
            self.connection,
            "primitive" if self.mutation_type == CellMutation.PRIMITIVE else "node",
            self.modified
        )

class MutationRollout(BaseRollout):
    """Mutation rollout"""
    NAME = "mutation"

    def __init__(self, population, parent_index, mutations, search_space, candidate_net=None):
        super(MutationRollout, self).__init__()

        self.population = population
        self.parent_index = parent_index
        self.mutations = mutations
        self.search_space = search_space
        self.candidate_net = candidate_net
        self.model_record = None
        # TODO: add help method to dump perfs to model record?

        self.arch = self.apply_mutation(
            self.search_space,
            self.search_space.rollout_from_genotype(
                self.population.get_model(parent_index).genotype).arch,
            self.mutations
        )

        self._genotype = None

    @classmethod
    def apply_mutation(cls, search_space, arch, mutations):
        arch = copy.deepcopy(arch)
        for mutation in mutations:
            mutation.apply(arch)
        return arch

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def genotype_list(self):
        return list(self.genotype._asdict().items())

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(self.genotype_list(),
                                           filename,
                                           label=label,
                                           edge_labels=edge_labels)
    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype(self.arch)
        return self._genotype

    @classmethod
    def random_sample(cls, population, parent_index, num_mutations=1, primitive_prob=0.5):
        """
        Random sample a MutationRollout with mutations.

        Duplication is checked for multiple mutations.
        """
        search_space = population.search_space
        base_arch = search_space.rollout_from_genotype(
            population.get_model(parent_index).genotype).arch
        
        mutations = []
        primitive_choices = collections.defaultdict(list)
        primitive_mutated = collections.defaultdict(int)
        node_choices = collections.defaultdict(list)
        node_mutated = collections.defaultdict(int)
        for i_mu in range(num_mutations):
            mutation_type = CellMutation.PRIMITIVE if np.random.rand() < primitive_prob \
                else CellMutation.NODE
            cell = np.random.randint(low=0, high=search_space.num_cell_groups)
            step = np.random.randint(low=0, high=search_space.num_steps)
            connection = np.random.randint(low=0, high=search_space.num_node_inputs)
            if mutation_type == CellMutation.PRIMITIVE:
                # modify primitive on the connection
                if (cell, step, connection) in primitive_choices:
                    choices = primitive_choices[(cell, step, connection)]
                else:
                    ori = base_arch[cell][1][search_space.num_node_inputs * step + connection]
                    num_prims = search_space._num_primitives if not search_space.cellwise_primitives \
                        else search_space._num_primitives_list[cell]
                    choices = list(range(num_prims))
                    choices.remove(ori)
                    primitive_choices[(cell, step, connection)] = choices
                expect(choices,
                       ("There are no non-duplicate primitive mutation available"
                        " anymore for ({}, {}, {}) after {} mutations").format(
                            cell, step, connection, primitive_mutated[(cell, step, connection)]))
                new_choice = np.random.choice(choices)
                choices.remove(new_choice)
                base_arch[cell][1][search_space.num_node_inputs * step + connection] = new_choice
                primitive_mutated[(cell, step, connection)] += 1
            else:
                # modify input node
                if (cell, step, connection) in node_choices:
                    choices = node_choices[(cell, step, connection)]
                else:
                    ori = base_arch[cell][0][search_space.num_node_inputs * step + connection]
                    choices = list(range(search_space.num_init_nodes + step))
                    choices.remove(ori)
                    node_choices[(cell, step, connection)] = choices
                expect(choices,
                       ("There are no non-duplicate input node mutation available"
                        " anymore for ({}, {}, {}) after {} mutations").format(
                            cell, step, connection, node_mutated[(cell, step, connection)]))
                new_choice = np.random.choice(choices)
                choices.remove(new_choice)
                base_arch[cell][0][search_space.num_node_inputs * step + connection] = new_choice
                node_mutated[(cell, step, connection)] += 1
            mutations.append(CellMutation(search_space, mutation_type, cell, step, connection,
                                          modified=new_choice))
        return cls(population, parent_index, mutations, search_space)

    def __repr__(self):
        return ("MutationRollout(search_space={sn}, arch={arch}, population={pop}, "
                "parent_index={pi}, mutation={mutations}, candidate_net={cn}, perf={perf})".format(
                    sn=self.search_space.NAME,
                    arch=self.arch,
                    pop=self.population,
                    pi=self.parent_index,
                    mutations=self.mutations,
                    cn=self.candidate_net,
                    perf=self.perf
                ))
