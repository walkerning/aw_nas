"""
Definitions of mutation-based rollout and population.
"""

import os
import copy
import glob
import shutil
import collections

import six
import yaml
import numpy as np

from aw_nas import utils
from aw_nas.utils import expect
from aw_nas.base import Component
from aw_nas.rollout.base import BaseRollout
from aw_nas.common import get_genotype_substr, genotype_from_str, ConfigTemplate
from aw_nas.utils import getLogger

class ModelRecord(object):
    def __init__(self, genotype, config, search_space,
                 info_path=None, checkpoint_path=None, finished=False,
                 confidence=None, perfs=None):
        self._genotype = genotype
        self.search_space = search_space
        # the config which is used to train this model
        self.config = config
        self.info_path = info_path
        self.checkpoint_path = checkpoint_path
        self.finished = finished
        self.confidence = confidence
        self.perfs = perfs

    def __repr__(self):
        return ("ModelRecord({_genotype}, info_path={info_path}, ckpt_path={ckpt_path}, "
                "finished={finished}, perfs={perfs}").format(
                    _genotype=self._genotype,
                    info_path=self.info_path, ckpt_path=self.checkpoint_path,
                    finished=self.finished, perfs=self.perfs)

    @property
    def genotype(self):
        return genotype_from_str(self._genotype, self.search_space)

    def save(self, path):
        meta_info = collections.OrderedDict()
        meta_info["genotypes"] = get_genotype_substr(str(self.genotype))
        meta_info["config"] = dict(self.config)
        meta_info["checkpoint_path"] = self.checkpoint_path
        meta_info["finished"] = self.finished
        meta_info["confidence"] = self.confidence
        meta_info["perfs"] = self.perfs
        self.info_path = path
        with open(path, "w") as o_stream:
            yaml.safe_dump(meta_info, stream=o_stream, default_flow_style=False)

    def save_config(self, fname):
        with open(fname, "w") as c_f:
            yaml.safe_dump(dict(self.config), c_f)

    @classmethod
    def init_from_file(cls, path, search_space):
        with open(path, "r") as meta_f:
            meta_info = yaml.safe_load(meta_f)
        record = cls(
            str(genotype_from_str(meta_info["genotypes"], search_space)),
            meta_info["config"], search_space,
            os.path.abspath(path),
            meta_info["checkpoint_path"], finished=meta_info["finished"],
            confidence=meta_info.get("confidence", None), perfs=meta_info["perfs"])
        return record

class Population(Component):
    """
    Model population.
    """

    def __init__(self, search_space, model_records, cfg_template, next_index=None):
        super(Population, self).__init__(schedule_cfg=None)
        self.search_space = search_space
        self._model_records = model_records
        self.genotype_records = collections.OrderedDict([
            (ind, genotype_from_str(
                record.genotype, self.search_space))
            for ind, record in six.iteritems(self._model_records)])
        self._size = len(model_records) # _size will be adjusted along with self._model_records
        self.cfg_template = cfg_template
        if next_index is None:
            self._next_index = np.max(list(model_records.keys())) + 1 if model_records else 0
        else:
            self._next_index = next_index
        self.start_save_index = self._next_index

    def __getstate__(self):
        state = super(Population, self).__getstate__().copy()
        del state["genotype_records"]
        return state

    def __setstate__(self, state):
        super(Population, self).__setstate__(state)
        self.genotype_records = collections.OrderedDict([
            (ind, genotype_from_str(
                record.genotype, self.search_space))
            for ind, record in six.iteritems(self._model_records)])

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
        self.genotype_records[index] = genotype_from_str(model_record.genotype, self.search_space)
        self._next_index += 1
        self._size += 1
        return index

    def save(self, path, start_index=None):
        """
        Save this population to path.
        """
        path = utils.makedir(path) # create dir if not exists
        backuped = 0
        saved = 0
        start_save_index = self.start_save_index if start_index is None else start_index
        for ind, record in six.iteritems(self.model_records):
            if ind < start_save_index:
                continue
            # save this model record
            save_path = os.path.join(path, "{}.yaml".format(ind))
            if os.path.exists(save_path):
                backup_dir = utils.makedir(os.path.join(path, "overwrite_backup"))
                backup_path = os.path.join(backup_dir, "{}.yaml".format(ind))
                self.logger.warning("%s already exists; overwrite and backup to %s",
                                    save_path, backup_path)
                shutil.copyfile(save_path, backup_path)
                backuped += 1
            record.save(save_path)
            saved += 1
        self.logger.info("Saving start from index %d. %d/%d records saved "
                         "(%d records overwrited and backuped). By default "
                         "next save will start from index %d.",
                         self.start_save_index, saved, len(self.model_records),
                         backuped, self._next_index)
        self.start_save_index = self._next_index
        return saved

    def contain_rollout(self, rollout):
        return rollout.genotype in self.genotype_records.values()

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
    def init_from_dirs(cls, dirs, search_space=None, cfg_template_file=None):
        """
        Init population from directories.

        Args:
          dirs: [directory paths]
          search_space: SearchSpace
          cfg_template_file: if not specified, default: "template.yaml" under `dirs[0]`
        Returns: Population

        There should be multiple meta-info (yaml) files named as "`<number>.yaml` under each
        directory, each of them specificy the meta information for a model in the population,
        with `<number>` represent its index.
        Note there should not be duplicate index, if there are duplicate index,
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
        with open(cfg_template_file, "r") as cfg_f:
            cfg_template = ConfigTemplate(yaml.safe_load(cfg_f))
        getLogger("population").info("Read the template config from %s", cfg_template_file)
        model_records = collections.OrderedDict()
        if search_space is None:
            # assume can parse search space from config template
            from aw_nas.common import get_search_space
            search_space = get_search_space(cfg_template["search_space_type"],
                                            **cfg_template["search_space_cfg"])
        for _, dir_ in enumerate(dirs):
            meta_files = glob.glob(os.path.join(dir_, "*.yaml"))
            for fname in meta_files:
                if "template.yaml" in fname:
                    # do not parse template.yaml
                    continue
                index = int(os.path.basename(fname).rsplit(".", 1)[0])
                expect(index not in model_records,
                       "There are duplicate index: {}. rename or soft-link the files".format(index))
                model_records[index] = ModelRecord.init_from_file(fname, search_space)
        getLogger("population").info("Parsed %d directories, total %d model records loaded.",
                                            len(dirs), len(model_records))
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

        if self.primitive:
            self.primitive_str = search_space.cell_shared_primitives[self.cell][self.primitive] \
                                 if search_space.cellwise_primitives \
                                    else search_space.shared_primitives[self.primitive]

    def apply(self, arch):
        """
        Apply this mutation on the `arch` in place.
        """
        arch[self.cell][self.mutation_type]\
            [self.search_space.num_node_inputs * self.step + self.connection] = self.modified
        return arch

    def __repr__(self):
        return "Mutation({}, {}, {}, {}, {}{})".format(
            self.cell,
            self.step,
            self.connection,
            "primitive" if self.mutation_type == CellMutation.PRIMITIVE else "node",
            self.modified,
            ", {}".format(self.primitive_str) \
            if self.mutation_type == CellMutation.PRIMITIVE else ""
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

        self.arch = self.apply_mutation(
            self.search_space,
            self.search_space.rollout_from_genotype(
                self.population.get_model(parent_index).genotype).arch,
            self.mutations
        )

        self._genotype = None
        self.model_record = ModelRecord(
            str(self.genotype),
            self.population.cfg_template.create_cfg(self.genotype),
            search_space,
            perfs=self.perf)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_genotype" in state:
            del state["_genotype"]
        return state

    @classmethod
    def apply_mutation(cls, search_space, arch, mutations):
        arch = copy.deepcopy(arch)
        for mutation in mutations:
            mutation.apply(arch)
        return arch

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def set_ckpt_path(self, path):
        assert self.model_record is not None
        self.model_record.checkpoint_path = path

    def set_perf(self, value, name="reward"):
        """
        Override: write perfs to `self.model_record` too.
        """
        assert self.model_record
        self.perf[name] = value
        if not self.model_record.perfs is self.perf:
            self.model_record.perfs[name] = value
        return self

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
        for _ in range(num_mutations):
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
                    num_prims = search_space._num_primitives \
                                if not search_space.cellwise_primitives \
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
