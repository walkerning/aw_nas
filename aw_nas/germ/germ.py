# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ,invalid-name
"""
Decision, search space, rollout.
And searchable block primitives.
"""

import copy
import operator
import contextlib
from collections import OrderedDict
from collections import abc as collection_abcs

import six
import yaml
import numpy as np

import torch
from torch import nn

from aw_nas.base import Component
from aw_nas.common import BaseRollout, SearchSpace
from aw_nas.utils import expect, ConfigException
from aw_nas.germ.decisions import BaseDecision, NonleafDecision


class GermSearchSpace(SearchSpace):
    NAME = "germ"

    def __init__(self, search_space_cfg_file=None):
        super().__init__()

        self.decisions = OrderedDict()
        self.nonleaf_decisions = OrderedDict()
        self._is_initialized = False

        if search_space_cfg_file is not None:
            with open(search_space_cfg_file, "r") as r_f:
                ss_cfg = yaml.load(r_f)
            self.set_cfg(ss_cfg)

    def set_cfg(self, ss_cfg):
        # leaf decisions
        for decision_id, value in ss_cfg["decisions"].items():
            if isinstance(value[1], str):
                decision = BaseDecision.get_class_(value[0]).from_string(value[1])
            elif isinstance(value[1], BaseDecision.get_class_(value[0])):
                decision = value[1]
            else:
                raise ValueError(
                    "Except str or {} type in ss_cfg['decisions'], got {} "
                    "instead.".format(value[0], type(value[1]))
                )
            self.decisions[decision_id] = decision

        # non-leaf decisions
        for decision_id, value in ss_cfg.get("nonleaf_decisions", {}).items():
            if isinstance(value[1], str):
                decision = BaseDecision.get_class_(value[0]).from_string(value[1])
            elif isinstance(value[1], BaseDecision.get_class_(value[0])):
                decision = value[1]
            else:
                raise ValueError(
                    "Except str or {} type in ss_cfg['nonleaf_decisions'], got {} "
                    "instead.".format(value[0], type(value[1]))
                )
            self.nonleaf_decisions[decision_id] = decision
        self.blocks = ss_cfg["blocks"]

        self.decision_ids = list(self.decisions.keys())
        self.num_decisions = len(self.decisions)
        self.num_nonleaf_decisions = len(self.nonleaf_decisions)
        self.num_blocks = len(self.blocks)

        self._is_initialized = True

    def get_size(self):
        assert (
            self._is_initialized
        ), "set_cfg should be called before calling other methods."
        # currently, only support discrete choice
        return np.prod(
            [decision.search_space_size for decision in self.decisions.values()]
        )

    def get_value(self, rollout, decision_or_id):
        if not isinstance(decision_or_id, NonleafDecision) and \
           not (isinstance(decision_or_id, str) and decision_or_id not in rollout.arch):
            # leaf
            decision_id = decision_or_id.decision_id \
                          if isinstance(decision_or_id, BaseDecision) else decision_or_id
            return rollout.arch[decision_id]

        # non-leaf
        if not isinstance(decision_or_id, NonleafDecision):
            # decision id
            assert decision_or_id in self.nonleaf_decisions, decision_or_id
            nonleaf_decision = self.nonleaf_decisions[decision_or_id]
        else:
            nonleaf_decision = decision_or_id

        # recursively inference the values for non-leaf decisions
        # TODO: if necessary, consider accelerate and caching for this decision inference process
        return nonleaf_decision.get_value(rollout)

    def genotype(self, arch):
        return str(arch)

    def genotype_from_str(self, string):
        return string

    def random_sample(self):
        assert (
            self._is_initialized
        ), "set_cfg should be called before calling other methods."
        # generate a random sample for each decision
        arch = OrderedDict(
            [
                (decision_id, decision.random_sample())
                for decision_id, decision in self.decisions.items()
            ]
        )
        return GermRollout(arch, search_space=self)

    def rollout_from_genotype(self, genotype):
        assert (
            self._is_initialized
        ), "set_cfg should be called before calling other methods."
        arch = eval(genotype)  # pylint: disable=eval-used
        return GermRollout(arch, self, candidate_net=None)

    def mutate(
        self, rollout, mutate_num=None, mutate_proportion=None, mutate_prob=None
    ):
        assert (
            self._is_initialized
        ), "set_cfg should be called before calling other methods."
        expect(
            sum(
                [
                    value is not None
                    for value in [mutate_num, mutate_proportion, mutate_prob]
                ]
            )
            == 1,
            "One and only one of `mutate_num, mutate_proportion, mutate_prob` should be specified.",
            ConfigException,
        )
        # filter the trivial Decisions
        _nontrivial_decision_ids = [
            d_id for d_id, dec in self.decisions.items() if dec.search_space_size != 1
        ]
        num_nontrivial_decisions = len(_nontrivial_decision_ids)
        if mutate_proportion is not None:
            # mutate_num = int(self.num_decisions * mutate_proportion)
            mutate_num = int(num_nontrivial_decisions * mutate_proportion)
        if mutate_num is not None:
            idxes = np.random.choice(
                _nontrivial_decision_ids, size=mutate_num, replace=False
            )
        elif mutate_prob is not None:
            idxes = [
                _nontrivial_decision_ids[ind]
                for ind in np.where(
                    np.random.rand(num_nontrivial_decisions) < mutate_prob
                )[0]
            ]
        new_arch = copy.deepcopy(rollout.arch)
        for idx in idxes:
            new_arch[idx] = self.decisions[idx].mutate(new_arch[idx])
        return GermRollout(new_arch, search_space=self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        with open(filename, "w") as w_f:
            w_f.write(genotypes)
        return [(genotypes, filename)]

    @classmethod
    def supported_rollout_types(cls):
        return ["germ"]

    def distance(self, arch1, arch2):
        return NotImplementedError()

    def on_epoch_start(self, epoch):
        assert (
            self._is_initialized
        ), "set_cfg should be called before calling other methods."
        # call on_epoch_start of all decisions
        [decision.on_epoch_start(epoch) for decision in self.decisions.values()]


class GermRollout(BaseRollout):
    NAME = "germ"
    supported_components = [
        ("trainer", "simple"),
        ("evaluator", "mepa"),
        ("evaluator", "discrete_shared_weights"),
        ("evaluator", "differentiable_shared_weights"),
    ]

    def __init__(self, decision_dict, search_space, candidate_net=None):
        super().__init__()
        self.arch = decision_dict
        self.masks = OrderedDict()  # for grouped convlutions
        self.search_space = search_space
        self.candidate_net = candidate_net
        self._perf = OrderedDict()
        # _genotype to construct rollout, _net_genotype is the true gene
        self._genotype = None
        self._net_genotype = None

    def __getitem__(self, decision_or_id):
        return self.search_space.get_value(self, decision_or_id)

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = str(self.arch)
        return self._genotype

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    def plot_arch(self, filename, label="", edge_labels=None):
        return self.search_space.plot_arch(
            self.genotype, filename=filename, label=label
        )

    def reset_masks(self):
        self.masks = OrderedDict()


# ---- Searchable Blocks ----
class SearchableBlock(Component, nn.Module):
    REGISTRY = "searchable_block"

    def __init__(self, ctx):
        super(SearchableBlock, self).__init__(schedule_cfg=None)
        nn.Module.__init__(self)
        self._decisions = OrderedDict()
        self.ctx = ctx

        # A simple finalize mechanism by just store the finalized rollout.
        # However, this mechanism would cause redundant module initialization and computation.
        # Thus, it is recommended to override the `finalize_rollout` method wherever needed.
        self.finalized_rollout = None

    def register_decision(self, name, value):
        decisions = self.__dict__.get("_decisions")
        if decisions is None:
            raise Exception(
                "cannot assign decision before SearchableBlock.__init__() call"
            )
        elif not isinstance(name, six.string_types):
            raise TypeError(
                "parameter name should be a string. "
                "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        decisions[name] = value

    def __setattr__(self, name, value):
        if isinstance(value, BaseDecision):
            self.register_decision(name, value)
        return super().__setattr__(name, value)

    def __getattr__(self, name):
        if "_decisions" in self.__dict__:
            if name in self.__dict__["_decisions"]:
                return self.__dict__["_decisions"][name]
        return super().__getattr__(name)

    def named_decisions(self, prefix="", recurse=True, avoid_repeat=True):
        # By default, not recursive
        _get_named_decisions = lambda mod: mod._decisions.items()
        memo = set()
        mod_list = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mod_name, mod in mod_list:
            if isinstance(mod, SearchableBlock):
                for d_id, d_obj in _get_named_decisions(mod):
                    if d_obj is None or (avoid_repeat and d_obj in memo):
                        continue
                    memo.add(d_obj)
                    d_name = mod_name + ("." if mod_name else "") + d_id
                    yield d_name, d_obj

    def _get_decision(self, decision_obj, rollout):
        if isinstance(decision_obj, BaseDecision):
            return rollout[decision_obj]
            # if decision_obj.decision_id in rollout.arch:
            #     return rollout.arch[decision_obj.decision_id]
            # else:
            #     return max(decision_obj.choices)
        return decision_obj

    def forward_rollout(self, rollout, *args, **kwargs):
        self.ctx.rollout = rollout
        return self(*args, **kwargs)

    # def finalize_rollout_outplace(self, rollout):
    #    new_mod = copy.deepcopy(self)
    #    return new_mod.finalize_rollout(rollout)

    @contextlib.contextmanager
    def finalize_context(self, rollout):
        self.ctx.rollout = rollout
        yield
        self.ctx.rollout = None

    def finalize_rollout(self, rollout):
        """
        In-place change into a finalized block.
        Can be overrided.
        """
        self.ctx.rollout = rollout
        mod = finalize_rollout(self, rollout)
        return mod


def finalize_rollout(final_mod, rollout):
    # call `finalize_rollout` for all the 1-level submodules
    for mod_name, mod in final_mod._modules.items():
        if isinstance(mod, SearchableBlock):
            final_sub_mod = mod.finalize_rollout(rollout)
        elif isinstance(mod, nn.Sequential):
            final_sub_mod = nn.Sequential(
                *[
                    m.finalize_rollout(rollout)
                    if isinstance(m, SearchableBlock)
                    else finalize_rollout(m, rollout)
                    for m in mod
                ]
            )
        elif isinstance(mod, nn.ModuleList):
            final_sub_mod = nn.ModuleList(
                [
                    m.finalize_rollout(rollout)
                    if isinstance(m, SearchableBlock)
                    else finalize_rollout(m, rollout)
                    for m in mod
                ]
            )
        else:
            final_sub_mod = finalize_rollout(mod, rollout)
        final_mod._modules[mod_name] = final_sub_mod
    return final_mod


# ---- End Searchable Blocks ----

# ---- Decision Container ----
class DecisionDict(SearchableBlock):
    def __init__(self, decisions=None):
        super(DecisionDict, self).__init__(None)
        if decisions is not None:
            self.update(decisions)

    def __getitem__(self, key):
        return self._decisions[key]

    def __setitem__(self, key, parameter):
        self.register_decision(key, parameter)

    def __delitem__(self, key):
        del self._decisions[key]

    def __len__(self):
        return len(self._decisions)

    def __iter__(self):
        return iter(self._decisions.keys())

    def __contains__(self, key):
        return key in self._decisions

    def clear(self):
        self._decisions.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def __getattr__(self, name):
        # other methods/attributes proxied to `self._decisions`
        return getattr(self._decisions, name)

    def update(self, decisions):
        if isinstance(decisions, collection_abcs.Mapping):
            if isinstance(decisions, (OrderedDict, DecisionDict)):
                for key, decision in decisions.items():
                    self[key] = decision
            else:
                for key, decision in sorted(decisions.items()):
                    self[key] = decision
        else:
            for j, p in enumerate(decisions):
                if not isinstance(p, collection_abcs.Iterable):
                    raise TypeError(
                        "DecisionDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                if not len(p) == 2:
                    raise ValueError(
                        "DecisionsDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                self[p[0]] = p[1]

    def forward_rollout(self, rollout, inputs):
        raise Exception("Should not be called")

    def finalize_rollout(self, rollout):
        """
        In-place change into a finalized block.
        Can be overrided.
        """
        mod = finalize_rollout(self, rollout)
        return mod


class DecisionList(SearchableBlock):
    r"""Holds decisions in a list.

    :class:`~aw_nas.germ.DecisionList` can be indexed like a regular Python list, but
    decisions it contains are properly registered.

    Arguments:
        decisions (iterable, optional): an iterable of decisions to add
    """

    def __init__(self, decisions=None):
        super(DecisionList, self).__init__(None)
        self._all_elements = OrderedDict()
        if decisions is not None:
            self += decisions

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of decisions"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__([
                self._all_elements[str(i)] for i in range(len(self._all_elements))][idx]
            )
        else:
            return self._all_elements[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, decision):
        idx = self._get_abs_string_index(idx)
        return self.register_element(idx, decision)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._all_elements))[idx]:
                del self._all_elements[str(k)]
                if str(k) in self._decisions:
                    del self._decisions[str(k)]
        else:
            to_del = self._get_abs_string_index(idx)
            del self._all_elements[to_del]
            if to_del in self._decisions:
                del self._decisions[to_del]
        # To preserve numbering,
        # self._decisions is being reconstructed with decisions after deletion
        values = [
            item[1] for item in
            sorted(self._all_elements.items(), key=lambda item: int(item[0]))
        ]
        self._decisions = {}
        self._all_elements = {}
        for i, value in enumerate(values):
            self.register_element(str(i), value)

    def __len__(self):
        return len(self._all_elements)

    def __iter__(self):
        return iter([self._all_elements[str(i)] for i in range(len(self._all_elements))])

    def __iadd__(self, decisions):
        return self.extend(decisions)

    def __dir__(self):
        keys = super(DecisionList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, decision):
        r"""Insert a given decision before a given index in the list.

        Arguments:
            index (int): index to insert.
            decision (germ.Decision): decision to insert
        """
        for i in range(len(self._all_elements), index, -1):
            self.register_element(str(i), self._all_elements[str(i - 1)])
        self._decisions.pop(str(index), None)
        self.register_element(str(index), decision)

    def append(self, decision):
        r"""Appends a given decision to the end of the list.

        Arguments:
            decision (germ.Decision): decision to append
        """
        self.register_element(str(len(self)), decision)
        return self

    def extend(self, decisions):
        r"""Appends decisions from a Python iterable to the end of the list.

        Arguments:
            decisions (iterable): iterable of decisions to append
        """
        if not isinstance(decisions, collection_abcs.Iterable):
            raise TypeError("DecisionList.extend should be called with an "
                            "iterable, but got " + type(decisions).__name__)
        offset = len(self)
        for i, decision in enumerate(decisions):
            self.register_element(str(offset + i), decision)
        return self

    def register_element(self, name, decision):
        if isinstance(decision, BaseDecision):
            self.register_decision(name, decision)
        self._all_elements[name] = decision

    def __repr__(self):
        return "{}(\n\t{}\n)".format(
            self.__class__.__name__, ",\n\t".join([str(value) for value in list(self)]))

# ---- End Decision Containers ----
