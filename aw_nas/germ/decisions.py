# -*- coding: utf-8 -*-
#pylint: disable=arguments-differ,invalid-name

import re
import abc
import itertools
from functools import reduce, partial
from io import StringIO

import yaml
import numpy as np
import scipy.special

from aw_nas.base import Component
from aw_nas.utils import abstractclassmethod


class BaseDecision(Component):
    REGISTRY = "decision"

    def __repr__(self):
        return self.to_string()

    @property
    @abc.abstractmethod # python3.3+
    def search_space_size(self):
        # only relevant for discrete variable
        pass

    @abc.abstractmethod
    def random_sample(self):
        pass

    @abc.abstractmethod
    def mutate(self, old):
        pass

    @abc.abstractmethod
    def range(self):
        pass

    @abc.abstractmethod
    def to_string(self):
        pass

    @abstractclassmethod
    def from_string(cls, string):
        pass


class Choices(BaseDecision):
    NAME = "choices"

    def __init__(self, choices, size=1, replace=True, p=None, epoch_callback=None, post_mul_fn=None, schedule_cfg=None):
        super().__init__(schedule_cfg=schedule_cfg)
        self._choices = choices
        self.size = size # choose size
        self.replace = replace
        self.p = p
        self.post_mul_fn = post_mul_fn # ensure choices are divisible

        if epoch_callback is not None:
            epoch_callback = partial(epoch_callback, self)
        self.epoch_callback = epoch_callback

    def is_leaf(self):
        return all([not isinstance(c, Choices) for c in self._choices])

    @property
    def choices(self):
        if self.is_leaf():
            return self._choices
        else:
            assert len(self._choices) == 2, "Non-leaf choices node must have only 2 subnodes."

            subchoices = []
            for c in self._choices:
                if isinstance(c, Choices):
                    choices = c.choices
                elif isinstance(c, (int, float)):
                    choices = [c]
                else:
                    raise ValueError("Only support numerical or Choices operator.")
                subchoices.append(choices)
            fn = self.post_mul_fn or (lambda x: x)
            choices = [reduce(lambda a, b: fn(a * b), x) for x in itertools.product(*subchoices)]
            return choices

    @choices.setter
    def choices(self, value):
        if not self.is_leaf():
            raise ValueError("Non-leaf choices do not support assign choices.")
        self._choices = value
        self._p = None

    @property
    def num_choices(self):
        return len(self.choices)

    @property
    def p(self):
        if self.is_leaf():
            if self._p is None:
                return [(1 / self.num_choices), ] * self.num_choices 
            assert np.isclose(sum(self._p), 1.0) and len(self._p) == self.num_choices
            return self._p
        else:
            assert len(self._choices) == 2, "Non-leaf choices node must have only 2 subnodes."
            probs = []
            for c in self._choices:
                if isinstance(c, Choices):
                    prob = c.p
                elif isinstance(c, (int, float)):
                    prob = [1.0]
                else:
                    raise ValueError("Only support numerical or Choices operator.")
                probs.append(prob)
            p = [reduce(lambda a, b: a * b, x) for x in itertools.product(*probs)]
            return p

    @p.setter
    def p(self, _p):
        if not self.is_leaf() and _p is not None:
            raise ValueError("It is not supported set non-leaf node's probability.")
        assert _p is None or np.isclose(sum(_p), 1.0) and len(_p) == self.num_choices
        self._p = _p

    def __mul__(self, choices):
        if isinstance(choices, (int, float, Choices)):
            if isinstance(choices, Choices):
                assert self.size == choices.size, "Only support __mul__ between two choices have the same size."
            return  Choices((self, choices), self.size, self.replace, None, None,
                    post_mul_fn=self.post_mul_fn, schedule_cfg=self.schedule_cfg)
        else:
            raise ValueError("Only support numerical or Choices operator.")

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
        if self.epoch_callback is not None:
            self.epoch_callback(epoch)

        is_bound = True
        if not hasattr(self, "decision_id"):
             self.logger.warning(f"choice {self} is not bound to any module.")
             is_bound = False

        for i, c in enumerate(self._choices):
            if isinstance(c, Choices):
                if is_bound and not hasattr(c, "decision_id"):
                    c.decision_id = self.decision_id + f".{i}"
                c.on_epoch_start(epoch)

    @property
    def search_space_size(self):
        # do not consider p
        # consider the choices as a set not list, this might not be the case
        # for some search space construction
        return scipy.special.comb(self.num_choices, self.size, exact=False, repetition=self.replace)

    def random_sample(self):
        chosen = np.random.choice(self.choices, size=self.size, replace=self.replace, p=self.p)
        if self.size == 1:
            return chosen[0]
        return chosen

    def mutate(self, old):
        if self.search_space_size == 1:
            # To handle trivial Decision mutations. We do not want the mutation to fail silently,
            # i.e., no error is reported, but the decision / rollout is not mutated.
            # We have two options:
            # 1. Developer should obey the convention that do not create trivial Decision,

            # can fail explicitly in the `__init__` method
            # 2. However, there might be cases when one'd like to expand/shrink the search space.
            # Then, temporary trivial Decision should be common. Thus, we need to handle this
            # case of by filter the trivial Decisions (`search_space_size==1`)
            # in the search space's mutate method.
            raise Exception(
                "Should not call `mutate` on a Choices object with only one possible choice"
            )
        if self.size > 1:
            # TODO: support (>1)-sized mutate to definitely modify?
            # currently, just call random_sample, do not check if it is not modified
            return self.random_sample()

        old_ind = self.choices.index(old)
        if self.p is not None:
            whole_p = 1 - self.p[old_ind]
            if whole_p == 0:
                raise Exception("Choice {} cannot mutate from {}".format(self, old))
            mutate_p = [self.p[(old_ind + bias) % self.num_choices] / whole_p
                        for bias in range(1, self.num_choices)]
            mutate_thresh = np.cumsum(mutate_p)
            bias = np.where(np.random.rand() < mutate_thresh)[0][0] + 1
        else:
            bias = np.random.randint(1, self.num_choices)
        new_ind = (old_ind + bias) % self.num_choices
        return self.choices[new_ind]

    def range(self):
        return (min(self.choices), max(self.choices))

    def to_string(self):
        return "Choices(choices: {}, size: {}, replace: {}, p: {})".format(
            self.choices, self.size, self.replace, self.p if self.p is not None else "null")

    @classmethod
    def from_string(cls, string):
        sub_string = re.search(r"Choices\((.+)\)", string).group(1)
        sub_stringio = StringIO("{" + sub_string + "}")
        kwargs = yaml.load(sub_stringio)
        return cls(**kwargs)

# ---- Non-leaf Decisions ----
class NonLeafDecision(object):
    pass
