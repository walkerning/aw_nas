# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ,invalid-name

import re
import abc
import operator
import itertools
from functools import reduce, partial, wraps
from io import StringIO

import yaml
import numpy as np
import scipy.special

from aw_nas.base import Component
from aw_nas.utils import abstractclassmethod
from aw_nas.utils.exception import InvalidUseException

## ---- helper functions ----
def _convert_to_number(num_str):
    try:
        res = int(num_str)
    except ValueError:
        try:
            res = float(num_str)
        except ValueError:
            return num_str, False
        return res, True
    return res, True

def _get_value(dec, rollout):
    if isinstance(dec, str):
        decision_id = dec
        return rollout[decision_id]
    elif isinstance(dec, BaseDecision):
        decision_id = dec.decision_id
        return rollout[decision_id]
    return dec

def apply_post_fn(func):
    @wraps(func)
    def func_with_post_apply(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        res = self.post_fn(res) if self.post_fn is not None else res
        return res
    return func_with_post_apply
## ---- END helper functions ----


class BaseDecision(Component):
    REGISTRY = "decision"

    def __init__(self, post_fn=None, schedule_cfg=None):
        super().__init__(schedule_cfg=schedule_cfg)
        self.post_fn = post_fn

    def __repr__(self):
        return self.to_string()

    @property
    @abc.abstractmethod  # python3.3+
    def search_space_size(self):
        # only relevant for discrete variable
        pass

    @abc.abstractmethod
    @apply_post_fn
    def random_sample(self):
        pass

    @abc.abstractmethod
    @apply_post_fn
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

    def apply(self, post_fn):
        """
        Add unary post-processing for this decision in place.
        The values of this decision (returned by `random_sample` or `get_value`)
        would be processed with the `post_fn` function.
        """
        if self.post_fn is not None:
            old_fn = self.post_fn
            self.post_fn = lambda value: post_fn(old_fn(value))
        else:
            self.post_fn = post_fn
        return self


class BaseChoices(BaseDecision):
    @property
    @abc.abstractmethod
    def size(self):
        """
        Return choose size
        """

    def __mul__(self, choices):
        if isinstance(choices, (int, float, BaseChoices)):
            if isinstance(choices, BaseChoices):
                assert (
                    self.size == choices.size
                ), "Only support __mul__ between two choices have the same selection size."
            return ChoiceMul(self, choices)
        else:
            raise ValueError("Only support numerical or Choices operator.")

    def __add__(self, choices):
        if isinstance(choices, (int, float, BaseChoices)):
            if isinstance(choices, Choices):
                assert (
                    self.size == choices.size
                ), "Only support __add__ between two choices have the same selection size."
            return ChoiceAdd(self, choices)
        else:
            raise ValueError("Only support numerical or Choices operator.")


class Choices(BaseChoices):
    """
    Leaf choices.
    """
    NAME = "choices"

    def __init__(
        self,
        choices,
        size=1,
        replace=True,
        p=None,
        epoch_callback=None,
        schedule_cfg=None,
    ):
        super().__init__(post_fn=None, schedule_cfg=schedule_cfg)
        self._choices = choices
        self._size = size # choose size
        self.replace = replace
        self.p = p

        if epoch_callback is not None:
            epoch_callback = partial(epoch_callback, self)
        self.epoch_callback = epoch_callback

    @property
    def size(self):
        return self._size

    @property
    def choices(self):
        return self._choices

    @choices.setter
    def choices(self, value):
        self._choices = value
        self._p = None

    @property
    def num_choices(self):
        return len(self.choices)

    @property
    def p(self):
        if self._p is None:
            return [(1 / self.num_choices),] * self.num_choices
        assert np.isclose(sum(self._p), 1.0) and len(self._p) == self.num_choices
        return self._p

    @p.setter
    def p(self, _p):
        assert _p is None or np.isclose(sum(_p), 1.0) and len(_p) == self.num_choices
        self._p = _p

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
        if self.epoch_callback is not None:
            self.epoch_callback(epoch)

        is_bound = True
        if not hasattr(self, "decision_id"):
            self.logger.warning(f"choice {self} is not bound to any module.")
            is_bound = False

    @property
    def search_space_size(self):
        # do not consider p
        # consider the choices as a set not list, this might not be the case
        # for some search space construction
        return scipy.special.comb(
            self.num_choices, self.size, exact=False, repetition=self.replace
        )

    @apply_post_fn
    def random_sample(self):
        chosen = np.random.choice(
            self.choices, size=self.size, replace=self.replace, p=self.p
        )
        if self.size == 1:
            return chosen[0]
        return chosen

    @apply_post_fn
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

        if self.post_fn is not None:
            # NOTE: post_fn should be deterministic for this to work
            actual_choices = [self.post_fn(c) for c in self.choices]
        else:
            actual_choices = self.choices
        old_ind = actual_choices.index(old)
        if self.p is not None:
            whole_p = 1 - self.p[old_ind]
            if whole_p == 0:
                raise Exception("Choice {} cannot mutate from {}".format(self, old))
            mutate_p = [
                self.p[(old_ind + bias) % self.num_choices] / whole_p
                for bias in range(1, self.num_choices)
            ]
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
            self.choices,
            self.size,
            self.replace,
            self.p if self.p is not None else "null"
        )

    @classmethod
    def from_string(cls, string):
        sub_string = re.search(r"Choices\((.+)\)", string).group(1)
        sub_stringio = StringIO("{" + sub_string + "}")
        kwargs = yaml.load(sub_stringio)
        return cls(**kwargs)


class NonleafDecision(BaseDecision):
    def random_sample(self):
        raise InvalidUseException("Cannot call `random_sample` on non-leaf decisions.")

    def mutate(self, old):
        raise InvalidUseException("Cannot call `mutate` on non-leaf decisions.")

    @property
    def search_space_size(self):
        return 1

    @abc.abstractmethod
    @apply_post_fn
    def get_value(self, rollout):
        pass


class BinaryNonleafDecision(NonleafDecision):
    def __init__(self, dec1, dec2, binary_op=None):
        super().__init__(schedule_cfg=None)

        assert binary_op is not None # the default value is just to suppress pylint

        if not (isinstance(dec1, (int, float)) or isinstance(dec2, (int, float))):
            # if neither dec1/dec2 are instant numbers, run a check
            is_decision_id = int(isinstance(dec1, str)) + int(isinstance(dec2, str))
            assert is_decision_id in {0, 2}, \
                "Either both operands are `decision_id`, or nor operands are `decision_id`"

        self.dec1 = dec1
        self.dec2 = dec2
        self.binary_op = binary_op

    def to_string(self):
        # warn if no `decision_id` is available
        if isinstance(self.dec1, BaseDecision) and not hasattr(self.dec1, "decision_id"):
            self.logger.warn(("`decision_id` not assigned for {}."
                              "The resulting string cannot be used to reconstruct the "
                              "non-leaf decision correctly").format(self.dec1))
        if isinstance(self.dec2, BaseDecision) and not hasattr(self.dec2, "decision_id"):
            self.logger.warn(("`decision_id` not assigned for {}."
                              "The resulting string cannot be used to reconstruct the "
                              "non-leaf decision correctly").format(self.dec2))
        return "{}(dec1={}, dec2={})".format(
            self.__class__.__name__,
            getattr(self.dec1, "decision_id", self.dec1),
            getattr(self.dec2, "decision_id", self.dec2)
        )

    @classmethod
    def from_string(cls, string):
        if cls == BinaryNonleafDecision:
            raise InvalidUseException(
                "Cannot construct BinaryNonleafDecision from string, construct one subclass of it")
        match = re.search(r"{}".format(cls.__name__) + r"\(dec1=(.+), dec2=(.+)\)", string)
        # only support parse id
        dec1_id, dec2_id = match.group(1), match.group(2)
        dec1_id, _ = _convert_to_number(dec1_id)
        dec2_id, _ = _convert_to_number(dec2_id)
        return cls(dec1_id, dec2_id) # pylint: disable=no-value-for-parameter

    @apply_post_fn
    def get_value(self, rollout):
        dec1_value = _get_value(self.dec1, rollout)
        dec2_value = _get_value(self.dec2, rollout)
        val = self.binary_op(dec1_value, dec2_value)
        return val


class BinaryNonleafChoices(BinaryNonleafDecision, BaseChoices):
    """
    To write a new binary nonleaf choices, one should inherit BinaryNonleafChoices.
    TODO ...

    Optionally, one can override corresponding operator special methods of
    some BaseDecision classes (e.g., BaseChoices)
    """
    @property
    def size(self):
        if isinstance(self.dec1, BaseChoices):
            return self.dec1.size
        return self.dec2.size

    @property
    def choices(self):
        # FIXME: when there are multi levels, this is not correct!!!
        # since the operands might not be independent, should use the whole graph to parse this, and cache
        subchoices = []
        for c in [self.dec1, self.dec2]:
            if isinstance(c, BaseChoices):
                choices = c.choices
            elif isinstance(c, (int, float)):
                choices = [c]
            else:
                raise ValueError("Only support numerical or Choices operator.")
            subchoices.append(choices)
        fn = self.post_fn or (lambda x: x)
        choices = list(np.unique([
            reduce(lambda a, b: fn(self.binary_op(a, b)), x)
            for x in itertools.product(*subchoices)
        ]))
        return choices

    def range(self):
        choices = self.choices
        return min(choices), max(choices)

    @property
    def num_choices(self):
        return len(self.choices)


def helper_register_nonleaf_binary_choice(
        binary_op, class_name_suffix, registry_name, operator_func_name=None):
    """
    helper_register_nonleaf_binary_choice(lambda x, y: max(x,y), "ChoicesMax", None)
    """
    def _constructor(self, dec1, dec2):
        BinaryNonleafDecision.__init__(self, dec1, dec2, binary_op)

    class_name = "Decision{}".format(class_name_suffix.capitalize())
    BinaryNonleafDecisionCls = type(class_name, (BinaryNonleafDecision,), {
        "__init__": _constructor, # constructor
    })
    class_name = "Choice{}".format(class_name_suffix.capitalize())
    BinaryNonleafChoiceCls = type(class_name, (BinaryNonleafDecisionCls, BinaryNonleafChoices), {
        "NAME": registry_name, # registry name
    })
    if operator_func_name is not None:
        def _override_operator(self, choices):
            if isinstance(choices, (int, float, BaseChoices)):
                if isinstance(choices, BaseChoices):
                    assert (
                        self.size == choices.size
                    ), "Only support __mul__ between two choices have the same selection size."
                return BinaryNonleafChoiceCls(self, choices)
            else:
                raise ValueError("Only support numerical or Choices operator.")
        setattr(BaseChoices, operator_func_name, _override_operator)
    return BinaryNonleafDecisionCls, BinaryNonleafChoiceCls

DecisionMul, ChoiceMul = helper_register_nonleaf_binary_choice(operator.mul, "mul", "choices_mul")
DecisionAdd, ChoiceAdd = helper_register_nonleaf_binary_choice(operator.add, "add", "choices_add")
DecisionMax, ChoiceMax = helper_register_nonleaf_binary_choice(max, "max", "choices_max")
DecisionMin, ChoiceMin = helper_register_nonleaf_binary_choice(min, "min", "choices_min")
DecisionMinus, ChoiceMinus = helper_register_nonleaf_binary_choice(
    operator.sub, "minus", "choices_minus", "__sub__")
DecisionDiv, ChoiceDiv = helper_register_nonleaf_binary_choice(
    operator.truediv, "div", "choices_div", "__truediv__")

