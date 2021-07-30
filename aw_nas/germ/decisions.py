# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ,invalid-name

import re
import abc
import operator
import itertools
import functools
from functools import reduce, partial, wraps
from io import StringIO
from collections import defaultdict
import copy

import yaml
import numpy as np
import scipy.special

from aw_nas.base import Component
from aw_nas.utils import abstractclassmethod
from aw_nas.utils.exception import InvalidUseException

# ---- helper functions ----


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

def set_derive_flag(func):
    @wraps(func)
    def func_set_derive(self, *args, **kwargs):
        if self._derive_flag:
            return
        # check if there is _choices or _optional_choices
        if '_choices' in dir(self) and \
            self._choices is not None and \
            len(self._choices) > 0:
                self._regard_self_leaf(self._choices)
        elif '_optional_choices' in dir(self) and \
            self._optional_choices is not None and \
            len(self._optional_choices) > 0:
                self._regard_self_leaf(self._optional_choices)
        else:
            func(self, *args, **kwargs)
        self._derive_flag = True
    return func_set_derive

def _get_parent_choices(choices_list):
    """
    get all possible values in choices_list
    """
    possible_value_list = list()
    for choices in choices_list:
        if isinstance(choices, BaseDecision):
            possible_value_list.append(copy.deepcopy(choices.choices))
        elif isinstance(choices, (int, float)):
            possible_value_list.append([choices])
        else:
            raise Exception(
                "Only support numerical or Choices operator."
            )
    return possible_value_list

def generator_target_list(choices_list):
    """
    inputs: choices_list: list of choices, like [dec1, dec2]
    yield: id_list, value_list, target_list
    """
    assert len(choices_list) > 0, \
        "The choices list can not be empty"
    # get all root id
    ori_id_list = [
        set(choices._choices_graph.get_id_list()) if isinstance(choices, BaseDecision) else set({}) \
        for choices in choices_list
    ]
    root_id_list = list(functools.reduce(lambda x, y: x | y, ori_id_list))
    root_id_list = sorted(root_id_list, key = lambda x: int(x))
    assert len(root_id_list) > 0, \
        "There must be dependent roots"

    # get and check the same id have the same
    root_value_list = defaultdict(list)
    for root_id in root_id_list:
        for i, slice_id_set in enumerate(ori_id_list):
            if root_id in slice_id_set:
                # there is not
                if root_id not in root_value_list.keys():
                    root_value_list[root_id] = copy.deepcopy(
                        choices_list[i]._choices_graph.get_value_list(
                            root_id,
                        )
                    )
                # there is
                else:
                    assert len(
                        set(root_value_list[root_id]) ^ \
                        set(choices_list[i]._choices_graph.get_value_list(root_id))
                    ) == 0, \
                        "the same root in different choices should have the same values"

        assert root_id in root_value_list.keys(), \
            "there should be value for root id"

    # cartesian product
    total_length = functools.reduce(
        lambda x, y: x*y,
        [len(root_value_list[root_id]) for root_id in root_id_list]
    )
    cartesian_space = [
        list(range(len(root_value_list[root_id])))
        for root_id in root_id_list
    ]
    cartesian_product = itertools.product(*cartesian_space)
    for sample in cartesian_product:
        sample_value = [root_value_list[root_id_list[k]][index] for k, index in enumerate(sample)]
        # traverse all choices
        target_list = []
        for choices in choices_list:
            if isinstance(choices, BaseDecision):
                target_list.append(
                    choices._choices_graph.get_target(
                        root_id_list,
                        sample_value,
                    )
                )
            elif isinstance(choices, (int, float)):
                target_list.append(choices)
            else:
                raise Exception(
                    "Only support numerical or Choices operator."
                )
        yield total_length, root_id_list, sample_value, target_list


class RootRecord(object):
    """
    record the target value corresponging different root value
    ID(A)_VALUE(B) is the pattern of the key
    A is the id of the root, B denotes the possible value of A
    """
    def __init__(self):
        self.__map = defaultdict()
        self.__id_list = list()
        self.__id_value_dict = defaultdict(list)

    def _clear_map(self):
        self.__init__()

    def set_target(self, id_list, value_list, target):
        self._update(id_list, value_list)
        gen_key = self._gen_key(id_list, value_list)
        self.__map[gen_key] = target

    def get_target(self, id_list, value_list):
        gen_key = self._gen_key(id_list, value_list)
        return self.__map.get(gen_key, None)

    def get_id_list(self):
        return self.__id_list

    def get_value_list(self, this_id):
        return self.__id_value_dict.get(this_id, list())

    def _update(self, id_list, value_list):
        for this_id, this_value in zip(id_list, value_list):
            # update id
            if this_id not in self.__id_list:
                self.__id_list.append(this_id)
                self.__id_value_dict[this_id] = list()
            # update value
            if this_value not in self.__id_value_dict[this_id]:
                self.__id_value_dict[this_id].append(this_value)

    def _gen_key(self, id_list, value_list):
        if not (len(id_list) == len(value_list)):
            raise Exception(
                "id_list and value list must have the same length, but {} and {}".format(
                    len(id_list), len(value_list)
                )
            )
        string_list = [
            "ID({})_VALUE({})".format(str(this_id), str(this_value))
            for this_id, this_value in filter(lambda x: x[0] in self.__id_list,
                zip(id_list, value_list)
            )
        ]
        return "_".join(string_list)

## ---- END helper functions ----


class BaseDecision(Component):
    REGISTRY = "decision"

    def __init__(self, post_fn=None, schedule_cfg=None):
        super().__init__(schedule_cfg=schedule_cfg)
        self.post_fn = post_fn
        self.decision_id = None

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
    """
    _CHOICES_EFFECT_DICT: key is id of leaf choices,
        value is the affected non leaf choices
    _choices_graph: the values of non leaf and leaf choices
        corresponding to leaf choices
    _derive_flag: flag for deriving of not
    """
    _CHOICES_EFFECT_DICT = defaultdict(list)
    _MAX_DECISION_THRES = 256
    def __init__(self, post_fn=None, schedule_cfg=None):
        super().__init__(post_fn=post_fn, schedule_cfg=schedule_cfg)
        self._choices_graph = RootRecord()
        self._derive_flag = False
        self._choices = None
        self._optional_choices = None

    @property
    @abc.abstractmethod
    def size(self):
        """
        Return choose size
        """

    # TODO: derive should be be executed on initialization or some leaf choices changing
    # Now only support on initialization
    @abc.abstractmethod
    @set_derive_flag
    def _derive_choices(self):
        """
        Derive choices for leaf and non leaf choice
        """

    def _regard_self_leaf(self, choices):
        """
        For non leaf Choices, there may be too many value candidates of
        all the root leaves to store the _choices_graph.
        Then we regard this non leaf Choices as virtual leaf to reduce computation cost.
        choices: list of the possible values
        """
        post_fn = self.post_fn or (lambda x: x)
        # record leaf choices
        id_str = str(id(self))
        for v in choices:
            self._choices_graph.set_target([id_str], [str(v)], post_fn(v))
        self._choices = copy.deepcopy(choices)

    @property
    def choices(self):
        self._derive_choices()
        post_fn = self.post_fn or (lambda x: x)
        return list(map(lambda x: post_fn(x), self._choices))

    def range(self):
        choices = self.choices
        return min(choices), max(choices)

    @property
    def num_choices(self):
        choices = self.choices
        return len(choices)

    @classmethod
    def expand_choices_effect_list(cls, source_list, target):
        for source in source_list:
            # for leaf Choices
            if isinstance(source, Choices):
                cls._CHOICES_EFFECT_DICT[str(id(source))].append(
                    target
                )
            elif isinstance(source, BaseChoices) and \
                source.__class__.__name__ != 'BaseChoices':
                for k, v in cls._CHOICES_EFFECT_DICT.items():
                    v_id = map(lambda x: id(x), v)
                    if id(source) in v_id:
                        cls._CHOICES_EFFECT_DICT[k].append(
                            target
                        )
            else:
                pass

    # set all affected Choices _derive_flag as False
    @classmethod
    def set_choices_effect_flag(cls, source):
        for choices in cls._CHOICES_EFFECT_DICT[
            str(id(source))
        ]:
            if isinstance(choices, BaseChoices):
                choices._derive_flag = False
                choices._choices = None
                choices._choices_graph._clear_map()

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
        self._size = size  # choose size
        self.replace = replace
        self.p = p

        if epoch_callback is not None:
            epoch_callback = partial(epoch_callback, self)
        self.epoch_callback = epoch_callback
        self.expand_choices_effect_list([self], self)

    @property
    def size(self):
        return self._size

    @set_derive_flag
    def _derive_choices(self):
        pass

    @property
    def choices(self):
        post_fn = self.post_fn or (lambda x: x)
        return list(map(lambda x: post_fn(x), self._choices))

    @choices.setter
    def choices(self, value):
        self._p = None
        if len(set(value) ^ set(self._choices)) != 0:
            self.set_choices_effect_flag(self)
            self._choices = value

    @property
    def p(self):
        if self._p is None:
            return [(1 / self.num_choices), ] * self.num_choices
        assert np.isclose(sum(self._p), 1.0) and len(
            self._p) == self.num_choices
        return self._p

    @p.setter
    def p(self, _p):
        assert _p is None or np.isclose(
            sum(_p), 1.0) and len(_p) == self.num_choices
        self._p = _p

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
        if self.epoch_callback is not None:
            self.epoch_callback(epoch)

        if not hasattr(self, "decision_id"):
            self.logger.warning(f"choice {self} is not bound to any module.")

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
            self._choices, size=self.size, replace=self.replace, p=self.p
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

        actual_choices = self.choices
        old_ind = actual_choices.index(old)
        if self.p is not None:
            whole_p = 1 - self.p[old_ind]
            if whole_p == 0:
                raise Exception(
                    "Choice {} cannot mutate from {}".format(self, old))
            mutate_p = [
                self.p[(old_ind + bias) % self.num_choices] / whole_p
                for bias in range(1, self.num_choices)
            ]
            mutate_thresh = np.cumsum(mutate_p)
            bias = np.where(np.random.rand() < mutate_thresh)[0][0] + 1
        else:
            bias = np.random.randint(1, self.num_choices)
        new_ind = (old_ind + bias) % self.num_choices
        return self._choices[new_ind]

    def to_string(self):
        return "Choices(choices: {}, size: {}, replace: {}, p: {})".format(
            self._choices,
            self.size,
            self.replace,
            self.p if self.p is not None else "null"
        )

    @classmethod
    def from_string(cls, string):
        sub_string = re.search(r"Choices\((.+)\)", string).group(1)
        sub_stringio = StringIO("{" + sub_string + "}")
        kwargs = yaml.load(sub_stringio, Loader=yaml.FullLoader)
        return cls(**kwargs)


class NonleafDecision(BaseDecision):
    def random_sample(self):
        raise InvalidUseException(
            "Cannot call `random_sample` on non-leaf decisions.")

    def mutate(self, old):
        raise InvalidUseException(
            "Cannot call `mutate` on non-leaf decisions.")

    @property
    def search_space_size(self):
        return 1

    @abc.abstractmethod
    @apply_post_fn
    def get_value(self, rollout):
        pass

    @abc.abstractmethod
    def get_choices(self, possible_value_list):
        """
        get the output _choices for possible_value_list
        """

    @abc.abstractmethod
    def set_children_id(self):
        pass


class BinaryNonleafDecision(NonleafDecision):
    def __init__(self, dec1, dec2, binary_op=None):
        super().__init__(schedule_cfg=None)

        assert binary_op is not None  # the default value is just to suppress pylint

        if not (isinstance(dec1, (int, float)) or isinstance(dec2, (int, float))):
            # if neither dec1/dec2 are instant numbers, run a check
            is_decision_id = int(isinstance(dec1, str)) + \
                int(isinstance(dec2, str))
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

    # TODO: How to construct the dependency between leaf and nonleaf Choices
    # when the global computation graph is not constructed yet and there is no decision_id for all Choices
    @classmethod
    def from_string(cls, string):
        if cls == BinaryNonleafDecision:
            raise InvalidUseException(
                "Cannot construct BinaryNonleafDecision from string, construct one subclass of it")
        match = re.search(r"{}".format(cls.__name__) +
                          r"\(dec1=(.+), dec2=(.+)\)", string)
        # only support parse id
        dec1_id, dec2_id = match.group(1), match.group(2)
        dec1_id, _ = _convert_to_number(dec1_id)
        dec2_id, _ = _convert_to_number(dec2_id)
        return cls(dec1_id, dec2_id)  # pylint: disable=no-value-for-parameter

    @apply_post_fn
    def get_value(self, rollout):
        dec1_value = _get_value(self.dec1, rollout)
        dec2_value = _get_value(self.dec2, rollout)
        val = self.binary_op(dec1_value, dec2_value)
        return val


    def get_choices(self, possible_value_list):
        value_list = itertools.product(*possible_value_list)
        value_list = list(map(lambda x: self.binary_op(*x), value_list))
        value_list = list(np.unique(value_list))
        return value_list

    def set_children_id(self):
        decs = []
        for i, dec in enumerate([self.dec1, self.dec2], 1):
            if isinstance(dec, BaseDecision):
                decs.append(dec)
                if not hasattr(dec, "decision_id") or dec.decision_id is None:
                    dec.decision_id = ".".join([self.decision_id, str(i)])
        return decs


class BinaryNonleafChoices(BinaryNonleafDecision, BaseChoices):
    """
    To write a new binary nonleaf choices, one should inherit BinaryNonleafChoices.
    Optionally, one can override corresponding operator special methods of
    some BaseDecision classes (e.g., BaseChoices)
    """
    @property
    def size(self):
        if isinstance(self.dec1, BaseChoices):
            return self.dec1.size
        return self.dec2.size

    @set_derive_flag
    def _derive_choices(self):
        # init
        self._choices = []
        for tc in [self.dec1, self.dec2]:
            if isinstance(tc, BaseDecision):
                tc._derive_choices()
        for total_length, root_id_list, value_list, target_list in generator_target_list([
            self.dec1,
            self.dec2,
        ]):
            if total_length > self._MAX_DECISION_THRES:
                # for plenty decision
                possible_value_list = _get_parent_choices([self.dec1, self.dec2])
                choices = self.get_choices(possible_value_list)
                self._regard_self_leaf(choices)
                break
            # other
            outputs_value = self.binary_op(*target_list)
            self._choices.append(outputs_value)
            # update self._choice_graph
            post_fn = self.post_fn or (lambda x: x)
            self._choices_graph.set_target(
                root_id_list,
                value_list,
                post_fn(outputs_value),
            )
        self._choices = list(np.unique(self._choices))


class BinaryNonleafChoiceGetter(object):
    """
    A helper class for pickling non-leaf choices (e.g., ChoiceMul),
    which are created dynamically as non-top-level classes by
    `helper_register_nonleaf_binary_choice`.

    See `helper_register_nonleaf_binary_choice`, in the dynamically created classes,
    the `__reduce__` special method is implemented to return an instance of
    BinaryNonleafChoiceGetter. Upon unpickling, `BinaryNonleafChoiceGetter().__call__()`
    is called to create an initial instance of the (dynamically created) non-leaf choice classes.
    """
    def __call__(self, sub_class_name):
        cls = BinaryNonleafChoices.get_class_(sub_class_name)
        instance = BinaryNonleafChoiceGetter()
        instance.__class__ = cls
        return instance


def helper_register_nonleaf_binary_choice(
        binary_op, class_name_suffix, registry_name, operator_func_name=None):
    """
    helper_register_nonleaf_binary_choice(lambda x, y: max(x,y), "ChoicesMax", None)
    """

    def _constructor(self, dec1, dec2):
        BinaryNonleafDecision.__init__(self, dec1, dec2, binary_op)

    def _choices_constructor(self, dec1, dec2, optional_choices=None):
        super(BinaryNonleafChoiceCls, self).__init__(dec1, dec2)
        self._optional_choices = optional_choices
        self.expand_choices_effect_list([dec1, dec2], self)

    def _reduce_choice_cls(self):
        return (
            BinaryNonleafChoiceGetter(),
            (registry_name,),
            self.__dict__.copy()
        )

    class_name = "Decision{}".format(class_name_suffix.capitalize())
    BinaryNonleafDecisionCls = type(class_name, (BinaryNonleafDecision,), {
        "__init__": _constructor, # constructor
    })
    class_name = "Choice{}".format(class_name_suffix.capitalize())
    BinaryNonleafChoiceCls = type(class_name, (BinaryNonleafDecisionCls, BinaryNonleafChoices), {
        "__init__": _choices_constructor, # choices constructor
        "NAME": registry_name, # registry name
        "__reduce__": _reduce_choice_cls
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

DecisionMul, ChoiceMul = helper_register_nonleaf_binary_choice(operator.mul, "mul", "choices_mul", "__mul__")
DecisionAdd, ChoiceAdd = helper_register_nonleaf_binary_choice(operator.add, "add", "choices_add", "__add__")
DecisionMax, ChoiceMax = helper_register_nonleaf_binary_choice(max, "max", "choices_max")
DecisionMin, ChoiceMin = helper_register_nonleaf_binary_choice(min, "min", "choices_min")
DecisionMinus, ChoiceMinus = helper_register_nonleaf_binary_choice(
    operator.sub, "minus", "choices_minus", "__sub__")
DecisionDiv, ChoiceDiv = helper_register_nonleaf_binary_choice(
    operator.truediv, "div", "choices_div", "__truediv__")


class SelectNonleafDecision(NonleafDecision):
    def __init__(self, select_list, select_key):
        super().__init__(schedule_cfg=None)

        if isinstance(select_key, str):
            for select_ele in select_list:
                if not isinstance(select_ele, (int, float, str)):
                    raise Exception(
                        "When select key is decision_id, all of the select_list should be decision_id"
                    )
        else:
            assert isinstance(select_key, BaseDecision), \
                "select key should be in type of str or BaseDecision"
            # check for select value range
            key_min, key_max = select_key.range()
            assert key_min >= 0 and key_max < len(select_list), \
                "the range of select key should be contained by range of select_list"
            for select_ele in select_list:
                if not (isinstance(select_ele, (int, float)) or isinstance(select_ele, BaseDecision)):
                    raise Exception(
                        "when select key is BaseDecision, all of the select list should be num or BaseDecision"
                    )
        # init
        self.select_list = select_list
        self.select_key = select_key

    def to_string(self):
        # warn if no `decision_id` is available
        for select_ele in self.select_list:
            if isinstance(select_ele, BaseDecision) and not hasattr(select_ele, "decision_id"):
                self.logger.warn((
                    "`decision_id` not assigned for {}."
                    "The resulting string cannot be used to reconstruct the "
                    "non-leaf decision correctly"
                    ).format(select_ele)
                )
        if isinstance(self.select_key, BaseDecision) and not hasattr(self.select_key, "decision_id"):
            self.logger.warn((
                "`decision_id` not assigned for {}."
                "The resulting string cannot be used to reconstruct the "
                "non-leaf decision correctly"
                ).format(self.select_key)
            )
        return "{}(select_list=[{}], select_key={})".format(
            self.__class__.__name__,
            ",".join([
                str(getattr(select_ele, "decision_id", select_ele)) \
                for select_ele in self.select_list
            ]),
            getattr(self.select_key, "decision_id", self.select_key)
        )

    @classmethod
    def from_string(cls, string):
        if cls == SelectNonleafDecision:
            raise InvalidUseException(
                "Cannot construct SelectNonleafDecision from string, construct one subclass of it")
        match = re.search(
            r"{}".format(cls.__name__) + \
            r"\(select_list=\[(.+)\], select_key=(.+)\)",
            string
        )
        # only support for decision id
        select_list_id, select_key_id = match.group(1), match.group(2)
        select_list_id = select_list_id.split(',')
        for i in range(len(select_list_id)):
            select_list_id[i], _ = _convert_to_number(select_list_id[i])
        select_key_id, _ = _convert_to_number(select_key_id)
        return cls(select_list_id, select_key_id) # pylint: disable=no-value-for-parameter

    @apply_post_fn
    def get_value(self, rollout):
        select_key_value = _get_value(self.select_key, rollout)
        val = _get_value(self.select_list[select_key_value], rollout)
        return val

    def get_choices(self, possible_value_list):
        value_list = [set(value) for value in possible_value_list[:-1]]
        value_list = functools.reduce(lambda x, y: x | y, value_list)
        return list(value_list)

class SelectNonleafChoices(SelectNonleafDecision, BaseChoices):
    """
    The first paramater should be list of Choices or number (int or float)
    The second para must be instance of Choices
    Optional_choices give the possible choices of the node, without deriving
    You can use Like:
        b = germ.SelectNonLeafChoices([a1, a2, a3], z, [1, 2, 3])
    """
    def __init__(self, select_list, select_key, optional_choices=None):
        super().__init__(select_list=select_list, select_key=select_key)
        self._optional_choices = optional_choices
        self.expand_choices_effect_list(select_list + [select_key], self)
    @property
    def size(self):
        return self.select_key.size

    @set_derive_flag
    def _derive_choices(self):
        # init
        self._choices = []
        for tc in self.select_list + [self.select_key]:
            if isinstance(tc, BaseDecision):
                tc._derive_choices()
        for total_length, root_id_list, value_list, target_list in generator_target_list(
            self.select_list + \
            [self.select_key]
        ):
            if total_length > self._MAX_DECISION_THRES:
                    # for plenty decision
                possible_value_list = _get_parent_choices(self.select_list + [self.select_key])
                choices = self.get_choices(possible_value_list)
                self._regard_self_leaf(choices)
                break
            outputs_value = target_list[target_list[-1]]
            self._choices.append(outputs_value)
            # update self._choice_graph
            post_fn = self.post_fn or (lambda x: x)
            self._choices_graph.set_target(
                root_id_list,
                value_list,
                post_fn(outputs_value),
            )
        self._choices = list(np.unique(self._choices))

    def set_children_id(self):
        decs = []
        for i, dec in enumerate(self.select_list + [self.select_key], 1):
            if isinstance(dec, BaseDecision):
                decs.append(dec)
                if not hasattr(dec, "decision_id") or dec.decision_id is None:
                    dec.decision_id = ".".join([self.decision_id, str(i)])
        return decs
