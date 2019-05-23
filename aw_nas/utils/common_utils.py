# -*- coding: utf-8 -*-
#pylint: disable=attribute-defined-outside-init

import os
import sys
import inspect
import collections
import six

import numpy as np
import scipy
import scipy.signal

from aw_nas.utils.registry import RegistryMeta

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class keydefaultdict(collections.defaultdict): #pylint: disable=invalid-name
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        ret = self[key] = self.default_factory(key) #pylint: disable=not-callable
        return ret

class attr_dict_wrapper(object): #pylint: disable=invalid-name
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]

def compute_returns(rewards, gamma, length=None):
    if not isinstance(rewards, collections.Sequence):
        assert length is not None
        _rewards = np.zeros((length,))
        _rewards[-1] = rewards
    else:
        _rewards = rewards
    return scipy.signal.lfilter([1], [1, -gamma], _rewards[::-1], axis=0)[::-1]

def get_default_argspec(func):
    if sys.version_info.major == 3:
        # python 3
        sig = inspect.signature(func) #pylint: disable=no-member
        return [(n, param.default) for n, param in six.iteritems(sig.parameters) \
                if not param.default is param.empty]
    # python 2
    sig = inspect.getargspec(func) #pylint: disable=deprecated-method
    return list(reversed(list(zip(reversed(sig.args),
                                  reversed(sig.defaults)))))

def add_text_prefix(text, prefix):
    lines = text.split("\n")
    return "\n".join([prefix + line if line else line for line in lines])

def component_sample_config_str(comp_name, prefix):
    all_text = prefix + "## ---- Component {} ----\n".format(comp_name)

    for type_name, cls in six.iteritems(RegistryMeta.all_classes(comp_name)):

        all_text += prefix + "# ---- Type {} ----\n".format(type_name)
        all_text += prefix + "{}_type: {}\n".format(comp_name, type_name)
        all_text += prefix + "{}_cfg:\n".format(comp_name)

        # write the default configuration
        config_str = cls.get_default_config_str()
        all_text += add_text_prefix(config_str, prefix + "  ")

        all_text += prefix + "# ---- End Type {} ----\n".format(type_name)

    all_text += prefix + "## ---- End Component {} ----\n".format(comp_name)
    return all_text

def get_cfg_wrapper(cfg, key):
    if isinstance(cfg, dict):
        return cfg
    return {key: cfg}

def _assert_keys(dct, mandatory_keys, possible_keys, name):
    if mandatory_keys:
        assert set(mandatory_keys).issubset(dct.keys()),\
            "{} schedule cfg must have keys: ({})".format(name, ", ".join(mandatory_keys))
    if possible_keys:
        addi_keys = set(dct.keys()).difference(possible_keys)
        assert not addi_keys,\
            "{} schedule cfg cannot have keys: ({}); all possible keys: ({})"\
                .format(name, ", ".join(addi_keys), ", ".join(possible_keys))

_SUPPORTED_TYPES = {"value", "mul", "add"}
def check_schedule_cfg(schedule):
    """
    Check the sanity of the schedule configuration.
    Currently supported type: mul, add, value.

    Rules: mul  : [boundary / every], step, start, [optional: min, max]
           add  : [boundary / every], step, start, [optional: min, max]
           value: boundary, value
    """
    assert "type" in schedule,\
        "Schedule config must have `type` specified: one in "+", ".join(_SUPPORTED_TYPES)
    type_ = schedule["type"]
    assert type_ in _SUPPORTED_TYPES, "Supported schedule config type: "+", ".join(_SUPPORTED_TYPES)

    if type_ == "value":
        _assert_keys(schedule, ["value", "boundary"], None, "value")
        assert len(schedule["value"]) == len(schedule["boundary"]),\
            "value schedule cfg `value` and `boundary` should be of the same length."
        assert schedule["boundary"][0] == 1,\
            "value schedule cfg must have `boundary` config start from 1."
    else: # mul/add
        _assert_keys(schedule, ["step", "start"],
                     ["type", "step", "start", "boundary", "every", "min", "max"], "value")
        assert "boundary" in schedule or "every" in schedule,\
            "{} schedule cfg must have one of `boundary` and `every` key existed.".format(type_)
        assert not ("boundary" in schedule and "every" in schedule),\
            "{} shcedule cfg cannot have `boundary` and `every` key in the mean time.".format(type_)

def get_schedule_value(schedule, epoch):
    """
    See docstring of `check_schedule_cfg` for details.
    """

    type_ = schedule["type"]
    if type_ == "value":
        ind = list(np.where(epoch < np.array(schedule["boundary"]))[0])
        if not ind: # if epoch is larger than the last boundary
            ind = len(schedule["boundary"]) - 1
        else:
            ind = ind[0] - 1
        next_v = schedule["value"][ind]
    else:
        min_ = schedule.get("min", -np.inf)
        max_ = schedule.get("max", np.inf)
        if "every" in schedule:
            ind = (epoch - 1) // schedule["every"]
        else: # "boundary" in schedule
            ind = list(np.where(epoch < np.array(schedule["boundary"]))[0])
            if not ind: # if epoch is larger than the last boundary
                ind = len(schedule["boundary"])
            else:
                ind = ind[0]
        if type_ == "mul":
            next_v = schedule["start"] * schedule["step"] ** ind
        else: # type_ == "add"
            next_v = schedule["start"] + schedule["step"] * ind
        next_v = max(min(next_v, max_), min_)
    return next_v

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def softmax(arr):
    e_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
    return e_arr / np.sum(e_arr, axis=-1, keepdims=True)
