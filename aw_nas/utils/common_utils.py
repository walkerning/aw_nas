# -*- coding: utf-8 -*-
#pylint: disable=attribute-defined-outside-init

import os
import sys
import time
import shutil
import inspect
import functools
import collections
from collections import OrderedDict
from contextlib import contextmanager
import six

import numpy as np
import scipy
import scipy.signal

from aw_nas.utils.registry import RegistryMeta
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.utils.log import logger as _logger

_HOME_DIR = os.environ.get("AWNAS_HOME", os.path.expanduser("~/awnas"))

## --- misc helpers ---
@contextmanager
def nullcontext():
    yield

def makedir(path, remove=False):
    if os.path.exists(path) and remove:
        shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_awnas_dir(env, name):
    # try to fetch from environment variable
    dir_ = os.environ.get(env, None)
    if dir_ is None:
        # if not in environment variable, return the default
        dir_ = os.path.join(_HOME_DIR, name)
    return makedir(dir_)

def flatten_list(lst):
    return functools.reduce(lambda s, l: s + list(l) \
                            if isinstance(l, (tuple, list)) else s + [l],
                            lst, [])

def recur_apply(func, lst, depth=0):
    return [recur_apply(func, item, depth-1) if isinstance(item, (tuple, list)) and depth > 0 \
            else func(item) for item in lst]

class Ticker(object):
    def __init__(self, name):
        self.name = name
        self.cur_time = None
        self.tick()
        self.logger = _logger.getChild("ticker_{}".format(name))

    def tick(self, message=""):
        cur_time = time.time()
        if self.cur_time is not None:
            elapsed = cur_time - self.cur_time
            self.logger.debug("Ticker %s: %s: %.3f s", self.name, message, elapsed)
        self.cur_time = cur_time

class OrderedStats(object):
    def __init__(self):
        self.stat_meters = None

    def __nonzero__(self):
        return self.stat_meters is not None

    __bool__ = __nonzero__

    def update(self, stats):
        if self.stat_meters is None:
            self.stat_meters = OrderedDict([(n, AverageMeter()) for n in stats])
        [self.stat_meters[n].update(v) for n, v in stats.items()]

    def avgs(self):
        if self.stat_meters is None:
            return None
        return OrderedDict((n, meter.avg) for n, meter in self.stat_meters.items())

    def items(self):
        return self.stat_meters.items() if self.stat_meters is not None else None

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


## --- math utils ---
def compute_returns(rewards, gamma, length=None):
    if not isinstance(rewards, collections.Sequence):
        assert length is not None
        _rewards = np.zeros((length,))
        _rewards[-1] = rewards
    else:
        _rewards = rewards
    return scipy.signal.lfilter([1], [1, -gamma], _rewards[::-1], axis=0)[::-1]

def softmax(arr):
    e_arr = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
    return e_arr / np.sum(e_arr, axis=-1, keepdims=True)


## --- Python 2/3 compatibility utils ---
class abstractclassmethod(classmethod):
    #pylint: disable=too-few-public-methods,invalid-name
    # for python2 compatibility
    __isabstractmethod__ = True

    def __init__(self, a_callable):
        a_callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(a_callable)

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


## --- text utils ---
def add_text_prefix(text, prefix):
    lines = text.split("\n")
    return "\n".join([prefix + line if line else line for line in lines])

def component_sample_config_str(comp_name, prefix, filter_funcs=None):
    filter_funcs = filter_funcs or []
    all_text = prefix + "## ---- Component {} ----\n".format(comp_name)

    for type_name, cls in six.iteritems(RegistryMeta.all_classes(comp_name)):
        try:
            is_skip = any(not func(cls) for func in filter_funcs)
        except Exception as e: #pylint: disable=broad-except
            # some plugin class might be wrongly implemented, check here
            import traceback
            traceback.print_exc()
            _logger.getChild("utils")\
                   .warn("Skip %s: %s(%s) as exception occurs in checking. %s: %s",
                         comp_name, type_name, cls, e.__class__.__name__, str(e))
        if is_skip:
            continue

        all_text += prefix + "# ---- Type {} ----\n".format(type_name)
        all_text += prefix + "{}_type: {}\n".format(comp_name, type_name)
        all_text += prefix + "{}_cfg:\n".format(comp_name)

        # write the default configuration
        config_str = cls.get_default_config_str()
        all_text += add_text_prefix(config_str, prefix + "  ")

        all_text += prefix + "# ---- End Type {} ----\n".format(type_name)

    all_text += prefix + "## ---- End Component {} ----\n".format(comp_name)
    return all_text


## --- schedule utils ---
def _assert_keys(dct, mandatory_keys, possible_keys, name):
    if mandatory_keys:
        expect(set(mandatory_keys).issubset(dct.keys()),
               "{} schedule cfg must have keys: ({})".format(name, ", ".join(mandatory_keys)))
    if possible_keys:
        addi_keys = set(dct.keys()).difference(possible_keys)
        expect(not addi_keys,
               "{} schedule cfg cannot have keys: ({}); all possible keys: ({})"\
               .format(name, ", ".join(addi_keys), ", ".join(possible_keys)))

_SUPPORTED_TYPES = {"value", "mul", "add"}
def check_schedule_cfg(schedule):
    """
    Check the sanity of the schedule configuration.
    Currently supported type: mul, add, value.

    Rules: mul  : [boundary / every], step, start, [optional: min, max, start_epoch]
           add  : [boundary / every], step, start, [optional: min, max, start_epoch]
           value: boundary, value
    """
    expect("type" in schedule,
           "Schedule config must have `type` specified: one in "+", ".join(_SUPPORTED_TYPES),
           ConfigException)
    type_ = schedule["type"]
    expect(type_ in _SUPPORTED_TYPES,
           "Supported schedule config type: "+", ".join(_SUPPORTED_TYPES),
           ConfigException)

    if type_ == "value":
        _assert_keys(schedule, ["value", "boundary"], None, "value")
        expect(len(schedule["value"]) == len(schedule["boundary"]),
               "value schedule cfg `value` and `boundary` should be of the same length.",
               ConfigException)
        expect(schedule["boundary"][0] == 1,
               "value schedule cfg must have `boundary` config start from 1.", ConfigException)
    else: # mul/add
        _assert_keys(schedule, ["step", "start"],
                     ["type", "step", "start", "boundary",
                      "every", "min", "max", "start_epoch"], "mul/add")
        expect("boundary" in schedule or "every" in schedule,
               "{} schedule cfg must have one of `boundary` and `every` key existed.".format(type_),
               ConfigException)
        expect(not ("boundary" in schedule and "every" in schedule),
               "{} shcedule cfg cannot have `boundary` and `every` key in the mean time."\
               .format(type_), ConfigException)

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
        start_epoch = schedule.get("start_epoch", 0)
        epoch = epoch - start_epoch
        if epoch <= 0:
            return schedule["start"]
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
