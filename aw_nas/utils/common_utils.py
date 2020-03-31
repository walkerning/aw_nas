# -*- coding: utf-8 -*-
#pylint: disable=attribute-defined-outside-init

import os
import sys
import time
import copy
import shutil
import inspect
import functools
import collections
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
import six

import numpy as np
import scipy
import scipy.signal

from aw_nas.utils.registry import RegistryMeta
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.utils.log import logger as _logger

_HOME_DIR = os.environ.get("AWNAS_HOME", os.path.expanduser("~/awnas"))

class Context(object):
    def __init__(self, num_init_nodes, num_layers, use_stem=True,
                 previous_cells=None, current_cell=None, previous_op=None, current_op=None):
        self.use_stem = use_stem
        self.num_init_nodes = num_init_nodes
        self.num_layers = num_layers
        self.previous_cells = previous_cells or []
        self.current_cell = current_cell or []
        self.previous_op = previous_op or []
        self.current_op = current_op or []
        self._is_inject = dict()
        self._num_conn = dict()
        self._last_conv_modules = dict()

    @property
    def next_op_index(self):
        return len(self.previous_op), len(self.current_op)

    @property
    def next_step_index(self):
        return len(self.previous_cells) - (1 if self.use_stem else 0), len(self.current_cell)

    @property
    def is_last_concat_op(self):
        _, n_s = self.next_step_index
        return self.is_end_of_cell or (n_s > self.num_init_nodes and self.is_end_of_step)

    @property
    def is_end_of_cell(self):
        # next_cell, next_step
        n_c, n_s = self.next_step_index
        return sum(self.next_op_index) == 0 and n_s == 0 and self.num_layers >= n_c > 0

    @property
    def is_end_of_step(self):
        _, n_s = self.next_step_index
        return sum(self.next_op_index) == 0 and n_s > 0

    @property
    def is_end_of_op(self):
        return len(self.current_op) == 0

    @property
    def last_state(self):
        for lst in [self.current_op, self.previous_op, self.current_cell, self.previous_cells]:
            if lst:
                return lst[-1]
        return None # empty context, which is not likely to happen

    @last_state.setter
    def last_state(self, state):
        for lst in [self.current_op, self.previous_op, self.current_cell, self.previous_cells]:
            if lst:
                lst[-1] = state
                break
        else:
            raise Exception("Empty context, set failed")

    @property
    def index(self):
        next_cell, next_step = self.next_step_index
        next_conn, next_op_step = self.next_op_index
        return next_cell, next_step, next_conn, next_op_step

    def flag_inject(self, is_inject):
        self._is_inject[self.index] = is_inject

    @property
    def is_last_inject(self):
        return self._is_inject.get(self.index, True)

    @property
    def last_conv_module(self):
        return self._last_conv_modules.get(self.index, None)

    @last_conv_module.setter
    def last_conv_module(self, value):
        self._last_conv_modules[self.index] = value

    def __repr__(self):
        next_cell, next_step, next_conn, next_op_step = self.index
        return "Context(next_cell={}, next_step={}, next_conn={}, next_op_step={})"\
            .format(next_cell, next_step, next_conn, next_op_step)

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

def recur_apply(func, lst, depth=0, out_type=list):
    return out_type([recur_apply(func, item, depth-1, out_type)
                     if isinstance(item, (tuple, list)) and depth > 0 \
                     else func(item) for item in lst])

class Ticker(object):
    def __init__(self, name):
        self.name = name
        self.total_time = 0.
        self.cur_time = None
        self.tick()
        self.logger = _logger.getChild("ticker_{}".format(name))

    def tick(self, message=""):
        cur_time = time.time()
        if self.cur_time is not None:
            elapsed = cur_time - self.cur_time
            self.logger.debug("Ticker %s: %s: %.6f s", self.name, message, elapsed)
            self.total_time += elapsed
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

def get_argspec(func):
    if sys.version_info.major == 3:
        # python 3
        sig = inspect.signature(func) #pylint: disable=no-member
        return OrderedDict([(n, param.default) for n, param in six.iteritems(sig.parameters)])

    sig = inspect.getargspec(func) #pylint: disable=deprecated-method
    return OrderedDict(list(zip(sig.args,
                                [None] * (len(sig.args) - len(sig.defaults)) + list(sig.defaults))))

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

def namedtuple_with_defaults(name, fields, defaults):
    if sys.version_info.major == 3 and (
            sys.version_info.minor > 7 or
            (sys.version_info.minor == 7 and sys.version_info.micro >= 6)):
        return namedtuple(name, fields, defaults)
    type_ = namedtuple(name, fields)
    if defaults:
        type_.__new__.__defaults__ = tuple(defaults)
    return type_

## --- text utils ---
def add_text_prefix(text, prefix):
    lines = text.split("\n")
    return "\n".join([prefix + line if line else line for line in lines])

def component_sample_config_str(comp_name, prefix, filter_funcs=None, cfg_name=None):
    if cfg_name is None:
        cfg_name = comp_name
    filter_funcs = filter_funcs or []
    all_text = prefix + "## ---- Component {} ----\n".format(cfg_name)

    for type_name, cls in six.iteritems(RegistryMeta.all_classes(comp_name)):
        try:
            is_skip = any(not func(cls) for func in filter_funcs)
        except Exception as e: #pylint: disable=broad-except
            # some plugin class might be wrongly implemented, check here
            import traceback
            traceback.print_exc()
            _logger.getChild("utils")\
                   .warning("Skip %s: %s(%s) as exception occurs in checking. %s: %s",
                            comp_name, type_name, cls, e.__class__.__name__, str(e))
        if is_skip:
            continue

        all_text += prefix + "# ---- Type {} ----\n".format(type_name)
        all_text += prefix + "{}_type: {}\n".format(cfg_name, type_name)
        all_text += prefix + "{}_cfg:\n".format(cfg_name)

        # write the default configuration
        config_str = cls.get_default_config_str()
        all_text += add_text_prefix(config_str, prefix + "  ")

        all_text += prefix + "# ---- End Type {} ----\n".format(type_name)

    all_text += prefix + "## ---- End Component {} ----\n".format(cfg_name)
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


## --- cache utils ---
def cache_results(cache_params, key_funcs, buffer_size):
    if callable(key_funcs):
        key_funcs = [key_funcs] * len(cache_params)
    def decorator(func):
        sig_dct = OrderedDict(get_argspec(func))
        cache_dict = OrderedDict()
        cache_hit_and_miss = [0, 0] # hit, miss
        @functools.wraps(func)
        def _inner_func(*args, **kwargs):
            params = copy.deepcopy(sig_dct)
            params.update(kwargs)
            for value, arg_name in zip(args, sig_dct):
                params[arg_name] = value
            key_tuple = []
            for name, key_func in zip(cache_params, key_funcs):
                key_tuple.append(key_func(params[name]))
            key_tuple = tuple(key_tuple)
            if key_tuple in cache_dict:
                cache_hit_and_miss[0] += 1
                return cache_dict[key_tuple]
            cache_hit_and_miss[1] += 1
            res = func(*args, **kwargs)
            cache_dict[key_tuple] = res
            if len(cache_dict) > buffer_size:
                cache_dict.popitem(last=False)
            return res
        _inner_func.cache_dict = cache_dict
        _inner_func.cache_hit_and_miss = cache_hit_and_miss
        return _inner_func
    return decorator


## ---- thread utils ----
class LazyThreadLocal(six.moves._thread._local):
    def __init__(self, creator_map=None):
        super(LazyThreadLocal, self).__init__()
        if creator_map is not None:
            assert isinstance(creator_map, dict)
        self.creator_map = creator_map

    def __getattr__(self, name):
        if name in self.creator_map:
            value = self.creator_map[name]()
            setattr(self, name, value)
            return value
        raise AttributeError(("LazyThreadlocal object do not have attribute named {}, "
                              "also not specified in the lazy creator map.").format(name))


def make_divisible(v, divisor, min_val=None):
    """
    ref: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def get_sub_kernel(kernel, sub_kernel_size):
    original_size = kernel.shape[-1]
    center = original_size // 2
    width = sub_kernel_size // 2
    left = center - width
    right = center + width + 1
    return kernel[:, :, left:right, left:right].contiguous()

def l1norm_order(filters):
    norm = torch.sum(torch.abs(filters.data), dim=(0, 2, 3))
    _, sorted_indices = torch.sort(norm, descending=True)
    return sorted_indices


def _get_channel_mask(filters, num_channels):
    norm_tensor = np.abs(filters.cpu().detach().numpy()).sum(axis=3).sum(axis=2).sum(axis=0)
    norm_tensor = sorted(zip(range(len(norm_tensor)), norm_tensor), key=lambda x: x[1], reverse=True)
    channel_order = [x[0] for x in norm_tensor]
    mask = np.zeros(filters.shape[1], dtype=np.bool)
    reserved_channels = channel_order[:num_channels]
    mask[reserved_channels] = 1
    return mask