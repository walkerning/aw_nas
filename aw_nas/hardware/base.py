# -*- coding: utf-8 -*-

import abc
from collections import OrderedDict
from six import StringIO
import yaml

from aw_nas import utils
from aw_nas.base import Component
from aw_nas.common import SearchSpace
class BaseHardwareCompiler(Component):
    REGISTRY = "hardware_compiler"
    
    def __init__(self):
        super(BaseHardwareCompiler, self).__init__(schedule_cfg=None)

    @abc.abstractmethod
    def compile(self, compile_name, net_cfg, result_dir):
        pass

    @abc.abstractmethod
    def hwobj_net_to_primitive(self, hwobj_type, prof_result_file, prof_prim_file):
        pass


class BaseHardwareObjectiveModel(Component):
    REGISTRY = "hardware_obj_model"

    def __init__(self, schedule_cfg):
        super(BaseHardwareObjectiveModel, self).__init__(schedule_cfg)

    @abc.abstractmethod
    def predict(self, rollout):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class MixinProfilingSearchSpace(Component):
    REGISTRY = 'mixin_search_space'

    @abc.abstractmethod
    def generate_profiling_primitives(self):
        pass

    @abc.abstractmethod
    def parse_profiling_primitives(
            self, prof_prims, prof_prims_cfg, hwobjmodel_type, hwobjmodel_cfg):
        model = BaseHardwareObjectiveModel.get_class_(hwobjmodel_type)(
            prof_prims, prof_prims_cfg, **hwobjmodel_cfg)
        return model

    @classmethod
    def get_default_prof_config_str(cls):
        default_cfg = utils.get_default_argspec(cls.generate_profiling_primitives)
        stream = StringIO()
        cfg = OrderedDict(default_cfg)
        yaml.safe_dump(cfg, stream=stream, default_flow_style=False)
        return stream.getvalue()
