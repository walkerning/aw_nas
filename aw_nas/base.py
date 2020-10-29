# -*- coding: utf-8 -*-

from collections import OrderedDict
import six
from six import StringIO
import yaml

from aw_nas import utils
from aw_nas.utils import RegistryMeta
from aw_nas.utils import getLogger
from aw_nas.utils.vis_utils import WrapWriter
from aw_nas.utils.exception import expect, ConfigException

# Make yaml.safe_dump support OrderedDict
yaml.add_representer(OrderedDict,
                     lambda dumper, data: dumper.represent_mapping(
                         'tag:yaml.org,2002:map',
                         data.items()
                     ), Dumper=yaml.dumper.SafeDumper)

@six.add_metaclass(RegistryMeta)
class Component(object):

    SCHEDULABLE_ATTRS = []

    def __init__(self, schedule_cfg):
        self._logger = None
        self.schedule_cfg = schedule_cfg
        if schedule_cfg is not None:
            self.schedule_cfg = sorted(self.schedule_cfg.items()) # convert to list of tuples
            for name, cfg in self.schedule_cfg:
                expect(name in self.SCHEDULABLE_ATTRS,
                       "{} not in the schedulable attributes of {}: {}".format(
                           name, self.__class__.__name__, self.SCHEDULABLE_ATTRS
                       ), ConfigException)
                utils.check_schedule_cfg(cfg)

        self.writer = WrapWriter(None) # a none writer
        self.epoch = 0

    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["writer"]
        if "_logger" in state:
            del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # just set self.writer to be a none writer
        # not reset to the original logdir, this is reasonable
        self.writer = WrapWriter(None)
        # set self._logger to None
        self._logger = None

    def setup_writer(self, writer):
        self.writer = writer

    def on_epoch_start(self, epoch):
        expect(epoch >= 1, "Epoch should >= 1")
        self.epoch = epoch
        if self.schedule_cfg:
            new_values = []
            for name, cfg in self.schedule_cfg:
                new_value = utils.get_schedule_value(cfg, epoch)
                setattr(self, name, new_value)
                new_values.append((name, new_value))
            _schedule_str = "\n\t".join(["{:10}: {:10}".format(n, str(v)) for n, v in new_values])
            self.logger.info("Epoch %3d on_epoch_start: schedule values:\n\t%s",
                             epoch, _schedule_str)

    def on_epoch_end(self, epoch):
        pass

    @classmethod
    def get_default_config(cls):
        return utils.get_default_argspec(cls.__init__)

    @classmethod
    def get_default_config_str(cls):
        stream = StringIO()
        stream.write("# Schedulable attributes: {}\n".format(", ".join(cls.SCHEDULABLE_ATTRS)))
        cfg = OrderedDict(cls.get_default_config())
        yaml.safe_dump(cfg, stream=stream, default_flow_style=False)
        return stream.getvalue()

    @classmethod
    def get_current_config_str(cls, cfg):
        stream = StringIO()
        whole_cfg = OrderedDict(cls.get_default_config())
        whole_cfg.update(cfg)
        yaml.safe_dump(whole_cfg, stream=stream, default_flow_style=False)
        return stream.getvalue()
