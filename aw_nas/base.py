# -*- coding: utf-8 -*-

import six

from aw_nas import utils
from aw_nas.utils import RegistryMeta
from aw_nas.utils import logger as _logger

@six.add_metaclass(RegistryMeta)
class Component(object):

    SCHEDULABLE_ATTRS = []

    def __init__(self, schedule_cfg):
        self.logger = _logger.getChild(self.__class__.__name__)
        self.schedule_cfg = schedule_cfg
        if schedule_cfg is not None:
            self.schedule_cfg = sorted(self.schedule_cfg.items()) # convert to list of tuples
            for name, cfg in self.schedule_cfg:
                assert name in self.SCHEDULABLE_ATTRS,\
                    "{} not in the schedulable attributes of {}: {}".format(
                        name, self.__class__.__name__, self.SCHEDULABLE_ATTRS
                    )
                utils.check_schedule_cfg(cfg)

    def on_epoch_start(self, epoch):
        assert epoch >= 1, "Epoch should >= 1"
        if self.schedule_cfg:
            new_values = []
            for name, cfg in self.schedule_cfg:
                new_value = utils.get_schedule_value(cfg, epoch)
                setattr(self, name, new_value)
                new_values.append((name, new_value))
            _schedule_str = "\n\t".join(["{:10}: {:10}".format(n, v) for n, v in new_values])
            self.logger.info("Epoch %3d on_epoch_start: schedule values:\n\t%s",
                             epoch, _schedule_str)

    def on_epoch_end(self, epoch):
        pass
