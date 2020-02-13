# -*- coding: utf-8 -*-
#pylint: disable-all

import os
from collections import OrderedDict

import yaml
import numpy as np

from aw_nas import AwnasPlugin
from aw_nas.controller import BaseController
from aw_nas.common import Rollout, assert_rollout_type, rollout_from_genotype_str

class PoolController(BaseController):
    NAME = "pool_controller"

    def __init__(self, search_space, device,
                 rollout_type="discrete", mode="eval",
                 arch_file_or_dir=None,
                 schedule_cfg=None):
        super(PoolController, self).__init__(search_space, rollout_type, mode, schedule_cfg)

        self.device = device
        self.arch_file_or_dir = arch_file_or_dir
        if os.path.isdir(self.arch_file_or_dir):
            self.arch_files = [os.path.join(self.arch_file_or_dir, fname)
                               for fname in os.listdir(self.arch_file_or_dir)]
        else:
            self.arch_files = [self.arch_file_or_dir]
        self.genotype_list = []
        for arch_file in self.arch_files:
            with open(arch_file, "r") as rf:
                genotypes = yaml.load(rf)
            self.genotype_list += list(genotypes)
        self._num_arch = len(self.genotype_list)
        self.logger.info("Number of genotypes: %d", self._num_arch)

        self.indexes = np.arange(len(self.genotype_list))
        np.random.shuffle(self.indexes)
        self.cur_ind = 0

    def sample(self, n=1, batch_size=1):
        s_num = 0
        rollouts = []
        for s_num in range(n):
            r = rollout_from_genotype_str(self.genotype_list[self.indexes[self.cur_ind]],
                                          self.search_space)
            rollouts.append(r)
            self.cur_ind += 1
            if self.cur_ind == self._num_arch:
                np.random.shuffle(self.indexes)
                self.cur_ind = 0
        return rollouts

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("discrete")]

    # ---- unused ----
    def set_mode(self, mode):
        pass

    def set_device(self, device):
        pass

    def step(self, rollouts, optimizer):
        return 0.

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        return OrderedDict()

    def save(self, path):
        pass

    def load(self, path):
        pass


class PoolControllerPlugin(AwnasPlugin):
    NAME = "pool"
    controller_list = [PoolController]
