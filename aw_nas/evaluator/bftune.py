# -*- coding: utf-8 -*-

import os
import re
import glob
import subprocess
from collections import OrderedDict

import yaml
import torch

from aw_nas.common import ConfigTemplate
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.utils.exception import expect, ConfigException

class BFTuneEvaluator(BaseEvaluator):
    """
    An evaluator that continue tuning each rollout for several epochs before evaluating performance,
    initialized from a trainer state.
    """

    NAME = "bf_tune"

    def __init__( #pylint: disable=dangerous-default-value
            self, dataset, weights_manager, objective, rollout_type="mutation",
            template_cfg_file=None, save_every=10, bf_checkpoints=[10, 20, 40, 60, 80],
            schedule_cfg=None
    ):
        # do not need dataset, weights manager
        super(BFTuneEvaluator, self).__init__(
            dataset=None, weights_manager=None,
            objective=objective, rollout_type=rollout_type, schedule_cfg=schedule_cfg)

        expect(template_cfg_file is not None, "Must specified `template_cfg_file` configuration",
               ConfigException)
        self.template_cfg_file = template_cfg_file
        self.save_every = save_every
        self.bf_checkpoints = bf_checkpoints

        with open(template_cfg_file, "r") as cfg_f:
            self.cfg_template = ConfigTemplate(yaml.safe_load(cfg_f))
        self.logger.info("Read the template config from %s", template_cfg_file)

        # assume gpu
        self.device = "0"
        self._perf_names = self.objective.perf_names()
        self.log_pattern = re.compile(
            "valid performances: " + \
            "; ".join(
                ["{}: ([-0-9.]+)".format(n) for n in self._perf_names]))

    def _parse_log(self, log_fname):
        with open(log_fname, "r") as log_f:
            all_perfs = self.log_pattern.findall(log_f.read())
        # for now: just return the last epoch
        last_epoch_perfs = all_perfs[-1]
        if len(self._perf_names) == 1:
            last_epoch_perfs = [last_epoch_perfs]
        last_epoch_perfs = [float(perf) for perf in last_epoch_perfs]
        perfs = OrderedDict(zip(self._perf_names, last_epoch_perfs))
        return perfs

    # ---- APIs ----
    @classmethod
    def supported_data_types(cls):
        """
        Return the supported data types
        """
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["mutation"]

    def set_device(self, device):
        if isinstance(device, int):
            device = str(device)
        elif isinstance(device, str):
            if device.startswith("cuda"):
                device = re.match(r"cuda:([0-9]+)", device).group(1)
        elif isinstance(device, torch.device):
            assert device.type == "cuda"
            device = str(device.index)
        self.device = device

    def evaluate_rollouts(self, rollouts, is_training, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        for rollout in rollouts:
            parent_train_dir = rollout.train_dir
            if not os.path.exists(parent_train_dir):
                os.makedirs(parent_train_dir)
            seed = 123 # TODO

            # parse which stage this rollout is in
            cur_stage = len(rollout.perf.get("reward", []))
            dir_pattern = "{}/train_*/".format(parent_train_dir)
            dirs = glob.glob(dir_pattern)
            if len(dirs) != cur_stage:
                self.logger.warn("#train dir (%d) and #perfs mismatch (%d)", len(dirs), cur_stage)

            # dump config to "`train_dir`/train.yaml"
            c_fname = os.path.join(parent_train_dir, "train_{}.yaml".format(cur_stage))
            cfg = self.cfg_template.create_cfg(rollout.genotype)
            cfg["final_trainer_cfg"]["epochs"] = self.bf_checkpoints[cur_stage]
            with open(c_fname, "w") as cfg_f:
                yaml.dump(cfg, cfg_f)

            # last_state_dir
            if cur_stage > 0:
                last_state_dir = os.path.join(
                    parent_train_dir, "train_{}".format(cur_stage - 1), "final")
                load_str = "--load {}".format(last_state_dir)
            else:
                load_str = ""
            actual_train_dir = os.path.join(parent_train_dir, "train_{}".format(cur_stage))
            subprocess.check_call(("awnas train {config} --save-every {save_every} --seed {seed} "
                                   "--gpus {gpus} {load_str} "
                                   "--train-dir {train_dir} >/dev/null 2>&1").format(
                                       config=c_fname,
                                       save_every=self.save_every,
                                       seed=seed,
                                       gpus=self.device,
                                       load_str=load_str,
                                       train_dir=actual_train_dir
                                   ),
                                  shell=True)
            # parse log to get final performance
            perfs = self._parse_log(os.path.join(actual_train_dir, "train.log"))
            rollout.perf.setdefault("reward", []).append(perfs["acc"])

        return rollouts

    def update_evaluator(self, controller):
        pass

    def update_rollouts(self, rollouts):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass
