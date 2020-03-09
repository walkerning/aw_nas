# -*- coding: utf-8 -*-

import os
import re
import shutil
import subprocess
from collections import OrderedDict

import torch

from aw_nas import utils
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.utils.exception import expect, ConfigException

class TuneEvaluator(BaseEvaluator):
    """
    An evaluator that tune each rollout for several epochs before evaluating performance,
    the tuned candidate network will be saved onto the disk.
    with optional performance predictor incorportated.
    """

    NAME = "tune"

    def __init__( #pylint: disable=dangerous-default-value
            self, dataset, weights_manager, objective, rollout_type="mutation",
            schedule_cfg=None
    ):
        super(TuneEvaluator, self).__init__(
            dataset=None, weights_manager=weights_manager,
            objective=objective, rollout_type=rollout_type, schedule_cfg=schedule_cfg)

        # check rollout type
        expect(self.rollout_type == self.weights_manager.rollout_type,
               "the rollout type of evaluator/weights_manager must match, "
               "check the configuration. ({}/{})".format(
                   self.rollout_type,
                   self.weights_manager.rollout_type), ConfigException)
        # assume gpu
        self.device = str(self.weights_manager.device.index)
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
        return ["image", "sequence"]

    @classmethod
    def supported_rollout_types(cls):
        return ["mutation"]

    def set_device(self, device):
        w_device = device
        if isinstance(device, int):
            device = w_device = str(device)
        elif isinstance(device, str):
            if device.startswith("cuda"):
                device = w_device = re.match(r"cuda:([0-9]+)", device).group(1)
            elif "," in device:
                w_device = device.split(",", 1)[0]
        elif isinstance(device, torch.device):
            assert device.type == "cuda"
            device = w_device = str(device.index)
        self.device = device
        self.weights_manager.set_device(torch.device("cuda:{}".format(w_device)))

    def evaluate_rollouts(self, rollouts, is_training, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        for rollout in rollouts:
            cand_net = self.weights_manager.assemble_candidate(rollout)
            ckpt_path = rollout.model_record.checkpoint_path
            train_dir = utils.makedir(ckpt_path + "-train-dir")

            # dump candidate net checkpoint to "`train_dir`/init.pt"
            init_ckpt_fname = os.path.join(train_dir, "init.pt")
            torch.save(cand_net.state_dict(), init_ckpt_fname)
            save_every = rollout.model_record.config.get("save_every", 5)
            seed = rollout.model_record.config.get("seed", 123)

            # dump config to "`train_dir`/train.yaml"
            c_fname = os.path.join(train_dir, "train.yaml")
            rollout.model_record.save_config(c_fname)

            actual_train_dir = os.path.join(train_dir, "train")
            subprocess.check_call(("awnas train {config} --save-every {save_every} --seed {seed} "
                                   "--gpus {gpus} --load-state-dict {load} "
                                   "--train-dir {train_dir} >/dev/null 2>&1").format(
                                       config=c_fname,
                                       save_every=save_every,
                                       seed=seed,
                                       gpus=self.device,
                                       load=init_ckpt_fname,
                                       train_dir=actual_train_dir
                                   ),
                                  shell=True)
            # parse log to get final performance
            perfs = self._parse_log(os.path.join(actual_train_dir, "train.log"))
            rollout.set_perfs(perfs)

            # copy final model to `ckpt_path`
            final_ckpt_fname = os.path.join(actual_train_dir, "final", "model.pt")
            if not os.path.exists(final_ckpt_fname):
                final_ckpt_fname = os.path.join(actual_train_dir, "final", "model_state.pt")
            shutil.copy(final_ckpt_fname, ckpt_path)

            # TODO: access model record through API is better
            rollout.model_record.finished = True
            rollout.model_record.confidence = 1.
        return rollouts

    def update_evaluator(self, controller):
        pass

    def update_rollouts(self, rollouts):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass
