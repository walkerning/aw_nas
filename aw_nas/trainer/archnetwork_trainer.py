# -*- coding: utf-8 -*-
"""
Flow for predictor-based trainer.
Batched controller sampling, evaluator evaluation and predictor training.
"""

from __future__ import print_function
from __future__ import division

import os

import numpy as np

from aw_nas import utils
from aw_nas.trainer.base import BaseTrainer
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.trainer.async_trainer import BaseDispatcher
from aw_nas.evaluator import BaseEvaluator

__all__ = ["ArchNetworkTrainer"]

class ArchNetworkTrainer(BaseTrainer):
    """
    ArchNetwork-based NAS searcher.
    The major difference: Batched evaluation results for predictor training.
    """

    NAME = "archnetwork_based"

    SCHEDULABLE_ATTRS = []

    def __init__(self,
                 controller, evaluator, rollout_type="discrete",
                 dispatcher_type="multiprocess", dispatcher_cfg=None,
                 arch_network_evaluator_type="batch_update_comparator",
                 arch_network_evaluator_cfg=None,
                 num_each_iter=[600, 200, 200], controller_use_archnetwork=True,
                 bf_ignore_ratio=[0.2, 0.2, 0.2, 0.2], bf_max_depth=5,
                 log_timeout=60.,
                 schedule_cfg=None):
        super(ArchNetworkTrainer, self).__init__(
            controller, evaluator, rollout_type, schedule_cfg)

        expect(self.rollout_type == self.controller.rollout_type == \
               self.evaluator.rollout_type,
               "the rollout type of trainer/controller/evaluator must match, "
               "check the configuration. ({}/{}/{})".format(
                   self.rollout_type, self.controller.rollout_type,
                   self.evaluator.rollout_type), ConfigException)

        # configurations
        self.num_each_iter = num_each_iter
        self.iterations = len(num_each_iter)
        self.controller_use_archnetwork = controller_use_archnetwork
        self.log_timeout = log_timeout
        self.bf_ignore_ratio = bf_ignore_ratio
        self.bf_max_depth = bf_max_depth
        expect(len(self.bf_ignore_ratio) == self.bf_max_depth - 1)

        d_cls = BaseDispatcher.get_class_(dispatcher_type)
        self.dispatcher = d_cls(**(dispatcher_cfg or {}))

        ane_cls = BaseEvaluator.get_class_(arch_network_evaluator_type)
        self.arch_network_evaluator = ane_cls(search_space=self.controller.search_space,
                                              **(arch_network_evaluator_cfg or {}))

        self.i_iter = -1
        self.save_every = None
        self.ckpt_dir = None

    # ---- APIs ----
    @classmethod
    def supported_rollout_types(cls):
        return ["discrete"]

    def train(self):
        for i_iter in range(self.iterations):
            self.i_iter = i_iter
            if self.controller_use_archnetwork:
                # TODO: add an util for avoiding sampling repeating archs
                rollouts = self.controller.sample(
                    n=self.num_each_iter[i_iter],
                    arch_network_evaluator=self.arch_network_evaluator)
            else:
                rollouts = self.controller.sample(n=self.num_each_iter[i_iter])

            # create checkpoint direcotries
            iter_ckpt_dir = os.path.join(self.ckpt_dir, str(i_iter))
            if not os.path.exists(iter_ckpt_dir):
                # FIXME: maybe remove previous directory is better?
                os.makedirs(iter_ckpt_dir)
            for r in rollouts:
                # `rollout.train_dir` will be used by BFTuneEvaluator
                r.train_dir = os.path.join(iter_ckpt_dir, str(hash(r)))

            cur_rollouts = rollouts
            stage_rollouts = []
            for i_bf in range(self.bf_max_depth):
                self.logger.info("Iter %3d, Checkpoint depth %3d: #Archs to eval: %3d",
                                 i_iter, i_bf, len(cur_rollouts))
                self.dispatcher.start_eval_rollouts(cur_rollouts)
                all_finished_rollouts = []
                while 1:
                    f_rollouts = self.dispatcher.get_finished_rollouts(timeout=self.log_timeout)
                    if not f_rollouts:
                        num_nofinish += 1
                        self.logger.debug("No rollout finished in the past %d seconds",
                                          num_nofinish * self.log_timeout)
                    else:
                        num_nofinish = 0
                        all_finished_rollouts += f_rollouts
                        for r in f_rollouts:
                            self.logger.debug("Rollout %s evaluation finished.", r)
                        if len(all_finished_rollouts) == len(cur_rollouts):
                            break
                self.logger.info("Iter %3d, Checkpoint depth %3d: Finish eval %3d archs",
                                 i_iter, i_bf, len(cur_rollouts))
                if i_bf != self.bf_max_depth - 1:
                    # choose archs that need to continue ``breadth-first'' evaluation
                    stop_rollouts, cur_rollouts = self._get_stop_and_continue(
                        all_finished_rollouts, self.bf_ignore_ratio[i_bf])
                    stage_rollouts.append(stop_rollouts)
                else:
                    stage_rollouts.append(cur_rollouts)
            self.logger.info("Iter %3d: #Archs: %3d\n\t%s", i_iter, len(rollouts),
                             "\n\t".join(["Stage {:3d}: {:3d}".format(
                                 i_bf, len(stage_rollouts[i_bf]))
                                          for i_bf in range(self.bf_max_depth)]))

            # TODO: dump rollouts meta information to file

            # Train predictor using rollouts, maybe merge previous ones?
            concat_rollouts = sum(stage_rollouts, [])
            self.arch_network_evaluator.update_rollouts(concat_rollouts)

            if not self.controller_use_archnetwork:
                self._update_controller_blackblox()

    def _update_controller_blackblox(self):
        # TODO: call `arch_network_evaluator.evaluate_rollouts` and `controller.step` alternatively
        pass

    def _get_stop_and_continue(self, rollouts, ignore_ratio):
        cur_len = len(rollouts[0].perf["reward"])
        assert all(len(r.perf["reward"]) == cur_len for r in rollouts)
        perfs = [r.perf["reward"] for r in rollouts]
        inds = np.argsort(perfs)
        ignore_num = int(len(inds) * ignore_ratio)
        stop_rollouts = [rollouts[ind] for ind in inds[:ignore_num]]
        cont_rollouts = [rollouts[ind] for ind in inds[ignore_num:]]
        return stop_rollouts, cont_rollouts
        
    def setup(self, load=None, save_every=None, train_dir=None, writer=None, load_components=None,
              interleave_report_every=None):
        # TODO: handle load components
        assert train_dir is not None, \
            "You'd better provide a path to save all the checkpoints"

        self.train_dir = train_dir
        self.ckpt_dir = utils.makedir(os.path.join(train_dir, "checkpoints"))
        self.dispatcher.init(self.evaluator, None)
        self.save_every = save_every

    def test(self):
        pass

    def derive(self, n, steps=None):
        pass

    def _save_path(self, name=""):
        if self.train_dir is None:
            return None
        dir_ = utils.makedir(os.path.join(self.train_dir, str(self.i_iter)))
        return os.path.join(dir_, name)

    def _save_all(self):
        if self.train_dir is not None:
            self.controller.save(self._save_path("controller"))
            self.evaluator.save(self._save_path("evaluator"))
            self.arch_network_evaluator.save(self._save_path("arch_network_evaluator"))
            self.logger.info("Iter %3d: Save all checkpoints to directory %s",
                             self.i_iter, self._save_path())

    def save(self, path):
        # TODO
        pass

    def load(self, path):
        # TODO
        pass

    @classmethod
    def get_default_config_str(cls):
        all_str = super(ArchNetworkTrainer, cls).get_default_config_str()
        # Possible dispatcher configs
        all_str += utils.component_sample_config_str("dispatcher", prefix="#   ") + "\n"
        # Possible arch network based evaluator configs
        all_str += utils.component_sample_config_str(
            "evaluator", prefix="#   ",
            filter_funcs=[lambda cls: "batch_update" in cls.NAME],
            cfg_name="arch_network_evaluator")
        return all_str
