# -*- coding: utf-8 -*-
"""
Trainer definition, this is the orchestration of all the components.
"""

from __future__ import print_function
from __future__ import division

from functools import partial
from collections import OrderedDict
import imageio
import os
import yaml

import numpy as np
import torch

from aw_nas import utils
from aw_nas.trainer.base import BaseTrainer
from aw_nas.utils.exception import expect, ConfigException

__all__ = ["SimpleTrainer"]

class SimpleTrainer(BaseTrainer):
    """
    A simple NAS searcher.

    Schedulable Attributes:
        mepa_surrogate_steps:
        mepa_samples:
        controller_steps:
        controller_surrogate_steps:
        controller_samples:
    """

    NAME = "simple"

    SCHEDULABLE_ATTRS = [
        "controller_samples",
        "derive_samples"
    ]

    def __init__(self, #pylint: disable=dangerous-default-value
                 controller, evaluator, rollout_type="discrete",
                 epochs=200, test_every=10,

                 # optimizer and scheduler
                 controller_optimizer={
                     "type": "Adam",
                     "lr": 0.001
                 },
                 controller_scheduler=None,

                 # number of rollout/arch samples
                 controller_samples=1,
                 derive_samples=8,

                 # >1 only work for differentiable rollout now
                 rollout_batch_size=1,

                 # alternative training config
                 evaluator_steps=None,
                 controller_steps=None,
                 controller_train_every=1,
                 controller_train_begin=1,
                 interleave_controller_every=None,

                 schedule_cfg=None):
        """
        Args:
            controller_steps (int): If None, (not explicitly given), assume every epoch consume
                one pass of the controller queue.
            interleave_controller_every (int): Interleave controller update steps every
                `interleave_controller_every` steps. If None, do not interleave, which means
                controller will only be updated after one epoch of mepa update.
        """
        super(SimpleTrainer, self).__init__(controller, evaluator, rollout_type, schedule_cfg)

        expect(self.rollout_type == self.controller.rollout_type == \
               self.evaluator.rollout_type,
               "the rollout type of trainer/controller/evaluator must match, "
               "check the configuration. ({}/{}/{})".format(
                   self.rollout_type, self.controller.rollout_type,
                   self.evaluator.rollout_type), ConfigException)

        # configurations
        self.epochs = epochs
        self.test_every = test_every

        self.controller_samples = controller_samples
        self.derive_samples = derive_samples

        self.rollout_batch_size = rollout_batch_size

        self.evaluator_steps = evaluator_steps
        self.controller_steps = controller_steps
        self.controller_train_every = controller_train_every
        self.controller_train_begin = controller_train_begin
        self.interleave_controller_every = interleave_controller_every

        # prepare `self.controller_steps`
        suggested = self.evaluator.suggested_controller_steps_per_epoch()
        if self.controller_steps is None:
            # if `controller_steps` not specified, use the suggested value by calling
            # `evaluator.suggested_controller_steps_per_epoch`
            expect(suggested is not None,
                   "Cannot infer `controller_steps`! Neigher `controller_steps` is given in "
                   "configuration, nor the evaluator return"
                   " a suggested `controller_steps`.", ConfigException)
            self.controller_steps = suggested
        else: # `controller_steps` is provided, check if it matches with the suggested value
            if suggested is not None and not suggested == self.controller_steps:
                self.logger.warning("The suggested `controller_steps` (%3d) from "
                                    "`evaluator.suggested_controller_steps_per_epoch()` differs "
                                    "from the config setting (%3d).",
                                    suggested, self.controller_steps)

        # prepare `self.evaluator_steps`
        expect(self.interleave_controller_every is None or (
            self.evaluator_steps is None or self.evaluator_steps == \
            self.controller_steps * self.interleave_controller_every),
               "`controller_steps` must not be given or must match with "
               "`evaluator_steps/interleave_controller_every`in interleave mode", ConfigException)

        suggested = self.evaluator.suggested_evaluator_steps_per_epoch()
        if self.evaluator_steps is None:
            if self.interleave_controller_every is None:
                # if `evaluator_steps` is not explicitly given, and not in interleave mode,
                # use the suggested value from `evaluator.suggested_evaluator_steps_per_epoch()`
                self.evaluator_steps = suggested
            else:
                # in interleave mode
                self.evaluator_steps = self.controller_steps * self.interleave_controller_every
        elif self.interleave_controller_every is None:
            # `evaluator_steps` is provided, check if it matches with the suggested value
            if suggested is not None and not suggested == self.evaluator_steps:
                self.logger.warning("The suggested `evaluator_steps` (%3d) from "
                                    "`evaluator.suggested_evaluator_steps_per_epoch()` differs "
                                    "from the config setting (%3d).",
                                    suggested, self.evaluator_steps)

        # init controller optimizer and scheduler
        if controller_optimizer:
            self.controller_optimizer = utils.init_optimizer(self.controller.parameters(),
                                                             controller_optimizer)
            self.controller_scheduler = utils.init_scheduler(self.controller_optimizer,
                                                             controller_scheduler)
        else:
            self.controller_optimizer = None
            self.controller_scheduler = None

        # states and other help attributes
        self.last_epoch = 0
        self.epoch = 0

    def _evaluator_update(self, steps, finished_e_steps, finished_c_steps):
        self.controller.set_mode("eval")
        eva_stat_meters = utils.OrderedStats()

        for i_eva in range(1, steps+1): # mepa stands for meta param
            print("\reva step {}/{} ; controller step {}/{}"\
                  .format(finished_e_steps+i_eva, self.evaluator_steps,
                          finished_c_steps, self.controller_steps),
                  end="")
            e_stats = self.evaluator.update_evaluator(self.controller)
            eva_stat_meters.update(e_stats)
        return eva_stat_meters.avgs()

    def _controller_update(self, steps, finished_e_steps, finished_c_steps):
        controller_loss_meter = utils.AverageMeter()
        controller_stat_meters = utils.OrderedStats()
        rollout_stat_meters = utils.OrderedStats()

        self.controller.set_mode("train")
        for i_cont in range(1, steps+1):
            print("\reva step {}/{} ; controller step {}/{}"\
                  .format(finished_e_steps, self.evaluator_steps,
                          finished_c_steps+i_cont, self.controller_steps),
                  end="")

            rollouts = self.controller.sample(self.controller_samples, self.rollout_batch_size)
            if self.rollout_type == "differentiable":
                self.controller.zero_grad()

            step_loss = {"_": 0.}
            rollouts = self.evaluator.evaluate_rollouts(rollouts, is_training=True,
                                                        callback=partial(
                                                            self._backward_rollout_to_controller,
                                                            step_loss=step_loss))
            self.evaluator.update_rollouts(rollouts)

            if self.rollout_type == "differentiable":
                # differntiable rollout (controller is optimized using differentiable relaxation)
                # adjust lr and call step_current_gradients
                # (update using the accumulated gradients)
                controller_loss = step_loss["_"] / self.controller_samples
                if self.controller_samples != 1:
                    # adjust the lr to keep the effective learning rate unchanged
                    lr_bak = self.controller_optimizer.param_groups[0]["lr"]
                    self.controller_optimizer.param_groups[0]["lr"] \
                        = lr_bak / self.controller_samples
                self.controller.step_current_gradient(self.controller_optimizer)
                if self.controller_samples != 1:
                    self.controller_optimizer.param_groups[0]["lr"] = lr_bak
            else: # other rollout types
                controller_loss = self.controller.step(
                    rollouts, self.controller_optimizer, perf_name="reward")

            # update meters
            controller_loss_meter.update(controller_loss)
            controller_stats = self.controller.summary(rollouts, log=False)
            if controller_stats is not None:
                controller_stat_meters.update(controller_stats)

            r_stats = OrderedDict()
            for n in rollouts[0].perf:
                r_stats[n] = np.mean([r.perf[n] for r in rollouts])
            rollout_stat_meters.update(r_stats)

        print("\r", end="")

        return controller_loss, rollout_stat_meters.avgs(), controller_stat_meters.avgs()

    def _backward_rollout_to_controller(self, rollout, step_loss):
        if self.rollout_type == "differentiable":
            # backward
            _loss = self.controller.gradient(rollout.get_perf(name="reward"),
                                             return_grads=False,
                                             zero_grads=False)
            step_loss["_"] += _loss

    # ---- APIs ----
    @classmethod
    def supported_rollout_types(cls):
        return ["discrete", "differentiable", "compare", "nasbench-101", "nasbench-201"]

    def train(self): #pylint: disable=too-many-branches
        assert self.is_setup, "Must call `trainer.setup` method before calling `trainer.train`."

        if self.interleave_controller_every is not None:
            inter_steps = self.controller_steps
            evaluator_steps = self.interleave_controller_every
            controller_steps = 1
        else:
            inter_steps = 1
            evaluator_steps = self.evaluator_steps
            controller_steps = self.controller_steps

        for epoch in range(self.last_epoch+1, self.epochs+1):
            c_loss_meter = utils.AverageMeter()
            rollout_stat_meters = utils.OrderedStats() # rollout performance stats from evaluator
            c_stat_meters = utils.OrderedStats() # other stats from controller
            eva_stat_meters = utils.OrderedStats() # other stats from `evaluator.update_evaluator`

            self.epoch = epoch # this is redundant as Component.on_epoch_start also set this

            # call `on_epoch_start` of sub-components
            # also schedule values and optimizer learning rates
            self.on_epoch_start(epoch)

            finished_e_steps = 0
            finished_c_steps = 0
            for i_inter in range(1, inter_steps+1): # interleave mepa/controller training
                # meta parameter training
                if evaluator_steps > 0:
                    e_stats = self._evaluator_update(evaluator_steps, finished_e_steps,
                                                     finished_c_steps)
                    eva_stat_meters.update(e_stats)
                    finished_e_steps += evaluator_steps

                if epoch >= self.controller_train_begin and \
                   epoch % self.controller_train_every == 0 and controller_steps > 0:
                    # controller training
                    c_loss, rollout_stats, c_stats \
                        = self._controller_update(controller_steps,
                                                  finished_e_steps, finished_c_steps)
                    # update meters
                    if c_loss is not None:
                        c_loss_meter.update(c_loss)
                    if rollout_stats is not None:
                        rollout_stat_meters.update(rollout_stats)
                    if c_stats is not None:
                        c_stat_meters.update(c_stats)

                    finished_c_steps += controller_steps

                if self.interleave_report_every and i_inter % self.interleave_report_every == 0:
                    # log for every `interleave_report_every` interleaving steps
                    self.logger.info("(inter step %3d): "
                                     "evaluator (%3d/%3d) %s ; "
                                     "controller (%3d/%3d) %s",
                                     i_inter, finished_e_steps, self.evaluator_steps,
                                     "; ".join(
                                         ["{}: {:.3f}".format(n, v) \
                                          for n, v in eva_stat_meters.avgs().items()]),
                                     finished_c_steps, self.controller_steps,
                                     "" if not rollout_stat_meters else "; ".join(
                                         ["{}: {:.3f}".format(n, v) \
                                          for n, v in rollout_stat_meters.avgs().items()]))

            # log infomations of this epoch
            if eva_stat_meters:
                self.logger.info("Epoch %3d: [evaluator update] %s", epoch,
                                 "; ".join(["{}: {:.3f}".format(n, v) \
                                            for n, v in eva_stat_meters.avgs().items()]))
            if rollout_stat_meters:
                self.logger.info("Epoch %3d: [controller update] controller loss: %.3f ; "
                                 "rollout performance: %s", epoch, c_loss_meter.avg,
                                 "; ".join(["{}: {:.3f}".format(n, v) \
                                            for n, v in rollout_stat_meters.avgs().items()]))
            if c_stat_meters:
                self.logger.info("[controller stats] %s", \
                                 "; ".join(["{}: {:.3f}".format(n, v) \
                                            for n, v in c_stat_meters.avgs().items()]))

            # maybe write tensorboard info
            if not self.writer.is_none():
                if eva_stat_meters:
                    for n, meter in eva_stat_meters.items():
                        self.writer.add_scalar("evaluator_update/{}".format(n.replace(" ", "-")),
                                               meter.avg, epoch)
                if rollout_stat_meters:
                    for n, meter in rollout_stat_meters.items():
                        self.writer.add_scalar("controller_update/{}".format(n.replace(" ", "-")),
                                               meter.avg, epoch)
                if c_stat_meters:
                    for n, meter in c_stat_meters.items():
                        self.writer.add_scalar("controller_stats/{}".format(n.replace(" ", "-")),
                                               meter.avg, epoch)
                if not c_loss_meter.is_empty():
                    self.writer.add_scalar("controller_loss", c_loss_meter.avg, epoch)

            # maybe save checkpoints
            self.maybe_save()

            # maybe derive archs and test
            if self.test_every and self.epoch % self.test_every == 0:
                self.test()

            self.on_epoch_end(epoch) # call `on_epoch_end` of sub-components

        # `final_save` pickle dump the weights_manager and controller directly,
        # instead of the state dict
        self.final_save()

    def test(self):
        """
        Derive and test, plot the best arch among the arch samples.
        """
        rollouts = self.derive(n=self.derive_samples)

        rewards = [r.get_perf("reward") for r in rollouts]
        mean_rew = np.mean(rewards)
        idx = np.argmax(rewards)
        # other_perfs = {n: [r.perf[n] for r in rollouts] for n in rollouts[0].perf}
        other_perfs = {n: [r.perf.get(n, None) for r in rollouts] for n in rollouts[0].perf}

        save_path = self._save_path("rollout/cell")
        if save_path is not None:
            # NOTE: If `train_dir` is None, the image will not be saved to tensorboard too
            fnames = rollouts[idx].plot_arch(save_path, label="epoch {}".format(self.epoch))
            if not self.writer.is_none() and fnames is not None:
                for cg_n, fname in fnames:
                    image = imageio.imread(fname)
                    self.writer.add_image("genotypes/{}".format(cg_n), image, self.epoch,
                                          dataformats="HWC")

        self.logger.info("TEST Epoch %3d: Among %d sampled archs: "
                         "BEST (in reward): %.5f (mean: %.5f); Performance: %s",
                         self.epoch, self.derive_samples, rewards[idx], mean_rew,
                         "; ".join(["{}: {} (mean {:.5f})".format(
                             n, "{:.5f}".format(other_perfs[n][idx]) if other_perfs[n][idx] is not None else None,
                             np.mean([perf for perf in other_perfs[n] if perf is not None])) for n in rollouts[0].perf]))
        self.logger.info("Saved this arch to %s.\nGenotype: %s",
                         save_path, rollouts[idx].genotype)
        self.controller.summary(rollouts, log=True, log_prefix="Rollouts Info: ", step=self.epoch)
        return rollouts

    def derive(self, n, steps=None, out_file=None):
        # # some scheduled value will be used in test too, e.g. surrogate_lr, gumbel temperature...
        # called in `load` method already
        # self.on_epoch_start(self.epoch)
        with self.controller.begin_mode("eval"):
            rollouts = self.controller.sample(n)
            save_dict = {}
            if out_file:
                if os.path.exists(out_file):
                    fin = open(out_file)
                    lines = fin.readlines()
                    reward = None
                    arch = None
                    for i in range(len(lines)):
                        if i % 3 == 0:
                            reward = float(lines[i][lines[i].find("Reward")+7:lines[i].find(")")])
                        elif i % 3 == 1:
                            arch = lines[i].strip()[3:-1]
                        else:
                            save_dict[arch] = reward
                    fin.close()
                    os.system("cp {} {}".format(out_file, out_file+".tmp"))
                fo = open(out_file, 'w')
            for i_sample in range(len(rollouts)):
                if out_file and rollouts[i_sample].genotype in save_dict.keys():
                    fo.write("# ---- Arch {} (Reward {}) ----\n".format(i_sample, save_dict[rollouts[i_sample].genotype]))
                    yaml.safe_dump([str(rollouts[i_sample].genotype)], fo)
                    fo.write("\n")
                    continue
                rollouts[i_sample] = self.evaluator.evaluate_rollouts([rollouts[i_sample]],
                                                                      is_training=False,
                                                                      eval_batches=steps)[0]
                print("Finish test {}/{}\r".format(i_sample+1, n), end="")
                if out_file:
                    fo.write("# ---- Arch {} (Reward {}) ----\n".format(i_sample, rollouts[i_sample].get_perf()))
                    yaml.safe_dump([str(rollouts[i_sample].genotype)], fo)
                    fo.write("\n")
            if out_file:
                fo.close()
        return rollouts

    def save(self, path):
        optimizer_state = self.controller_optimizer.state_dict()\
            if self.controller_optimizer is not None else None
        scheduler_state = self.controller_scheduler.state_dict()\
            if self.controller_scheduler is not None else None
        state_dict = {
            "epoch": self.epoch,
            "controller_optimizer": optimizer_state,
            "controller_scheduler": scheduler_state
        }
        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.last_epoch = self.epoch = checkpoint["epoch"]
        if self.controller_optimizer is not None:
            self.controller_optimizer.load_state_dict(checkpoint["controller_optimizer"])
        if self.controller_scheduler is not None:
            self.controller_scheduler.load_state_dict(checkpoint["controller_scheduler"])
        self.on_epoch_start(self.last_epoch)

    def on_epoch_start(self, epoch):
        super(SimpleTrainer, self).on_epoch_start(epoch)
        if self.controller_scheduler is not None:
            self.controller_scheduler.step(epoch-1)
            self.logger.info("Epoch %3d: controller LR: %.5f", epoch,
                             self.controller_scheduler.get_lr()[0])
