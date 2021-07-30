# -*- coding: utf-8 -*-
"""
Trainer definition, this is the orchestration of all the components.
"""

from __future__ import print_function
from __future__ import division

import os
import time
import contextlib
import subprocess
from functools import partial
from collections import OrderedDict

import imageio
import numpy as np
import torch

from aw_nas import utils
from aw_nas.utils.common_utils import _dump_with_perf, _parse_derive_file
from aw_nas.trainer.base import BaseTrainer
from aw_nas.utils.exception import expect, ConfigException

__all__ = ["MetaTrainer"]


class MetaTrainer(BaseTrainer):
    """
    A meta learning NAS searcher.

    Schedulable Attributes:
        mepa_surrogate_steps:
        mepa_samples:
        controller_steps:
        controller_surrogate_steps:
        controller_samples:
    """

    NAME = "meta"

    SCHEDULABLE_ATTRS = ["controller_samples", "derive_samples"]

    def __init__(
        self,  # pylint: disable=dangerous-default-value
        controller,
        evaluator,
        rollout_type="discrete",
        is_differentiable=False,
        epochs=200,
        test_every=10,
        # optimizer and scheduler
        controller_base_optimizer={
            "type": "SGD",
            "lr": 0.01,
        },
        controller_base_scheduler=None,
        controller_meta_optimizer={"type": "Adam", "lr": 0.001},
        controller_meta_scheduler=None,
        # meta learning config
        train_base_style="together",
        train_base_steps=5,
        test_base_style="together",
        test_base_steps=10,
        meta_batch_size=4,
        meta_update_method="fo_maml",
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
        schedule_cfg=None,
    ):
        """
        Args:
            controller_steps (int): If None, (not explicitly given), assume every epoch consume
                one pass of the controller queue.
            interleave_controller_every (int): Interleave controller update steps every
                `interleave_controller_every` steps. If None, do not interleave, which means
                controller will only be updated after one epoch of mepa update.
            train/test_base_style (str): for base update, `style` refers to
                which part of the trainer is updated:
                evaluator: evaluator only
                together: both evaluator and controller
            meta_update_method (str): for meta update: `method` refers to
                how meta gradient is calculates:
                fo_maml: first-order MAML
                reptile: REPTILE
        """
        super(MetaTrainer, self).__init__(
            controller, evaluator, rollout_type, schedule_cfg
        )

        expect(
            self.rollout_type
            == self.controller.rollout_type
            == self.evaluator.rollout_type,
            "the rollout type of trainer/controller/evaluator must match, "
            "check the configuration. ({}/{}/{})".format(
                self.rollout_type,
                self.controller.rollout_type,
                self.evaluator.rollout_type,
            ),
            ConfigException,
        )

        # A fix for backward compatability of old configuration files
        if rollout_type == "differentiable":
            is_differentiable = True

        if "differentiable" in rollout_type and not is_differentiable:
            is_differentiable = True
            self.logger.warn(
                'The `rollout_type` "%s" contains "differentiable", '
                "however, `is_differentiable` is set to False. "
                "Maybe the configuration is mismatched? "
                "Automatically set `is_differentiable` to True",
                rollout_type,
            )
        # configurations
        self.epochs = epochs
        self.test_every = test_every
        self.is_differentiable = is_differentiable

        self.controller_samples = controller_samples
        self.derive_samples = derive_samples

        self.rollout_batch_size = rollout_batch_size

        self.evaluator_steps = evaluator_steps
        self.controller_steps = controller_steps
        self.derive_steps = None
        self.controller_train_every = controller_train_every
        self.controller_train_begin = controller_train_begin
        self.interleave_controller_every = interleave_controller_every

        # prepare `self.controller_steps`
        suggested = self.evaluator.suggested_controller_steps_per_epoch()
        if self.controller_steps is None:
            # if `controller_steps` not specified, use the suggested value by calling
            # `evaluator.suggested_controller_steps_per_epoch`
            expect(
                suggested is not None,
                "Cannot infer `controller_steps`! Neither `controller_steps` is given in "
                "configuration, nor the evaluator return"
                " a suggested `controller_steps`.",
                ConfigException,
            )
            self.controller_steps = suggested
        else:  # `controller_steps` is provided, check if it matches with the suggested value
            if suggested is not None and not suggested == self.controller_steps:
                self.logger.warning(
                    "The suggested `controller_steps` (%3d) from "
                    "`evaluator.suggested_controller_steps_per_epoch()` differs "
                    "from the config setting (%3d).",
                    suggested,
                    self.controller_steps,
                )

        # prepare `self.evaluator_steps`
        expect(
            self.interleave_controller_every is None
            or (
                self.evaluator_steps is None
                or self.evaluator_steps
                == self.controller_steps * self.interleave_controller_every
            ),
            "`controller_steps` must not be given or must match with "
            "`evaluator_steps/interleave_controller_every`in interleave mode",
            ConfigException,
        )

        suggested = self.evaluator.suggested_evaluator_steps_per_epoch()
        if self.evaluator_steps is None:
            if self.interleave_controller_every is None:
                # if `evaluator_steps` is not explicitly given, and not in interleave mode,
                # use the suggested value from `evaluator.suggested_evaluator_steps_per_epoch()`
                self.evaluator_steps = suggested
            else:
                # in interleave mode
                self.evaluator_steps = (
                    self.controller_steps * self.interleave_controller_every
                )
        elif self.interleave_controller_every is None:
            # `evaluator_steps` is provided, check if it matches with the suggested value
            if suggested is not None and not suggested == self.evaluator_steps:
                self.logger.warning(
                    "The suggested `evaluator_steps` (%3d) from "
                    "`evaluator.suggested_evaluator_steps_per_epoch()` differs "
                    "from the config setting (%3d).",
                    suggested,
                    self.evaluator_steps,
                )

        suggested = self.evaluator.suggested_derive_steps_per_epoch()
        if self.derive_steps is None:
            self.derive_steps = suggested
        else:
            # `derive_steps` is provided, check if it matches with the suggested value
            if suggested is not None and not suggested == self.derive_steps:
                self.logger.warning(
                    "The suggested `derive_steps` (%3d) from "
                    "`evaluator.suggested_derive_steps_per_epoch()` differs "
                    "from the config setting (%3d).",
                    suggested,
                    self.derive_steps,
                )

        # init controller optimizer and scheduler
        self.controller_base_optimizer = None
        self.controller_base_scheduler = None
        self.controller_meta_optimizer = None
        self.controller_meta_scheduler = None
        if isinstance(self.controller, torch.nn.Module):
            # TODO: layer2 controller should diff param group, not like this
            self.controller_base_optimizer = utils.init_optimizer(
                self.controller.base_update_parameters(), controller_base_optimizer
            )
            self.controller_base_scheduler = utils.init_scheduler(
                self.controller_base_optimizer, controller_base_scheduler
            )
            self.controller_meta_optimizer = utils.init_optimizer(
                self.controller.parameters(), controller_meta_optimizer
            )
            self.controller_meta_scheduler = utils.init_scheduler(
                self.controller_base_optimizer, controller_meta_scheduler
            )

        self.train_base_style = train_base_style
        self.train_base_steps = train_base_steps
        self.test_base_style = test_base_style
        self.test_base_steps = test_base_steps
        self.meta_batch_size = meta_batch_size
        self.meta_update_method = meta_update_method

        if train_base_style not in ["together", "evaluator"]:
            raise Exception(
                f"Unrecognized base update style `{train_base_style}` for training."
            )
        if test_base_style not in ["together", "evaluator"]:
            raise Exception(
                f"Unrecognized base update style `{test_base_style}` for test."
            )

        if self.meta_update_method == "fo_maml":
            self.backward_query = True
        elif self.meta_update_method == "reptile":
            self.backward_query = False
        else:
            raise Exception(f"Unrecognized meta update method `{meta_update_method}`.")

        # states and other help attributes
        self.last_epoch = 0
        self.epoch = 0

    def _evaluator_update(self, steps, task=None, training=True):
        self.controller.set_mode("eval")
        eva_stat_meters = utils.OrderedStats()

        for i_eva in range(1, steps + 1):  # mepa stands for meta param
            e_stats = self.evaluator.update_evaluator(self.controller)
            eva_stat_meters.update(e_stats)
            print(
                "\reva step {}/{} ; controller step {}/{}; {}".format(
                    finished_e_steps + i_eva,
                    self.evaluator_steps,
                    finished_c_steps,
                    self.controller_steps,
                    ";".join([" %.3f" % v for k, v in eva_stat_meters.avgs().items()]),
                ),
                end="",
            )
        return eva_stat_meters.avgs()

    def _controller_update(self):
        pass

    def _together_update(self, steps, is_training=True, support_set=None):
        controller_loss_meter = utils.AverageMeter()
        controller_stat_meters = utils.OrderedStats()
        rollout_stat_meters = utils.OrderedStats()

        self.controller.set_mode("train")
        for i_base in range(1, steps + 1):
            if i_base == 1 or self.is_differentiable:
                # for non-differentiable rollouts, only sample new rollouts at first step
                rollouts = self.controller.sample(
                    self.controller_samples, self.rollout_batch_size, support_set
                )

            self.evaluator.zero_grad()
            if self.is_differentiable:
                self.controller.zero_grad()

            step_loss = {"_": 0.0}
            rollouts = self.evaluator.evaluate_rollouts(
                rollouts,
                query=False,
                callback=partial(
                    self._backward_rollout_to_controller, step_loss=step_loss
                ),
            )
            self.evaluator.update_rollouts(rollouts)

            self.evaluator.update_weights_manager()
            if self.is_differentiable:
                # differntiable rollout (controller is optimized using differentiable relaxation)
                # adjust lr and call step_current_gradients
                # (update using the accumulated gradients)
                controller_loss = step_loss["_"] / self.controller_samples
                if self.controller_samples != 1:
                    # adjust the lr to keep the effective learning rate unchanged
                    lr_bak = self.controller_base_optimizer.param_groups[0]["lr"]
                    self.controller_base_optimizer.param_groups[0]["lr"] = (
                        lr_bak / self.controller_samples
                    )
                self.controller.step_current_gradient(self.controller_base_optimizer)
                if self.controller_samples != 1:
                    self.controller_base_optimizer.param_groups[0]["lr"] = lr_bak
            else:  # other rollout types
                controller_loss = self.controller.step(
                    rollouts, self.controller_base_optimizer, perf_name="reward"
                )

        # eval on query set
        if self.is_differentiable:
            rollouts = self.controller.sample(
                self.controller_samples, self.rollout_batch_size, support_set
            )

        self.evaluator.zero_grad()
        if self.is_differentiable:
            self.controller.zero_grad()

        step_loss = {"_": 0.0}
        rollouts = self.evaluator.evaluate_rollouts(
            rollouts,
            query=True,
            callback=partial(self._backward_rollout_to_controller, step_loss=step_loss)
            if is_training and self.backward_query
            else None,
        )
        self.evaluator.update_rollouts(rollouts)

        if self.is_differentiable:
            controller_loss = step_loss["_"] / self.controller_samples
        else:  # other rollout types
            controller_loss = self.controller.step(
                rollouts, self.controller_base_optimizer, perf_name="reward"
            )

        # update meters
        controller_loss_meter.update(controller_loss)
        controller_stats = self.controller.summary(rollouts, log=False)
        if controller_stats is not None:
            controller_stat_meters.update(controller_stats)

        r_stats = OrderedDict()
        for n in rollouts[0].perf:
            r_stats[n] = np.mean([r.perf[n] for r in rollouts])
        rollout_stat_meters.update(r_stats)

        return (
            controller_loss,
            rollout_stat_meters.avgs(),
            controller_stat_meters.avgs(),
        )

    def _backward_rollout_to_controller(self, rollout, step_loss):
        if self.is_differentiable:
            # backward
            _loss = self.controller.gradient(
                rollout.get_perf(name="reward"), return_grads=False, zero_grads=False
            )

            step_loss["_"] += _loss

    def _together_update_epoch(self, is_training=True):
        if is_training:
            base_steps = self.train_base_steps
            meta_steps = self.evaluator_steps // self.meta_batch_size
            meta_batch_size = self.meta_batch_size
        else:
            base_steps = self.test_base_steps
            meta_steps = self.derive_steps
            meta_batch_size = 1

        c_loss_meter = utils.AverageMeter()
        rollout_stat_meters = utils.OrderedStats()
        c_stat_meters = utils.OrderedStats()
        eva_stat_meters = utils.OrderedStats()
        update_time_meter = utils.AverageMeter()

        start = time.time()
        for i_meta in range(1, meta_steps + 1):
            self.controller.meta_clone(include_buffers=not is_training)
            self.evaluator.meta_clone(include_buffers=not is_training)
            for i_batch in range(1, meta_batch_size + 1):
                # sample meta batch
                support_set = self.evaluator.sample_meta_batch(is_training)
                # perform base update
                c_loss, rollout_stats, c_stats = self._together_update(
                    base_steps, is_training, support_set
                )

                # update meters
                if c_loss is not None:
                    c_loss_meter.update(c_loss)
                if rollout_stats is not None:
                    rollout_stat_meters.update(rollout_stats)
                    query_perf = "; ".join(
                        [f"{k}: {v:.3f}" for k, v in rollout_stats.items()]
                    )
                    print(
                        (
                            f"\rmeta step {i_meta:4d}/{meta_steps:4d}, "
                            f"meta batch {i_batch}/{meta_batch_size}, "
                            f"query perf {query_perf}"
                        ),
                        end="",
                    )
                if c_stats is not None:
                    c_stat_meters.update(c_stats)

                # accumulate meta gradient and recover
                if is_training:
                    self.controller.meta_gradient(
                        self.meta_update_method,
                        self.controller_base_optimizer.param_groups[0]["lr"]
                        if self.controller_base_optimizer is not None
                        else None,
                    )
                    self.evaluator.meta_gradient(self.meta_update_method)
                self.controller.meta_recover(include_buffers=not is_training)
                self.evaluator.meta_recover(include_buffers=not is_training)

            # perform meta update
            if is_training:
                self.controller.meta_update(self.controller_meta_optimizer)
                self.evaluator.meta_update()

            end = time.time()
            update_time = end - start
            update_time_meter.update(update_time)
            start = end

            if (
                self.interleave_report_every
                and i_meta % self.interleave_report_every == 0
            ):
                print("\r", end="")
                self.logger.info(
                    (
                        f"(together meta update {i_meta:4d}/{meta_steps:4d}) "
                        f"time {update_time:.3f}, "
                        f"query perf {query_perf}"
                    )
                )

        print("\r", end="")
        return (
            eva_stat_meters,
            rollout_stat_meters,
            c_stat_meters,
            c_loss_meter,
            update_time_meter,
        )

    # ---- APIs ----
    @classmethod
    def supported_rollout_types(cls):
        return [
            "discrete",
            "differentiable",
            "compare",
            "nasbench-101",
            "nasbench-201",
            "ofa",
        ]

    def train(self):  # pylint: disable=too-many-branches
        assert (
            self.is_setup
        ), "Must call `trainer.setup` method before calling `trainer.train`."

        for epoch in range(self.last_epoch + 1, self.epochs + 1):
            self.on_epoch_start(epoch)

            if self.train_base_style == "together":
                (
                    eva_stat_meters,
                    rollout_stat_meters,
                    c_stat_meters,
                    c_loss_meter,
                    update_time_meter,
                ) = self._together_update_epoch(is_training=True)

            # log infomations of this epoch
            if eva_stat_meters:
                self.logger.info(
                    "Epoch %3d: [evaluator update] %s",
                    epoch,
                    "; ".join(
                        [
                            "{}: {:.3f}".format(n, v)
                            for n, v in eva_stat_meters.avgs().items()
                        ]
                    ),
                )
            if rollout_stat_meters:
                self.logger.info(
                    "Epoch %3d: [controller update] controller loss: %.3f ; "
                    "rollout performance: %s",
                    epoch,
                    c_loss_meter.avg,
                    "; ".join(
                        [
                            "{}: {:.3f}".format(n, v)
                            for n, v in rollout_stat_meters.avgs().items()
                        ]
                    ),
                )
            if c_stat_meters:
                self.logger.info(
                    "[controller stats] %s",
                    "; ".join(
                        [
                            "{}: {:.3f}".format(n, v)
                            for n, v in c_stat_meters.avgs().items()
                        ]
                    ),
                )
            if update_time_meter:
                self.logger.info(
                    "Epoch %3d [update time] %.3f", epoch, update_time_meter.avg
                )

            # maybe write tensorboard info
            if not self.writer.is_none():
                if eva_stat_meters:
                    for n, meter in eva_stat_meters.items():
                        self.writer.add_scalar(
                            "evaluator_update/{}".format(n.replace(" ", "-")),
                            meter.avg,
                            epoch,
                        )
                if rollout_stat_meters:
                    for n, meter in rollout_stat_meters.items():
                        self.writer.add_scalar(
                            "controller_update/{}".format(n.replace(" ", "-")),
                            meter.avg,
                            epoch,
                        )
                if c_stat_meters:
                    for n, meter in c_stat_meters.items():
                        self.writer.add_scalar(
                            "controller_stats/{}".format(n.replace(" ", "-")),
                            meter.avg,
                            epoch,
                        )
                if not c_loss_meter.is_empty():
                    self.writer.add_scalar("controller_loss", c_loss_meter.avg, epoch)

            # maybe save checkpoints
            self.maybe_save()

            # maybe meta test
            if self.test_every and self.epoch % self.test_every == 0:
                self.meta_test(epoch)

            self.on_epoch_end(epoch)  # call `on_epoch_end` of sub-components

        # `final_save` pickle dump the weights_manager and controller directly,
        # instead of the state dict
        self.final_save()

    def meta_test(self, epoch):
        if self.train_base_style == "together":
            (
                eva_stat_meters,
                rollout_stat_meters,
                c_stat_meters,
                c_loss_meter,
                update_time_meter,
            ) = self._together_update_epoch(is_training=False)

        # log infomations of this epoch
        if eva_stat_meters:
            self.logger.info(
                "META TEST Epoch %3d: [evaluator update] %s",
                epoch,
                "; ".join(
                    [
                        "{}: {:.3f}".format(n, v)
                        for n, v in eva_stat_meters.avgs().items()
                    ]
                ),
            )
        if rollout_stat_meters:
            self.logger.info(
                "META TEST Epoch %3d: [controller update] controller loss: %.3f ; "
                "rollout performance: %s",
                epoch,
                c_loss_meter.avg,
                "; ".join(
                    [
                        "{}: {:.3f}".format(n, v)
                        for n, v in rollout_stat_meters.avgs().items()
                    ]
                ),
            )
        if c_stat_meters:
            self.logger.info(
                "[controller stats] %s",
                "; ".join(
                    ["{}: {:.3f}".format(n, v) for n, v in c_stat_meters.avgs().items()]
                ),
            )
        if update_time_meter:
            self.logger.info(
                "META TEST Epoch %3d [update time] %.3f", epoch, update_time_meter.avg
            )

    def test(self):
        """
        Derive and test, plot the best arch among the arch samples.
        """
        rollouts = self.derive(n=self.derive_samples)

        rewards = [r.get_perf("reward") or 0.0 for r in rollouts]
        mean_rew = np.mean(rewards)
        idx = np.argmax(rewards)
        # other_perfs = {n: [r.perf[n] for r in rollouts] for n in rollouts[0].perf}
        other_perfs = {
            n: [r.perf.get(n, 0.0) for r in rollouts] for n in rollouts[0].perf
        }

        save_path = self._save_path("rollout/cell")
        if save_path is not None:
            # NOTE: If `train_dir` is None, the image will not be saved to tensorboard too
            fnames = rollouts[idx].plot_arch(
                save_path, label="epoch {}".format(self.epoch)
            )
            if not self.writer.is_none() and fnames is not None:
                for cg_n, fname in fnames:
                    try:
                        image = imageio.imread(fname)
                        self.writer.add_image(
                            "genotypes/{}".format(cg_n),
                            image,
                            self.epoch,
                            dataformats="HWC",
                        )
                    except:
                        pass

        self.logger.info(
            "TEST Epoch %3d: Among %d sampled archs: "
            "BEST (in reward): %.5f (mean: %.5f); Performance: %s",
            self.epoch,
            self.derive_samples,
            rewards[idx],
            mean_rew,
            "; ".join(
                [
                    "{}: {} (mean {:.5f})".format(
                        n,
                        "{:.5f}".format(other_perfs[n][idx])
                        if other_perfs[n][idx] is not None
                        else None,
                        np.mean([perf for perf in other_perfs[n] if perf is not None]),
                    )
                    for n in rollouts[0].perf
                ]
            ),
        )
        self.logger.info(
            "Saved this arch to %s.\nGenotype: %s", save_path, rollouts[idx].genotype
        )
        self.controller.summary(
            rollouts, log=True, log_prefix="Rollouts Info: ", step=self.epoch
        )
        return rollouts

    @contextlib.contextmanager
    def _open_derive_out_file(self, derive_out_file):
        """
        On entering: open the file named `derive_out_file` if it is not None and exists,
                     and yield the derived results read from the file with the file (write mode).
        On exiting: close the file.
        """
        if derive_out_file is None:
            yield None, {}
        else:
            if os.path.exists(derive_out_file):
                # load the saved derived results from out_file
                self.logger.info(
                    "Backup the original file {dfile} into {dfile}.tmp".format(
                        dfile=derive_out_file
                    )
                )
                subprocess.check_call(
                    "cp {} {}".format(derive_out_file, derive_out_file + ".tmp"),
                    shell=True,
                )
                with open(derive_out_file) as r_f:
                    save_dict = _parse_derive_file(r_f)
                out_f = open(derive_out_file, "w")
                yield out_f, save_dict
                out_f.close()

    def derive(self, n, steps=None, out_file=None):
        # # some scheduled value will be used in test too, e.g. surrogate_lr, gumbel temperature...
        # called in `load` method already
        # self.on_epoch_start(self.epoch)
        with self.controller.begin_mode("eval"):
            rollouts = self.controller.sample(n)
            save_dict = {}
            with self._open_derive_out_file(out_file) as (out_f, save_dict):
                for i_sample, rollout in enumerate(rollouts):
                    if str(rollout.genotype) in save_dict:
                        _dump_with_perf(rollout, "str", out_f, index=i_sample)
                        rollout.set_perf(save_dict[str(rollout.genotype)])
                        continue
                    rollout = self.evaluator.evaluate_rollouts(
                        [rollout], is_training=False, eval_batches=steps
                    )[0]
                    print("Finish test {}/{}\r".format(i_sample + 1, n), end="")
                    if out_f is not None:
                        _dump_with_perf(rollout, "str", out_f, index=i_sample)
        return rollouts

    def save(self, path):
        base_optimizer_state = (
            self.controller_base_optimizer.state_dict()
            if self.controller_base_optimizer is not None
            else None
        )
        base_scheduler_state = (
            self.controller_base_scheduler.state_dict()
            if self.controller_base_scheduler is not None
            else None
        )
        meta_optimizer_state = (
            self.controller_meta_optimizer.state_dict()
            if self.controller_meta_optimizer is not None
            else None
        )
        meta_scheduler_state = (
            self.controller_meta_scheduler.state_dict()
            if self.controller_meta_scheduler is not None
            else None
        )
        state_dict = {
            "epoch": self.epoch,
            "base_controller_optimizer": base_optimizer_state,
            "base_controller_scheduler": base_scheduler_state,
            "meta_controller_optimizer": meta_optimizer_state,
            "meta_controller_scheduler": meta_scheduler_state,
        }
        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.last_epoch = self.epoch = checkpoint["epoch"]
        if self.controller_baseoptimizer is not None:
            self.controller_base_optimizer.load_state_dict(
                checkpoint["controller_base_optimizer"]
            )
        if self.controller_base_scheduler is not None:
            self.controller_base_scheduler.load_state_dict(
                checkpoint["controller_base_scheduler"]
            )
        if self.controller_meta_optimizer is not None:
            self.controller_meta_optimizer.load_state_dict(
                checkpoint["controller_meta_optimizer"]
            )
        if self.controller_meta_scheduler is not None:
            self.controller_meta_scheduler.load_state_dict(
                checkpoint["controller_meta_scheduler"]
            )
        self.on_epoch_start(self.last_epoch)

    def on_epoch_start(self, epoch):
        super(MetaTrainer, self).on_epoch_start(epoch)
        if self.controller_meta_scheduler is not None:
            self.controller_meta_scheduler.step(epoch - 1)
            self.logger.info(
                "Epoch %3d: controller LR: %.5f",
                epoch,
                self.controller_scheduler.get_lr()[0],
            )
