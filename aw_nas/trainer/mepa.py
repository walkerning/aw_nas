# -*- coding: utf-8 -*-
"""
Trainer definition, this is the orchestration of all the components.
"""

from __future__ import print_function
from __future__ import division

from collections import defaultdict
import imageio
import six

import numpy as np
import torch
from torch import nn, optim

from aw_nas.trainer.base import BaseTrainer
from aw_nas import utils
from aw_nas.utils.torch_utils import accuracy

__all__ = ["MepaTrainer"]

def _ce_loss_mean(*args, **kwargs):
    return nn.CrossEntropyLoss()(*args, **kwargs).mean().item()

def _top1_acc(*args, **kwargs):
    return float(accuracy(*args, **kwargs)[0])

class MepaTrainer(BaseTrainer):
    """
    Actually it's better called SurrogateStepTrainer...
    As the surrogate step technique can be incorporated
    with other weights manager, such as using a HyperNet to generate weights.
    In this trainer, high-order gradients respect to the meta parameters are ignored.

    Schedulable Attributes:
        mepa_surrogate_steps:
        mepa_samples:
        controller_steps:
        controller_surrogate_steps:
        controller_samples:
    """

    NAME = "mepa"
    SCHEDULEABLE_ATTRS = [
        "mepa_surrogate_steps",
        "mepa_samples",
        "controller_steps",
        "controller_surrogate_steps",
        "controller_samples"
    ]

    def __init__(self, #pylint: disable=dangerous-default-value,too-many-arguments
                 controller, weights_manager, dataset,
                 epochs=200, batch_size=64, test_every=10,
                 # optimizers
                 surrogate_optimizer={"type": "SGD", "lr": 0.001},
                 controller_optimizer={
                     "type": "Adam",
                     "lr": 3e-4,
                 },
                 mepa_optimizer={
                     "type": "SGD",
                     "lr": 0.01,
                     "momentum": 0.9,
                     "weight_decay": 3e-4
                 },
                 surrogate_scheduler=None,
                 controller_scheduler=None,
                 mepa_scheduler=None,
                 mepa_surrogate_steps=1, mepa_samples=1,
                 controller_steps=313, controller_surrogate_steps=1, controller_samples=4,
                 data_portion=(0.2, 0.4, 0.4), derive_queue="controller",
                 derive_surrogate_steps=1, derive_samples=8,
                 schedule_cfg=None):
        super(MepaTrainer, self).__init__(controller, weights_manager, dataset, schedule_cfg)

        # configurations
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_every = test_every
        self.mepa_surrogate_steps = mepa_surrogate_steps
        self.mepa_samples = mepa_samples
        self.controller_steps = controller_steps
        self.controller_surrogate_steps = controller_surrogate_steps
        self.controller_samples = controller_samples
        self.derive_surrogate_steps = derive_surrogate_steps
        self.derive_samples = derive_samples
        # do some checks
        assert derive_queue in {"surrogate", "controller", "mepa"}
        assert len(data_portion) == 3
        if self.mepa_surrogate_steps == 0 and self.mepa_surrogate_steps == 0:
            assert data_portion[0] == 0, \
                "Do not waste data, set the first element of `data_portion` to 0 "\
                "when there are not surrogate steps."

        # initialize the optimizers
        self.surrogate_optimizer = self._init_optimizer(self.weights_manager.parameters(),
                                                        surrogate_optimizer)
        self.controller_optimizer = self._init_optimizer(self.controller.parameters(),
                                                         controller_optimizer)
        self.mepa_optimizer = self._init_optimizer(self.weights_manager.parameters(),
                                                   mepa_optimizer)

        # initialize the learning rate scheduler for optimizers
        self.surrogate_scheduler = self._init_scheduler(self.surrogate_optimizer,
                                                        surrogate_scheduler)
        self.controller_scheduler = self._init_scheduler(self.controller_optimizer,
                                                         controller_scheduler)
        self.mepa_scheduler = self._init_scheduler(self.mepa_optimizer,
                                                   mepa_scheduler)

        # initialize the data queues
        queue_cfgs = [{"split": "train", "portion": p,
                       "batch_size": self.batch_size} for p in data_portion]
        self.surrogate_queue, self.controller_queue, self.mepa_queue \
            = self.prepare_data_queues(self.dataset.splits(), queue_cfgs)
        self.derive_queue = getattr(self, derive_queue + "_queue")

        self.mepa_steps = len(self.mepa_queue)
        self.derive_steps = len(self.derive_queue)

        # states and other help attributes
        self.begin_epoch = 1
        self.epoch = 1
        self._criterion = nn.CrossEntropyLoss()

    def train(self): #pylint: disable=too-many-statements,too-many-branches
        assert self.is_setup, "Must call `trainer.setup` method before calling `trainer.train`."
        for epoch in range(self.begin_epoch, self.epochs+1):
            self.epoch = epoch # this is redundant as Component.on_epoch_start also set this
            surrogate_loss_meter = utils.AverageMeter()
            surrogate_acc_meter = utils.AverageMeter()
            valid_loss_meter = utils.AverageMeter()
            valid_acc_meter = utils.AverageMeter()
            controller_loss_meter = utils.AverageMeter()

            # schedule values and optimizer learning rates
            self.on_epoch_start(epoch) # call `on_epoch_start` of sub-components
            lrs = self._lr_scheduler_step()
            _lr_schedule_str = "\n\t".join(["{:10}: {:.5f}".format(n, v) for n, v in lrs])
            self.logger.info("Epoch %3d: LR values:\n\t%s", epoch, _lr_schedule_str)

            # meta parameter training
            for i_mepa in range(self.mepa_steps): # mepa stands for meta param
                print("\rmepa step {}/{}".format(i_mepa, self.mepa_steps), end="")
                mepa_data = next(self.mepa_queue)
                all_gradients = defaultdict(float)
                counts = defaultdict(int)
                rollouts = self.get_new_candidates(self.mepa_samples)
                for i_sample in range(self.mepa_samples):
                    candidate_net = rollouts[i_sample].candidate_net
                    if self.mepa_surrogate_steps:
                        _surrogate_optimizer = self.get_surrogate_optimizer()
                        with candidate_net.begin_virtual(): # surrogate train steps
                            train_loss, train_acc \
                                = candidate_net.train_queue(self.surrogate_queue,
                                                            optimizer=_surrogate_optimizer,
                                                            criterion=self._criterion,
                                                            eval_criterions=[_ce_loss_mean,
                                                                             _top1_acc],
                                                            steps=self.mepa_surrogate_steps)
                            # gradients: List(Tuple(parameter name, gradient))
                            gradients, (loss, acc) = candidate_net.gradient(
                                mepa_data,
                                criterion=self._criterion,
                                eval_criterions=[_ce_loss_mean,
                                                 _top1_acc]
                            )
                            surrogate_loss_meter.update(train_loss)
                            surrogate_acc_meter.update(train_acc / 100)
                    else: # ENAS
                        # gradients: List(Tuple(parameter name, gradient))
                        gradients, (loss, acc) = candidate_net.gradient(
                            mepa_data,
                            criterion=self._criterion,
                            eval_criterions=[_ce_loss_mean,
                                             _top1_acc]
                        )

                    valid_loss_meter.update(loss)
                    valid_acc_meter.update(acc / 100)
                    for n, g_v in gradients:
                        all_gradients[n] += g_v
                        counts[n] += 1

                # average the gradients and update the meta parameters
                all_gradients = {k: v / counts[k] for k, v in six.iteritems(all_gradients)}
                self.weights_manager.step(all_gradients.items(), self.mepa_optimizer)

            print("\r", end="")
            self.logger.info("Epoch %3d: [mepa update] surrogate train acc: %.2f %% ; loss: %.3f ; "
                             "valid acc: %.2f %% ; valid loss: %.3f",
                             epoch, surrogate_acc_meter.avg * 100, surrogate_loss_meter.avg,
                             valid_acc_meter.avg * 100, valid_loss_meter.avg)

           # maybe write tensorboard info
            if not self.writer.is_none():
                self.writer.add_scalars("acc", {
                    "mepa_update/valid": valid_acc_meter.avg
                }, self.epoch)
                self.writer.add_scalars("loss", {
                    "mepa_update/valid": valid_loss_meter.avg,
                }, self.epoch)
                if not surrogate_loss_meter.is_empty():
                    self.writer.add_scalars("acc", {
                        "mepa_update/train_surrogate": surrogate_acc_meter.avg
                    }, self.epoch)
                    self.writer.add_scalars("loss", {
                        "mepa_update/train_surrogate": surrogate_loss_meter.avg,
                    }, self.epoch)

            surrogate_acc_meter.reset()
            surrogate_loss_meter.reset()
            valid_acc_meter.reset()
            valid_loss_meter.reset()

            # controller training
            for i_cont in range(self.controller_steps):
                print("\rcontroller step {}/{}".format(i_cont, self.controller_steps), end="")
                controller_data = next(self.controller_queue)
                rollouts = self.get_new_candidates(self.controller_samples)
                for i_sample in range(self.controller_samples):
                    rollout = rollouts[i_sample]
                    candidate_net = rollout.candidate_net
                    _surrogate_optimizer = self.get_surrogate_optimizer()
                    if self.controller_surrogate_steps:
                        with candidate_net.begin_virtual(): # surrogate train steps
                            train_loss, train_acc \
                                = candidate_net.train_queue(self.surrogate_queue,
                                                            optimizer=_surrogate_optimizer,
                                                            criterion=self._criterion,
                                                            eval_criterions=[_ce_loss_mean,
                                                                             _top1_acc],
                                                            steps=self.controller_surrogate_steps)
                            loss, acc = candidate_net.eval_data(controller_data,
                                                                criterions=[_ce_loss_mean,
                                                                            _top1_acc])
                        surrogate_loss_meter.update(train_loss)
                        surrogate_acc_meter.update(train_acc / 100)
                    else: # ENAS
                        loss, acc = candidate_net.eval_data(controller_data,
                                                            criterions=[_ce_loss_mean,
                                                                        _top1_acc])

                    acc = acc / 100
                    valid_loss_meter.update(loss)
                    valid_acc_meter.update(acc)
                    rollout.set_perf(acc)

                controller_loss = self.controller.step(rollouts, self.controller_optimizer)
                controller_loss_meter.update(controller_loss)

            print("\r", end="")
            self.logger.info("Epoch %3d: [controller update] controller loss: %.3f ; "
                             "surrogate train acc: %.2f %% ; loss: %.3f ; "
                             "mean accuracy (reward): %.2f %% ; mean cross entropy loss: %.3f",
                             epoch, controller_loss_meter.avg,
                             surrogate_acc_meter.avg * 100, surrogate_loss_meter.avg,
                             valid_acc_meter.avg * 100, valid_loss_meter.avg)

            # maybe write tensorboard info
            if not self.writer.is_none():
                self.writer.add_scalars("acc", {
                    "arch_update/valid": valid_acc_meter.avg
                }, self.epoch)
                self.writer.add_scalars("loss", {
                    "arch_update/valid": valid_loss_meter.avg,
                }, self.epoch)
                if not surrogate_loss_meter.is_empty():
                    self.writer.add_scalars("acc", {
                        "arch_update/train_surrogate": surrogate_acc_meter.avg
                    }, self.epoch)
                    self.writer.add_scalars("loss", {
                        "arch_update/train_surrogate": surrogate_loss_meter.avg,
                    }, self.epoch)
                self.writer.add_scalar("loss/arch_update/controller_loss",
                                       controller_loss_meter.avg, self.epoch)

            # maybe save checkpoints
            self.maybe_save()

            # maybe derive archs and test
            if self.test_every and self.epoch % self.test_every == 0:
                self.test()

            self.on_epoch_end(epoch) # call `on_epoch_end` of sub-components

    def test(self):
        """
        Derive and test, plot the best arch among the arch samples.
        """
        rollouts = self.derive(n=self.derive_samples)

        accs, losses = zip(*[(r.get_perf("acc"), r.get_perf("loss")) for r in rollouts])
        idx = np.argmax(accs)
        mean_acc = np.mean(accs)
        mean_loss = np.mean(losses)

        save_path = self._save_path("rollout/cell")
        if save_path is not None:
            # NOTE: If `train_dir` is None, the image will not be saved to tensorboard too
            fnames = rollouts[idx].plot_arch(save_path, label="epoch {}".format(self.epoch))
            if not self.writer.is_none():
                for cg_n, fname in fnames:
                    image = imageio.imread(fname)
                    self.writer.add_image("genotypes/{}".format(cg_n), image, self.epoch,
                                          dataformats="HWC")

        self.logger.info("TEST Epoch %3d: Among %d sampled archs: "
                         "BEST (in acc): accuracy %.1f %% (mean: %.1f %%); "
                         "loss: %.2f (mean :%.3f); "
                         "Saved this arch to %s.\nGenotype: %s",
                         self.epoch, self.derive_samples, accs[idx], mean_acc,
                         losses[idx], mean_loss, save_path, rollouts[idx].genotype)

    def derive(self, n):
        rollouts = self.get_new_candidates(n)
        for i_sample in range(n):
            rollout = rollouts[i_sample]
            candidate_net = rollout.candidate_net
            _surrogate_optimizer = self.get_surrogate_optimizer()

            with candidate_net.begin_virtual(): # surrogate steps
                candidate_net.train_queue(self.surrogate_queue, optimizer=_surrogate_optimizer,
                                          criterion=self._criterion,
                                          steps=self.controller_surrogate_steps)
                loss, acc = candidate_net.eval_queue(
                    self.derive_queue,
                    criterions=[
                        _ce_loss_mean,
                        _top1_acc
                    ], steps=self.derive_steps)
            rollout.set_perf(loss, name="loss")
            rollout.set_perf(acc, name="acc")
        return rollouts

    def save(self, path):
        optimizer_states = {}
        for compo_name in ["mepa", "controller", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer_states[compo_name] = optimizer.state_dict()
        state_dict = {
            "epoch": self.epoch,
            "optimizers": optimizer_states
        }
        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.begin_epoch = self.epoch = checkpoint["epoch"]
        optimizer_states = checkpoint["optimizers"]
        for compo_name in ["mepa", "controller", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_states[compo_name])

    # ---- helper methods ----
    @staticmethod
    def _init_optimizer(params, cfg):
        if cfg:
            cfg = {k:v for k, v in six.iteritems(cfg)}
            opt_cls = getattr(optim, cfg.pop("type", "SGD"))
            return opt_cls(params, **cfg)
        return None

    @staticmethod
    def _init_scheduler(optimizer, cfg):
        if cfg:
            cfg = {k:v for k, v in six.iteritems(cfg)}
            sch_cls = utils.get_scheduler_cls(cfg.pop("type"))
            return sch_cls(optimizer, **cfg)
        return None

    def _lr_scheduler_step(self):
        lrs = []
        for name in ("surrogate", "controller", "mepa"):
            scheduler = getattr(self, name + "_scheduler")
            if scheduler is not None:
                scheduler.step()
                lrs.append((name + " LR", scheduler.get_lr()[0]))
        return lrs

    def get_surrogate_optimizer(self, params=None):
        return self.surrogate_optimizer
