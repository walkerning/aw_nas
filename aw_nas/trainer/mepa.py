# -*- coding: utf-8 -*-
"""
Trainer definition, this is the orchestration of all the components.
"""

from __future__ import print_function
from __future__ import division

from collections import defaultdict, OrderedDict
from functools import partial
import imageio
import six

import numpy as np
import torch
from torch import nn, optim

from aw_nas.trainer.base import BaseTrainer
from aw_nas import utils, assert_rollout_type
from aw_nas.utils.torch_utils import accuracy
from aw_nas.utils.exception import expect, ConfigException

__all__ = ["MepaTrainer"]

def _ce_loss(inputs, targets):
    return nn.CrossEntropyLoss()(inputs, targets).item()

def _top1_acc(*args, **kwargs):
    return float(accuracy(*args, **kwargs)[0]) / 100

def _rnn_tensor_loss(inputs, targets, act_reg, slow_reg):
    logits, raw_outs, outs, _ = inputs
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
    if act_reg > 0: # activation L2 reguarlization on dropped outputs
        loss += act_reg * outs.pow(2).mean()
    if slow_reg > 0: # slowness regularization
        loss += slow_reg * (raw_outs[1:] - raw_outs[:-1]).pow(2).mean()
    return loss

def _rnn_loss(inputs, targets, act_reg, slow_reg):
    return _rnn_tensor_loss(inputs, targets, act_reg, slow_reg).item()

def _rnn_perp(inputs, targets):
    return np.exp(_rnn_loss(inputs, targets, 0, 0))


class MepaTrainer(BaseTrainer): #pylint: disable=too-many-instance-attributes
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
    SCHEDULABLE_ATTRS = [
        "mepa_surrogate_steps",
        "mepa_samples",
        "controller_steps",
        "controller_surrogate_steps",
        "controller_samples"
    ]

    def __init__(self, #pylint: disable=dangerous-default-value,too-many-arguments,too-many-locals
                 controller, weights_manager, dataset,
                 rollout_type="discrete",
                 epochs=200, batch_size=64,
                 test_every=10,
                 # optimizers
                 surrogate_optimizer={"type": "SGD", "lr": 0.001},
                 controller_optimizer={
                     "type": "Adam",
                     "lr": 0.0035
                 },
                 mepa_optimizer={
                     "type": "SGD",
                     "lr": 0.01,
                     "momentum": 0.9,
                     "weight_decay": 1e-4
                 },
                 surrogate_scheduler=None,
                 controller_scheduler=None,
                 mepa_scheduler={
                     "type": "CosineWithRestarts",
                     "t_0": 10,
                     "eta_min": 0.0001,
                     "factor": 2.0
                 },
                 mepa_surrogate_steps=1, mepa_samples=1,
                 controller_steps=None, controller_surrogate_steps=1, controller_samples=4,
                 controller_train_every=1, controller_train_begin=1,
                 data_portion=(0.2, 0.2, 0.6), derive_queue="controller",
                 derive_surrogate_steps=1, derive_samples=8,
                 mepa_as_surrogate=False,
                 interleave_controller_every=None, interleave_report_every=50,
                 # only for rnn, sequence modeling
                 bptt_steps=35, reset_hidden_prob=None, rnn_reward_c=80.,
                 rnn_act_reg=0., rnn_slowness_reg=0.,
                 schedule_cfg=None):
        """
        Args:
            controller_steps (int): If None, (not explicitly given), assume every epoch consume
                one pass of the controller queue.
            interleave_controller_every (int): Interleave controller update steps every
                `interleave_controller_every` steps. If None, do not interleave, which means
                controller will only be updated after one epoch of mepa update.
        """
        super(MepaTrainer, self).__init__(controller, weights_manager, dataset, schedule_cfg)

        # check rollout type
        expect(rollout_type in self.supported_rollout_types(), "Unknown `rollout_type`",
               ConfigException) # supported rollout types
        self._rollout_type = assert_rollout_type(rollout_type)
        expect(self._rollout_type == \
               self.controller.rollout_type() == self.weights_manager.rollout_type(),
               "the rollout type of trainer/controller/weights_manager must match, "
               "check the configuration. ({}/{}/{})".format(
                   self._rollout_type,
                   self.controller.rollout_type(),
                   self.weights_manager.rollout_type()), ConfigException)
        self._data_type = self.dataset.data_type()
        self._device = self.weights_manager.device

        # configurations
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_every = test_every
        self.mepa_surrogate_steps = mepa_surrogate_steps
        self.mepa_samples = mepa_samples
        self.mepa_as_surrogate = mepa_as_surrogate
        self.controller_steps = controller_steps
        self.controller_surrogate_steps = controller_surrogate_steps
        self.controller_samples = controller_samples
        self.controller_train_every = controller_train_every
        self.controller_train_begin = controller_train_begin
        self.derive_surrogate_steps = derive_surrogate_steps
        self.derive_samples = derive_samples
        self.interleave_controller_every = interleave_controller_every
        self.interleave_report_every = interleave_report_every

        # rnn specific configurations
        self.bptt_steps = bptt_steps
        self.reset_hidden_prob = reset_hidden_prob # reset_hidden_prob not implemented yet
        self.rnn_reward_c = rnn_reward_c
        self.rnn_act_reg = rnn_act_reg
        self.rnn_slowness_reg = rnn_slowness_reg

        # do some checks
        expect(derive_queue in {"surrogate", "controller", "mepa"},
               "Unknown `derive_queue`", ConfigException)
        expect(len(data_portion) == 3,
               "`data_portion` should have length 3.", ConfigException)
        if self.mepa_surrogate_steps == 0 and self.controller_surrogate_steps == 0:
            expect(data_portion[0] == 0,
                   "Do not waste data, set the first element of `data_portion` to 0 "
                   "when there are not surrogate steps.", ConfigException)
        if mepa_as_surrogate:
            expect(data_portion[0] == 0,
                   "`mepa_as_surrogate` is set true, will use mepa valid data as surrogate data "
                   "set the first element of `data_portion` to 0.", ConfigException)

        # initialize the optimizers
        self.surrogate_optimizer = self._init_optimizer(self.weights_manager.parameters(),
                                                        surrogate_optimizer)
        self.controller_optimizer = self._init_optimizer(self.controller.parameters(),
                                                         controller_optimizer)
        self.controller_step_current = False
        if self._rollout_type == "differentiable":
            if self.controller_surrogate_steps == 0:
                # Will call `step_current_gradients` of controller
                self.logger.info("As controller are optimized using differentiable relaxation, "
                                 "and `controller_surrogate_steps==0`(ENAS), to speed up, "
                                 "will accumulate controller gradients in-place and call "
                                 "`controller.step_current_gradients`.")
                self.controller_step_current = True

        self.mepa_optimizer = self._init_optimizer(self.weights_manager.parameters(),
                                                   mepa_optimizer)
        if self.mepa_surrogate_steps == 0 and self.mepa_samples == 1:
            # Will call `step_current_gradients` of weights manager
            self.logger.info("As `mepa_surrogate_steps==0`(ENAS) and `mepa_sample==1`, "
                             "to speed up, will accumulate mepa gradients in-place and call "
                             "`super_net.step_current_gradients`.")
            self.mepa_step_current = True
        else:
            self.mepa_step_current = False

        # initialize the learning rate scheduler for optimizers
        self.surrogate_scheduler = self._init_scheduler(self.surrogate_optimizer,
                                                        surrogate_scheduler)
        self.controller_scheduler = self._init_scheduler(self.controller_optimizer,
                                                         controller_scheduler)
        self.mepa_scheduler = self._init_scheduler(self.mepa_optimizer,
                                                   mepa_scheduler)

        # initialize the data queues
        self._initialize_data_queues_and_hidden(self._data_type, data_portion,
                                                mepa_as_surrogate, derive_queue)

        self.mepa_steps = len(self.mepa_queue)
        expect(self.interleave_controller_every is None or (
            self.controller_steps is None or self.controller_steps == \
            self.mepa_steps // self.interleave_controller_every),
               "`controller_steps` must not be given or must match with "
               "`mepa_steps/interleave_controller_every`in interleave mode", ConfigException)
        if self.controller_steps is None:
            if self.interleave_controller_every is None:
                # if controller_steps is not explicitly given, and not in interleave mode,
                # assume every epoch consume one pass of the controller queue
                self.controller_steps = len(self.controller_queue)
            else:
                # in interleave mode, `controller_steps=mepa_steps / interleave_controller-every`
                self.controller_steps = self.mepa_steps // self.interleave_controller_every
        self.derive_steps = len(self.derive_queue)

        # states and other help attributes
        self.last_epoch = 0
        self.epoch = 1

        # eval criterions for controller
        self._init_criterions(self._data_type, self._rollout_type)

    def mepa_update(self, steps, finished_m_steps, finished_c_steps):
        surrogate_loss_meter = utils.AverageMeter()
        surrogate_acc_meter = utils.AverageMeter()
        valid_loss_meter = utils.AverageMeter()
        valid_acc_meter = utils.AverageMeter()

        self.controller.set_mode("eval")

        for i_mepa in range(1, steps+1): # mepa stands for meta param
            print("\rmepa step {}/{} ; controller step {}/{}"\
                  .format(finished_m_steps+i_mepa, self.mepa_steps,
                          finished_c_steps, self.controller_steps),
                  end="")
            mepa_data = next(self.mepa_queue)
            mepa_data = utils.to_device(mepa_data, self._device)

            all_gradients = defaultdict(float)
            counts = defaultdict(int)

            for _ in range(self.mepa_samples):
                rollout = self.get_new_candidates(1)[0]
                candidate_net = rollout.candidate_net
                _surrogate_optimizer = self.get_surrogate_optimizer()

                with self._begin_virtual(candidate_net, self.mepa_surrogate_steps):
                    # surrogate train steps
                    train_loss, train_acc = candidate_net.train_queue(
                        self.surrogate_queue,
                        optimizer=_surrogate_optimizer,
                        criterion=self._criterion,
                        eval_criterions=self._test_criterions,
                        steps=self.mepa_surrogate_steps,
                        **self.s_hid_kwargs
                    )

                    # gradients: List(Tuple(parameter name, gradient))
                    return_grads = not self.mepa_step_current
                    gradients, (loss, acc) = candidate_net.gradient(
                        mepa_data,
                        criterion=self._criterion,
                        eval_criterions=self._test_criterions,
                        mode="train",
                        return_grads=return_grads,
                        **self.m_hid_kwargs
                    )
                if train_loss is not None: # surrogate_steps > 0 (==0 is ENAS)
                    surrogate_loss_meter.update(train_loss)
                    surrogate_acc_meter.update(train_acc)

                valid_loss_meter.update(loss)
                valid_acc_meter.update(acc)

                if not self.mepa_step_current:
                    for n, g_v in gradients:
                        all_gradients[n] += g_v
                        counts[n] += 1

            # average the gradients and update the meta parameters
            if not self.mepa_step_current:
                all_gradients = {k: v / counts[k] for k, v in six.iteritems(all_gradients)}
                self.weights_manager.step(all_gradients.items(), self.mepa_optimizer)
            else:
                # call step_current_gradients; mepa_sample == 1
                self.weights_manager.step_current_gradients(self.mepa_optimizer)

        del all_gradients

        print("\r", end="")

        return surrogate_loss_meter.avg, surrogate_acc_meter.avg,\
            valid_loss_meter.avg, valid_acc_meter.avg

    def controller_update(self, steps, finished_m_steps, finished_c_steps):
        #pylint: disable=too-many-locals,too-many-branches
        surrogate_loss_meter = utils.AverageMeter()
        surrogate_acc_meter = utils.AverageMeter()
        valid_loss_meter = utils.AverageMeter()
        valid_acc_meter = utils.AverageMeter()
        controller_loss_meter = utils.AverageMeter()
        controller_stat_meters = None
        all_gradients = None

        self.controller.set_mode("train")
        for i_cont in range(1, steps+1):
            print("\rmepa step {}/{} ; controller step {}/{}"\
                  .format(finished_m_steps, self.mepa_steps,
                          finished_c_steps+i_cont, self.controller_steps),
                  end="")
            controller_data = next(self.controller_queue)
            controller_data = utils.to_device(controller_data, self._device)
            rollouts = self.get_new_candidates(self.controller_samples)

            # gradient accumulator for controller parameters, when it's differentiable
            if self._rollout_type == "differentiable":
                all_gradients = defaultdict(float)
                step_loss = 0.

            if self.controller_step_current:
                # no surrogate steps, call step_current_gradietns zero grad here.
                self.controller.zero_grads()

            for i_sample in range(self.controller_samples):
                rollout = rollouts[i_sample]
                candidate_net = rollout.candidate_net
                _surrogate_optimizer = self.get_surrogate_optimizer()

                with self._begin_virtual(candidate_net, self.controller_surrogate_steps):
                    # surrogate train steps
                    train_loss, train_acc = candidate_net.train_queue(
                        self.surrogate_queue,
                        optimizer=_surrogate_optimizer,
                        criterion=self._criterion,
                        eval_criterions=self._test_criterions,
                        steps=self.controller_surrogate_steps,
                        **self.s_hid_kwargs
                    )

                    c_eval_kwargs = {k: v for k, v in self._eval_kwargs.items()}
                    c_eval_kwargs.update(self.c_hid_kwargs)
                    loss, acc = candidate_net.eval_data(
                        controller_data,
                        criterions=self._eval_criterions,
                        mode="train",
                        **c_eval_kwargs
                    )

                    if self._rollout_type == "discrete":
                        rollout.set_perf(acc if self._data_type == "image" \
                                         else self.rnn_reward_c/acc)
                    else: # differentiable sampling/rollouts
                        if self.controller_surrogate_steps > 0:
                            # need keep track of the gradients
                            _loss, gradients = self.controller.gradient(loss)
                            for n, g_v in gradients:
                                all_gradients[n] += g_v
                            step_loss += _loss
                        else: # self.controller_step_current == True
                            _loss = self.controller.gradient(loss, return_grads=False,
                                                             zero_grads=False)
                            step_loss += _loss
                        ## this consume too much memory when controller_samples > 1
                        # rollout.set_perf(loss)

                if train_loss is not None: # surrogate_steps > 0 (==0 is ENAS)
                    surrogate_loss_meter.update(train_loss)
                    surrogate_acc_meter.update(train_acc)

                valid_loss_meter.update(utils.get_numpy(loss))
                valid_acc_meter.update(acc)

                del loss

            if self._rollout_type == "discrete":
                controller_loss = self.controller.step(rollouts, self.controller_optimizer)
            else: # differntiable rollout (controller is optimized using differentiable relaxation)
                if self.controller_samples != 1:
                    all_gradients = {n: g/self.controller_samples \
                                     for n, g in six.iteritems(all_gradients)}
                controller_loss = step_loss / self.controller_samples
                if self.controller_surrogate_steps > 0:
                    # update using the keeped gradients
                    self.controller.step_gradient(all_gradients.items(),
                                                  self.controller_optimizer)
                else:
                    # adjust lr and call step_current_gradients
                    # (update using the accumulated gradients)
                    if self.controller_samples != 1:
                        # adjust the lr to keep the effective learning rate unchanged
                        lr_bak = self.controller_optimizer.param_groups[0]["lr"]
                        self.controller_optimizer.param_groups[0]["lr"] \
                            = lr_bak / self.controller_samples
                    self.controller.step_current_gradient(self.controller_optimizer)
                    if self.controller_samples != 1:
                        self.controller_optimizer.param_groups[0]["lr"] = lr_bak

            controller_loss_meter.update(controller_loss)
            controller_stats = self.controller.summary(rollouts, log=False)
            if controller_stat_meters is None:
                controller_stat_meters = OrderedDict([(n, utils.AverageMeter())\
                                                      for n in controller_stats])
            [controller_stat_meters[n].update(v) for n, v in controller_stats.items()]

        del all_gradients
        print("\r", end="")

        return surrogate_loss_meter.avg, surrogate_acc_meter.avg,\
            valid_loss_meter.avg, valid_acc_meter.avg, controller_loss_meter.avg,\
            OrderedDict((n, meter.avg) for n, meter in controller_stat_meters.items())

    def train(self): #pylint: disable=too-many-statements
        assert self.is_setup, "Must call `trainer.setup` method before calling `trainer.train`."

        if self.interleave_controller_every is not None:
            inter_steps = self.mepa_steps // self.interleave_controller_every
            mepa_steps = self.interleave_controller_every
            controller_steps = self.controller_steps // inter_steps
        else:
            inter_steps = 1
            mepa_steps = self.mepa_steps
            controller_steps = self.controller_steps

        for epoch in range(self.last_epoch+1, self.epochs+1):
            m_surrogate_loss_meter = utils.AverageMeter()
            m_surrogate_acc_meter = utils.AverageMeter()
            m_valid_loss_meter = utils.AverageMeter()
            m_valid_acc_meter = utils.AverageMeter()

            c_surrogate_loss_meter = utils.AverageMeter()
            c_surrogate_acc_meter = utils.AverageMeter()
            c_valid_loss_meter = utils.AverageMeter()
            c_valid_acc_meter = utils.AverageMeter()
            c_loss_meter = utils.AverageMeter()
            c_stat_meters = None

            self.epoch = epoch # this is redundant as Component.on_epoch_start also set this

            # schedule values and optimizer learning rates
            self.on_epoch_start(epoch) # call `on_epoch_start` of sub-components
            lrs = self._lr_scheduler_step()
            _lr_schedule_str = "\n\t".join(["{:10}: {:.5f}".format(n, v) for n, v in lrs])
            self.logger.info("Epoch %3d: LR values:\n\t%s", epoch, _lr_schedule_str)

            finished_m_steps = 0
            finished_c_steps = 0
            for i_inter in range(1, inter_steps+1): # interleave mepa/controller training
                # meta parameter training
                s_loss, s_acc, v_loss, v_acc = self.mepa_update(mepa_steps,
                                                                finished_m_steps,
                                                                finished_c_steps)
                m_surrogate_loss_meter.update(s_loss)
                m_surrogate_acc_meter.update(s_acc)
                m_valid_loss_meter.update(v_loss)
                m_valid_acc_meter.update(v_acc)

                finished_m_steps += mepa_steps

                if epoch >= self.controller_train_begin and \
                   epoch % self.controller_train_every == 0:
                    # controller training
                    s_loss, s_acc, v_loss, v_acc, c_loss, c_stats \
                        = self.controller_update(controller_steps,
                                                 finished_m_steps, finished_c_steps)
                    c_surrogate_loss_meter.update(s_loss)
                    c_surrogate_acc_meter.update(s_acc)
                    c_valid_loss_meter.update(v_loss)
                    c_valid_acc_meter.update(v_acc)
                    c_loss_meter.update(c_loss)

                    if c_stat_meters is None:
                        c_stat_meters = OrderedDict([(n, utils.AverageMeter())\
                                                     for n in c_stats])
                    [c_stat_meters[n].update(v) for n, v in c_stats.items()]

                    finished_c_steps += controller_steps

                if i_inter % self.interleave_report_every == 0:
                    # log for every `interleave_report_every` interleaving steps
                    self.logger.info("(inter step %3d): "
                                     "mepa (%3d/%3d): %s: %.3f; loss: %.3f;  "
                                     "controller (%3d/%3d): %s: %.3f; loss: %.3f;",
                                     i_inter, finished_m_steps, self.mepa_steps,
                                     self._accperp_name, m_valid_acc_meter.avg,
                                     m_valid_loss_meter.avg,
                                     finished_c_steps, self.controller_steps,
                                     self._accperp_name, c_valid_acc_meter.avg,
                                     c_valid_loss_meter.avg)

            # log infomations of this epoch
            self.logger.info("Epoch %3d: [mepa update] surrogate train %s: %.3f ; loss: %.3f ; "
                             "valid %s: %.3f ; valid loss: %.3f",
                             epoch, self._accperp_name, m_surrogate_acc_meter.avg,
                             m_surrogate_loss_meter.avg, self._accperp_name,
                             m_valid_acc_meter.avg, m_valid_loss_meter.avg)
            self.logger.info("Epoch %3d: [controller update] controller loss: %.3f ; "
                             "surrogate train %s: %.3f ; loss: %.3f ; "
                             "mean %s (reward): %.3f ; mean cross entropy loss: %.3f",
                             epoch, c_loss_meter.avg, self._accperp_name,
                             c_surrogate_acc_meter.avg, c_surrogate_loss_meter.avg,
                             self._accperp_name,
                             c_valid_acc_meter.avg, c_valid_loss_meter.avg)
            self.logger.info("[controller stats] %s", \
                             "; ".join(["{}: {:.3f}".format(n, meter.avg) \
                                        for n, meter in c_stat_meters.items()]))

            # maybe write tensorboard info
            if not self.writer.is_none():
                self.writer.add_scalar(self._accperp_name+"/mepa_update/valid",
                                       m_valid_acc_meter.avg, epoch)
                self.writer.add_scalar("loss/mepa_update/valid", m_valid_loss_meter.avg, epoch)
                if not m_surrogate_loss_meter.is_empty():
                    self.writer.add_scalar(self._accperp_name+"/mepa_update/train_surrogate",
                                           m_surrogate_acc_meter.avg,
                                           epoch)
                    self.writer.add_scalar("loss/mepa_update/train_surrogate",
                                           m_surrogate_loss_meter.avg,
                                           epoch)

            # maybe write tensorboard info
            if not self.writer.is_none():
                self.writer.add_scalar(self._accperp_name+"/arch_update/valid",
                                       c_valid_acc_meter.avg, epoch)
                self.writer.add_scalar("loss/arch_update/valid",
                                       c_valid_loss_meter.avg, epoch)
                if not c_surrogate_loss_meter.is_empty():
                    self.writer.add_scalar(self._accperp_name+"/arch_update/train_surrogate",
                                           c_surrogate_acc_meter.avg,
                                           epoch)
                    self.writer.add_scalar("loss/arch_update/train_surrogate",
                                           c_surrogate_loss_meter.avg,
                                           epoch)
                self.writer.add_scalar("controller_loss",
                                       c_loss_meter.avg, epoch)

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

        # accs, losses = zip(*[(r.get_perf("acc"), r.get_perf("loss")) for r in rollouts])
        accs = [r.get_perf() for r in rollouts]
        idx = np.argmax(accs)
        mean_acc = np.mean(accs)
        # mean_loss = np.mean(losses)

        save_path = self._save_path("rollout/cell")
        if save_path is not None:
            # NOTE: If `train_dir` is None, the image will not be saved to tensorboard too
            fnames = rollouts[idx].plot_arch(save_path, label="epoch {}".format(self.epoch))
            if not self.writer.is_none():
                for cg_n, fname in fnames:
                    image = imageio.imread(fname)
                    self.writer.add_image("genotypes/{}".format(cg_n), image, self.epoch,
                                          dataformats="HWC")

        if self._data_type == "sequence":
            # additional information for language modeling perplexity
            perp_str = "; Perplexity {:.3f} (mean: {:.3f})".format(
                self.rnn_reward_c/accs[idx],
                np.mean(self.rnn_reward_c/np.array(accs))
            )
        else:
            perp_str = ""
        self.logger.info("TEST Epoch %3d: Among %d sampled archs: "
                         "BEST (in reward): %.3f (mean: %.3f)%s",
                         #"loss: %.2f (mean :%.3f)",
                         self.epoch, self.derive_samples, accs[idx], mean_acc, perp_str)
                         #losses[idx], mean_loss)
        self.logger.info("Saved this arch to %s.\nGenotype: %s",
                         save_path, rollouts[idx].genotype)
        self.controller.summary(rollouts, log=True, log_prefix="Rollouts Info: ", step=self.epoch)
        return rollouts

    def derive(self, n, steps=None, rollouts=None):
        # some scheduled value might be used in test too, e.g., surrogate_lr, gumbel temperature...
        self.on_epoch_start(self.epoch)
        derive_steps = steps or self.derive_steps

        with self.controller.begin_mode("eval"):
            if rollouts is None:
                rollouts = self.controller.sample(n)
            for i_sample in range(n):
                rollout = rollouts[i_sample]
                candidate_net = self.weights_manager.assemble_candidate(rollout)
                _surrogate_optimizer = self.get_surrogate_optimizer()

                with self._begin_virtual(candidate_net, self.controller_surrogate_steps):
                    # surrogate steps
                    candidate_net.train_queue(self.surrogate_queue, optimizer=_surrogate_optimizer,
                                              criterion=self._criterion,
                                              steps=self.controller_surrogate_steps,
                                              **self.s_hid_kwargs)
                    _, acc = candidate_net.eval_queue(
                        self.derive_queue,
                        criterions=self._test_criterions,
                        steps=derive_steps,
                        mode="train",
                        **self.d_hid_kwargs)
                    # NOTE: if virtual buffers, must use train mode here...
                    # if not virtual buffers(virtual parameter only), can use train/eval mode
                    print("Finish test {}/{}\r".format(i_sample+1, n), end="")
                # rollout.set_perf(loss, name="loss")
                # rollout.set_perf(acc, name="acc")
                # del rollout.candidate_net
                # rollout.candidate_net = None
                rollout.set_perf(acc if self._data_type == "image" else self.rnn_reward_c/acc)
        return rollouts

    def save(self, path):
        optimizer_states = {}
        scheduler_states = {}
        for compo_name in ["mepa", "controller", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer_states[compo_name] = optimizer.state_dict()
            scheduler = getattr(self, compo_name + "_scheduler")
            if scheduler is not None:
                scheduler_states[compo_name] = scheduler.state_dict()

        state_dict = {
            "epoch": self.epoch,
            "optimizers": optimizer_states,
            "schedulers": scheduler_states
        }
        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.last_epoch = self.epoch = checkpoint["epoch"]
        optimizer_states = checkpoint["optimizers"]
        scheduler_states = checkpoint["schedulers"]
        for compo_name in ["mepa", "controller", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_states[compo_name])
            scheduler = getattr(self, compo_name + "_scheduler")
            if scheduler is not None:
                scheduler.load_state_dict(scheduler_states[compo_name])

    @classmethod
    def supported_rollout_types(cls):
        return ["discrete", "differentiable"]

    @classmethod
    def supported_data_types(cls):
        return ["image", "sequence"]

    # ---- helper methods ----
    @staticmethod
    def _init_optimizer(params, cfg):
        if cfg:
            cfg = {k:v for k, v in six.iteritems(cfg)}
            opt_cls = getattr(optim, cfg.pop("type"))
            return opt_cls(params, **cfg)
        return None

    @staticmethod
    def _init_scheduler(optimizer, cfg):
        if cfg and optimizer is not None:
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

    def _get_hiddens_resetter(self, name):
        def _func():
            getattr(self, name + "_hiddens").zero_()
        _func.__name__ = "{}_hiddens_resetter".format(name)
        return _func

    def _reset_hidden(self):
        if self._data_type == "image":
            return
        # reset the hidden states
        [func() for func in self.hiddens_resetter]

    def _init_criterions(self, data_type, rollout_type):
        if data_type == "image":
            # the criterion used to train weights_manager
            self._criterion = nn.CrossEntropyLoss()
            # sclar_criterion return the scalarized verion of `self._criterion`
            _scalar_criterion = _ce_loss
            # the criterion used to update controller
            # (NOTE: this name might be misleading)
            _acc_or_perp = _top1_acc
            self._accperp_name = "acc"
            self._test_criterions = [_scalar_criterion, _acc_or_perp]
            self._eval_criterions = self._test_criterions if self._rollout_type == "discrete" \
                                    else [self._criterion, _acc_or_perp]
        else: # data_type == "sequence"
            self._criterion = partial(_rnn_tensor_loss,
                                      act_reg=self.rnn_act_reg, slow_reg=self.rnn_slowness_reg)
            _scalar_criterion = partial(_rnn_loss,
                                        act_reg=0., slow_reg=0.)
            _tensor_criterion = partial(_rnn_tensor_loss,
                                        act_reg=0., slow_reg=0.)
            # the criterion used to update controller
            _acc_or_perp = _rnn_perp
            self._accperp_name = "perp"
            self._test_criterions = [_scalar_criterion, _acc_or_perp]
            # for controller update, if `differentiable`, use `self._criterion`,
            #     which returns a tensor;
            #     if `discrete`, use `_top1_acc` or `_rnn_reward`, which returns a scalar
            self._eval_criterions = self._test_criterions if self._rollout_type == "discrete" \
                                    else [_tensor_criterion, _acc_or_perp]
        self._eval_kwargs = {} if self._rollout_type == "discrete" else {"detach_arch": False}

    def _initialize_data_queues_and_hidden(self, data_type, data_portion,
                                           mepa_as_surrogate, derive_queue):
        if data_type == "image":
            queue_cfgs = [{"split": "train", "portion": p,
                           "batch_size": self.batch_size} for p in data_portion]
            self.s_hid_kwargs = {}
            self.c_hid_kwargs = {}
            self.m_hid_kwargs = {}
        else: # "sequence"
            # initialize hidden
            self.surrogate_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.controller_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.mepa_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.hiddens_resetter = [self._get_hiddens_resetter(n)
                                     for n in ["surrogate", "controller", "mepa"]]
            queue_cfgs = []
            for callback, portion in zip(self.hiddens_resetter, data_portion):
                queue_cfgs.append({"split": "train", "portion": portion,
                                   "batch_size": self.batch_size,
                                   "bptt_steps": self.bptt_steps,
                                   "callback": callback})
            self.s_hid_kwargs = {"hiddens": self.surrogate_hiddens}
            self.c_hid_kwargs = {"hiddens": self.controller_hiddens}
            self.m_hid_kwargs = {"hiddens": self.mepa_hiddens}

        self.surrogate_queue, self.controller_queue, self.mepa_queue \
            = self.prepare_data_queues(self.dataset.splits(), queue_cfgs,
                                       data_type=data_type)
        if mepa_as_surrogate:
            # use mepa data queue as surrogate data queue
            self.surrogate_queue = self.mepa_queue
        self.derive_queue = getattr(self, derive_queue + "_queue")
        self.d_hid_kwargs = getattr(self, derive_queue[0] + "_hid_kwargs")
        len_surrogate = len(self.surrogate_queue) * self.batch_size \
                        if self.surrogate_queue else 0
        self.logger.info("Data sizes: surrogate: %s; controller: %d; mepa: %d; derive: (%s queue)",
                         str(len_surrogate) if not mepa_as_surrogate else "(mepa queue)",
                         len(self.controller_queue) * self.batch_size,
                         len(self.mepa_queue) * self.batch_size,
                         derive_queue)

    def get_surrogate_optimizer(self, params=None):
        return self.surrogate_optimizer

    @staticmethod
    def _begin_virtual(candidate_net, surrogate_steps):
        if surrogate_steps > 0:
            return candidate_net.begin_virtual()
        return utils.nullcontext()
