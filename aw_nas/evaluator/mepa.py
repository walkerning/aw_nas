# -*- coding: utf-8 -*-

from functools import partial
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import six

from aw_nas import utils
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.utils.exception import expect, ConfigException

class MepaEvaluator(BaseEvaluator): #pylint: disable=too-many-instance-attributes
    """
    An evaluator incorporate surrogate steps for evaluating performance ("controller" phase),
    and the corresponding meta-update surrogate steps for weights manager udpate ("wm" phase).


    The surrogate step technique can be incorporated with different types of
    weights managers.
    However, for now, only shared-weights based weights manager are supported.
    High-order gradients respect to the mepa parameters are ignored.

    Args:


    Schedulable Attributes:
        * controller_surrogate_steps
        * mepa_surrogate_steps
        * mepa_samples

    Example:

    """

    NAME = "mepa"

    SCHEDULABLE_ATTRS = [
        "controller_surrogate_steps",
        "mepa_surrogate_steps",
        "mepa_samples"
    ]

    def __init__( #pylint: disable=dangerous-default-value
            self, dataset, weights_manager, objective, rollout_type="discrete",
            batch_size=128, controller_surrogate_steps=1, mepa_surrogate_steps=1,
            mepa_optimizer={
                "type": "SGD",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 1e-4
            },
            mepa_scheduler={
                "type": "CosineWithRestarts",
                "t_0": 10,
                "eta_min": 0.0001,
                "factor": 2.0
            },
            surrogate_optimizer={"type": "SGD", "lr": 0.001}, surrogate_scheduler=None,
            # mepa samples for `update_evaluator`
            mepa_samples=1,
            # data queue configs: (surrogate, mepa, controller)
            data_portion=(0.1, 0.4, 0.5), mepa_as_surrogate=False,
            # only for rnn data
            bptt_steps=35,
            schedule_cfg=None):
        super(MepaEvaluator, self).__init__(dataset, weights_manager,
                                            objective, rollout_type, schedule_cfg)

        # check rollout type
        expect(self.rollout_type == self.weights_manager.rollout_type,
               "the rollout type of evaluator/weights_manager must match, "
               "check the configuration. ({}/{})".format(
                   self.rollout_type,
                   self.weights_manager.rollout_type), ConfigException)

        self._data_type = self.dataset.data_type()
        self._device = self.weights_manager.device

        # configs
        self.batch_size = batch_size
        self.controller_surrogate_steps = controller_surrogate_steps
        self.mepa_surrogate_steps = mepa_surrogate_steps
        self.data_portion = data_portion
        self.mepa_as_surrogate = mepa_as_surrogate
        self.mepa_samples = mepa_samples

        # rnn specific configs
        self.bptt_steps = bptt_steps

        # initialize optimizers and schedulers
        # do some checks
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

        # initialize the optimizers and schedulers
        self.surrogate_optimizer = utils.init_optimizer(self.weights_manager.parameters(),
                                                        surrogate_optimizer)
        self.mepa_optimizer = utils.init_optimizer(self.weights_manager.parameters(),
                                                   mepa_optimizer)

        self.surrogate_scheduler = utils.init_scheduler(self.surrogate_optimizer,
                                                        surrogate_scheduler)
        self.mepa_scheduler = utils.init_scheduler(self.mepa_optimizer,
                                                   mepa_scheduler)

        # for performance when doing 1-sample ENAS in `update_evaluator`
        if self.mepa_surrogate_steps == 0 and self.mepa_samples == 1:
            # Will call `step_current_gradients` of weights manager
            self.logger.info("As `mepa_surrogate_steps==0`(ENAS) and `mepa_sample==1`, "
                             "to speed up, will accumulate mepa gradients in-place and call "
                             "`super_net.step_current_gradients`.")
            self.mepa_step_current = True
        else:
            self.mepa_step_current = False

        # initialize the data queues
        self._init_data_queues_and_hidden(self._data_type, data_portion, mepa_as_surrogate)

        # initialize reward criterions used by `get_rollout_reward`
        self._init_criterions(self.rollout_type)

        # for report surrogate loss
        self.epoch_average_meters = defaultdict(utils.AverageMeter)

    # ---- APIs ----
    @classmethod
    def supported_data_types(cls):
        """
        Return the supported data types
        """
        return ["image", "sequence"]

    @classmethod
    def supported_rollout_types(cls):
        return ["discrete", "differentiable"]

    def suggested_controller_steps_per_epoch(self):
        return len(self.controller_queue)

    def suggested_evaluator_steps_per_epoch(self):
        return len(self.mepa_queue)

    def evaluate_rollouts(self, rollouts, is_training, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        """
        Args:
            is_training: If true, only use one data batch from controller queue to evalaute.
                Otherwise, use the whole controller queue (or `portion` of controller queue if
                `portion` is specified).
            portion (float): Has effect only when `is_training==False`. If specified, evaluate
                on this `portion` of data of the controller queue.
            eval_batches (float): Has effect only when `is_training==False`. If specified, ignore
                `portion` argument, and only run eval on these batches of data.
            callback (callable): If specified, called with the evaluated rollout argument
                in `virtual` context.
                For `differentiable` rollout, when backward controller, the parameters of
                candidate net must be the one after surrogate steps, so this is necessary.
        .. warning::
            If `portion` or `eval_batches` is set, when `is_training==False`, different rollout
            will be tested on different data. The performance comparison might not be accurate.
        """
        if is_training: # the returned reward will be used for training controller
            # get one data batch from controller queue
            cont_data = next(self.controller_queue)
            cont_data = utils.to_device(cont_data, self._device)

            # prepare forward keyword arguments for candidate network
            _reward_kwargs = {k: v for k, v in self._reward_kwargs.items()}
            _reward_kwargs.update(self.c_hid_kwargs)

            # evaluate these rollouts on one batch of data
            for rollout in rollouts:
                cand_net = self.weights_manager.assemble_candidate(rollout)
                if return_candidate_net:
                    rollout.candidate_net = cand_net
                # prepare criterions
                criterions = [self._reward_func] + self._report_loss_funcs
                criterions = [partial(func, cand_net=cand_net) for func in criterions]

                # run surrogate steps and eval
                res = self._run_surrogate_steps(partial(self._eval_reward_func,
                                                        data=cont_data,
                                                        cand_net=cand_net,
                                                        criterions=criterions,
                                                        kwargs=_reward_kwargs,
                                                        callback=callback,
                                                        rollout=rollout),
                                                cand_net,
                                                self.controller_surrogate_steps,
                                                phase="controller_update")
        else: # only for test
            if eval_batches is not None:
                eval_steps = eval_batches
            else:
                eval_steps = len(self.controller_queue)
                if portion is not None:
                    expect(0.0 < portion < 1.0)
                    eval_steps = int(portion * eval_steps)

            for rollout in rollouts:
                cand_net = self.weights_manager.assemble_candidate(rollout)
                if return_candidate_net:
                    rollout.candidate_net = cand_net
                # prepare criterions
                criterions = [self._scalar_reward_func] + self._report_loss_funcs
                criterions = [partial(func, cand_net=cand_net) for func in criterions]

                # run surrogate steps and evalaute on queue
                # NOTE: if virtual buffers, must use train mode here...
                # if not virtual buffers(virtual parameter only), can use train/eval mode
                eval_func = lambda: cand_net.eval_queue(
                    self.controller_queue,
                    criterions=criterions,
                    steps=eval_steps,
                    mode="train",
                    # if test, differentiable rollout does not need to set detach_arch=True too
                    **self.c_hid_kwargs)
                res = self._run_surrogate_steps(eval_func, cand_net,
                                                self.controller_surrogate_steps,
                                                phase="controller_test")
                rollout.set_perfs(OrderedDict(zip(["reward", self._perf_name, "loss"], res)))
        return rollouts

    def update_rollouts(self, rollouts):
        """
        Nothing to be done.
        """

    def update_evaluator(self, controller):
        """
        Training meta parameter of the `weights_manager` (shared super network).
        """
        mepa_data = next(self.mepa_queue)
        mepa_data = utils.to_device(mepa_data, self._device)

        all_gradients = defaultdict(float)
        counts = defaultdict(int)
        report_stats = []
        for _ in range(self.mepa_samples):
            # sample rollout
            rollout = controller.sample()[0]

            # assemble candidate net
            cand_net = self.weights_manager.assemble_candidate(rollout)

            # prepare criterions
            eval_criterions = [self._scalar_reward_func] + self._report_loss_funcs
            eval_criterions = [partial(func, cand_net=cand_net) for func in eval_criterions]

            # return gradients if not update in-place
            # here, use loop variable as closure/cell var, this is goodn for now,
            # as all samples are evaluated sequentially (no parallel/delayed eval)
            gradient_func = lambda: cand_net.gradient(
                mepa_data,
                criterion=partial(self._mepa_loss_func, cand_net=cand_net),
                eval_criterions=eval_criterions,
                mode="train",
                return_grads=not self.mepa_step_current,
                **self.m_hid_kwargs
            )

            # run surrogate steps and evalaute on queue
            gradients, res = self._run_surrogate_steps(gradient_func, cand_net,
                                                       self.controller_surrogate_steps,
                                                       phase="mepa_update")
            if not self.mepa_step_current:
                for n, g_v in gradients:
                    all_gradients[n] += g_v
                    counts[n] += 1
            report_stats.append(res)

        # average the gradients and update the meta parameters
        if not self.mepa_step_current:
            all_gradients = {k: v / counts[k] for k, v in six.iteritems(all_gradients)}
            self.weights_manager.step(all_gradients.items(), self.mepa_optimizer)
        else:
            # call step_current_gradients; mepa_sample == 1
            self.weights_manager.step_current_gradients(self.mepa_optimizer)

        del all_gradients
        return OrderedDict(zip(["reward", self._perf_name, "loss"], np.mean(report_stats, axis=0)))

    def on_epoch_start(self, epoch):
        super(MepaEvaluator, self).on_epoch_start(epoch)
        self.weights_manager.on_epoch_start(epoch)
        self.objective.on_epoch_start(epoch)

        # scheduler epoch is 0-based, epoch of aw_nas components is 1-based
        lr_str = ""
        if self.mepa_scheduler is not None:
            self.mepa_scheduler.step(epoch-1)
            lr_str += "mepa LR: {:.5f}; ".format(self.mepa_scheduler.get_lr()[0])
        if self.surrogate_scheduler is not None:
            self.surrogate_scheduler.step(epoch-1)
            lr_str += "surrogate LR: {:.5f};".format(self.surrogate_scheduler.get_lr()[0])
        if lr_str:
            self.logger.info("Epoch %3d: %s", epoch, lr_str)

    def on_epoch_end(self, epoch):
        super(MepaEvaluator, self).on_epoch_end(epoch)
        self.weights_manager.on_epoch_end(epoch)
        self.objective.on_epoch_end(epoch)

        # logs meters info
        for name, meter in six.iteritems(self.epoch_average_meters):
            self.writer.add_scalar(name, meter.avg, epoch)

        for name, meter in six.iteritems(self.epoch_average_meters):
            if not meter.is_empty():

                self.writer.add_scalar(name, meter.avg, epoch)

        # optionally write tensorboard info
        if not self.writer.is_none():
            for name, meter in six.iteritems(self.epoch_average_meters):
                if not meter.is_empty():
                    self.writer.add_scalar(name, meter.avg, epoch)

        # reset all meters
        for meter in  self.epoch_average_meters.values():
            meter.reset()

    def save(self, path):
        optimizer_states = {}
        scheduler_states = {}
        for compo_name in ["mepa", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer_states[compo_name] = optimizer.state_dict()
            scheduler = getattr(self, compo_name + "_scheduler")
            if scheduler is not None:
                scheduler_states[compo_name] = scheduler.state_dict()
        state_dict = {
            "epoch": self.epoch,
            "weights_manager": self.weights_manager.state_dict(),
            "optimizers": optimizer_states,
            "schedulers": scheduler_states
        }

        if self._data_type == "sequence":
            hidden_states = {}
            for compo_name in ["mepa", "surrogate", "controller"]:
                # save hidden states
                hidden = getattr(self, compo_name + "_hiddens")
                hidden_states[compo_name] = hidden
            state_dict["hiddens"] = hidden_states

        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.weights_manager.load_state_dict(checkpoint["weights_manager"])

        # load hidden states if exists
        if "hiddens" in checkpoint:
            for compo_name in ["mepa", "surrogate", "controller"]:
                getattr(self, compo_name + "_hiddens").copy_(checkpoint["hiddens"][compo_name])

        optimizer_states = checkpoint["optimizers"]
        scheduler_states = checkpoint["schedulers"]
        for compo_name in ["mepa", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer.load_state_dict(optimizer_states[compo_name])
            scheduler = getattr(self, compo_name + "_scheduler")
            if scheduler is not None:
                scheduler.load_state_dict(scheduler_states[compo_name])

        # call `on_epoch_start` for scheduled values
        self.on_epoch_start(checkpoint["epoch"])

    # ---- helper methods ----
    def _run_surrogate_steps(self, func, cand_net, surrogate_steps, phase, update_metric=True):
        if surrogate_steps <= 0:
            return func()

        with cand_net.begin_virtual():
            sur_loss, sur_perf = cand_net.train_queue(
                self.surrogate_queue,
                optimizer=self.surrogate_optimizer,
                criterion=partial(self._mepa_loss_func, cand_net=cand_net),
                eval_criterions=[partial(func, cand_net=cand_net) \
                                 for func in self._report_loss_funcs],
                steps=surrogate_steps,
                **self.s_hid_kwargs
            )
            if update_metric:
                self.epoch_average_meters["loss/{}/surrogate".format(phase)].update(sur_loss)
                self.epoch_average_meters["perf/{}/surrogate".format(phase)].update(sur_perf)
            return func()

    def _get_hiddens_resetter(self, name):
        def _func():
            getattr(self, name + "_hiddens").zero_()
        _func.__name__ = "{}_hiddens_resetter".format(name)
        return _func

    def _reset_hidden(self): # not used now
        if self._data_type == "image":
            return
        # reset the hidden states
        [func() for func in self.hiddens_resetter]

    def _init_criterions(self, rollout_type):
        # criterion and forward keyword arguments for evaluating rollout in `evaluate_rollout`
        if rollout_type == "discrete":
            self._reward_func = self.objective.get_reward
            self._reward_kwargs = {}
            self._scalar_reward_func = self._reward_func
        else: # rollout_type == "differentiable"
            self._reward_func = partial(self.objective.get_loss,
                                        add_controller_regularization=True,
                                        add_evaluator_regularization=False)
            self._reward_kwargs = {"detach_arch": False}
            self._scalar_reward_func = lambda *args, **kwargs: \
                utils.get_numpy(self._reward_func(*args, **kwargs))
        self._perf_name = self.objective.perf_name()
        # criterion funcs for meta parameter training
        self._mepa_loss_func = partial(self.objective.get_loss,
                                       add_controller_regularization=False,
                                       add_evaluator_regularization=True)
        # criterion funcs for log/report
        self._report_loss_funcs = [self.objective.get_perf,
                                   partial(self.objective.get_loss_item,
                                           add_controller_regularization=False,
                                           add_evaluator_regularization=False)]

    def _init_data_queues_and_hidden(self, data_type, data_portion, mepa_as_surrogate):
        self._dataset_related_attrs = []
        if data_type == "image":
            queue_cfgs = [{"split": "train", "portion": p,
                           "batch_size": self.batch_size} for p in data_portion]
            self.s_hid_kwargs = {}
            self.c_hid_kwargs = {}
            self.m_hid_kwargs = {}
        else: # "sequence"
            # initialize hidden
            self.surrogate_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.mepa_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.controller_hiddens = self.weights_manager.init_hidden(self.batch_size)

            self.hiddens_resetter = [self._get_hiddens_resetter(n)
                                     for n in ["surrogate", "mepa", "controller"]]
            queue_cfgs = []
            for callback, portion in zip(self.hiddens_resetter, data_portion):
                queue_cfgs.append({"split": "train", "portion": portion,
                                   "batch_size": self.batch_size,
                                   "bptt_steps": self.bptt_steps,
                                   "callback": callback})
            self.s_hid_kwargs = {"hiddens": self.surrogate_hiddens}
            self.c_hid_kwargs = {"hiddens": self.controller_hiddens}
            self.m_hid_kwargs = {"hiddens": self.mepa_hiddens}
            self._dataset_related_attrs += ["surrogate_hiddens", "mepa_hiddens",
                                            "controller_hiddens"]

        self.surrogate_queue, self.mepa_queue, self.controller_queue\
            = utils.prepare_data_queues(self.dataset.splits(), queue_cfgs,
                                        data_type=self._data_type)
        if mepa_as_surrogate:
            # use mepa data queue as surrogate data queue
            self.surrogate_queue = self.mepa_queue
        len_surrogate = len(self.surrogate_queue) * self.batch_size \
                        if self.surrogate_queue else 0
        self.logger.info("Data sizes: surrogate: %s; controller: %d; mepa: %d;",
                         str(len_surrogate) if not mepa_as_surrogate else "(mepa queue)",
                         len(self.controller_queue) * self.batch_size,
                         len(self.mepa_queue) * self.batch_size)
        self._dataset_related_attrs += ["surrogate_queue", "mepa_queue", "controller_queue",
                                        "s_hid_kwargs", "c_hid_kwargs", "m_hid_kwargs"]

    def _eval_reward_func(self, data, cand_net, criterions, rollout, callback, kwargs):
        res = cand_net.eval_data(
            data,
            criterions=criterions,
            mode="train",
            **kwargs)
        rollout.set_perfs(OrderedDict(zip(["reward", self._perf_name, "loss"], res)))
        callback(rollout)
        # set reward to be the scalar
        rollout.set_perf(utils.get_numpy(rollout.get_perf(name="reward")))
        return res

    def set_dataset(self, dataset):
        self.dataset = dataset
        self._data_type = self.dataset.data_type()
        self._init_data_queues_and_hidden(self._data_type, self.data_portion,
                                          self.mepa_as_surrogate)

    def __setstate__(self, state):
        super(MepaEvaluator, self).__setstate__(state)
        self.logger.warn("After load the evaluator from a pickle file, the dataset does not "
                         "get loaded automatically, initialize a dataset and call "
                         "`set_dataset(dataset)` ")

    def __getstate__(self):
        state = super(MepaEvaluator, self).__getstate__()
        del state["dataset"]
        for attr_name in self._dataset_related_attrs:
            if attr_name in state:
                del state[attr_name]
        return state
