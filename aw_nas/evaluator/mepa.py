# -*- coding: utf-8 -*-

import sys
import math
import copy
import contextlib
from functools import partial
from collections import defaultdict, OrderedDict

import six
import numpy as np
import torch
from torch import nn

from aw_nas import utils
from aw_nas.base import Component
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.utils.exception import expect, ConfigException


def _dropout_eval_forward(module, inputs):
    module.training = False
    return nn.Dropout.forward(module, inputs)

@contextlib.contextmanager
def _patch_dropout_forward(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.forward = partial(_dropout_eval_forward, module)
    yield
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.forward = partial(nn.Dropout.forward, module)

def _summary_inner_diagnostics(t_accs,
                               t_losses,
                               v_accs,
                               v_losses,
                               c_accs=None,
                               c_losses=None):
    # different inner step is not supported now
    steps = len(t_accs[0])
    t_accs = np.array(t_accs).reshape((-1, steps, 2))
    t_acc_bases = t_accs[:, :, 0]
    t_acc_diffs = t_accs[:, :, 1] - t_accs[:, :, 0]
    t_losses = np.array(t_losses).reshape((-1, steps, 2))
    t_loss_diffs = t_losses[:, :, 1] - t_losses[:, :, 0]

    v_accs = np.array(v_accs).reshape((-1, steps + 1))
    v_acc_bases = v_accs[:, 0]
    v_losses = np.array(v_losses).reshape((-1, steps + 1))
    v_acc_diffs = v_accs[:, 1:] - v_accs[:, 0:1]
    v_loss_diffs = v_losses[:, 1:] - v_losses[:, 0:1]
    if c_accs is not None and c_accs:
        c_accs = np.array(c_accs).reshape((-1, steps + 1))
        c_acc_bases = v_accs[:, 0]
        c_losses = np.array(c_losses).reshape((-1, steps + 1))
        c_acc_diffs = c_accs[:, 1:] - c_accs[:, 0:1]
        c_loss_diffs = c_losses[:, 1:] - c_losses[:, 0:1]
        return [t_acc_diffs, t_loss_diffs, v_acc_diffs, v_loss_diffs, c_acc_diffs, c_loss_diffs],\
            [t_acc_bases, v_acc_bases, c_acc_bases]
    return [t_acc_diffs, t_loss_diffs, v_acc_diffs,
            v_loss_diffs], [t_acc_bases, v_acc_bases]


class LearnableLrOutPlaceSGD(Component, nn.Module):
    def __init__(self,
                 named_params,
                 init_learning_rate,
                 device,
                 num_inner_steps,
                 learnable_lr=False,
                 adam_beta=None,
                 amsgrad=False,
                 adam_eps=1e-8,
                 base_schedule_cfg=None,
                 optimizer_cfg=None,
                 scheduler_cfg=None,
                 max_grad_norm=None):
        nn.Module.__init__(self)
        Component.__init__(self, schedule_cfg=None)
        assert init_learning_rate > 0., "learning_rate should be positive."
        if adam_beta is not None:
            expect(
                isinstance(adam_beta, (list, tuple)) and len(adam_beta) == 2,
                "Bad `betas` specification for Adam optimizer")

        self.device = device
        self.base_lr = torch.ones(1, device=device) * init_learning_rate
        self.num_inner_steps = num_inner_steps
        self.learnable_lr = learnable_lr
        self.adam_beta = adam_beta
        self.amsgrad = amsgrad
        self.adam_eps = adam_eps
        self.max_grad_norm = max_grad_norm

        if learnable_lr:
            self.named_log_lrs = nn.ParameterDict()
            for key, _ in six.iteritems(named_params):
                self.named_log_lrs[key.replace(".", "-")] = nn.Parameter(
                    data=torch.zeros(self.num_inner_steps),
                    requires_grad=self.learnable_lr)
        if adam_beta:
            self.adam_beta_tensor = torch.tensor(adam_beta).to(device)
            self.logger.info(
                "Use adam optimizer: betas: %s; eps: %e; amsgrad: %s",
                adam_beta, adam_eps, amsgrad)
            self.named_exp_avg = {}
            self.named_exp_avg_sq = {}
            self.named_steps = {}
            if amsgrad:
                self.named_max_exp_avg_sq = {}
            for key, param in six.iteritems(named_params):
                key = key.replace(".", "-")
                self.named_exp_avg[key] = torch.zeros_like(param.data)
                self.named_exp_avg_sq[key] = torch.zeros_like(param.data)
                self.named_steps[key] = torch.zeros([1], device=device)
                self.register_buffer("adam_exp_avg_" + key,
                                     self.named_exp_avg[key])
                self.register_buffer("adam_exp_avg_sq_" + key,
                                     self.named_exp_avg_sq[key])
                self.register_buffer("adam_steps_" + key,
                                     self.named_steps[key])
                if amsgrad:
                    self.named_max_exp_avg_sq[key] = torch.zeros_like(
                        param.data)
                    self.register_buffer("adam_max_exp_avg_sq_" + key,
                                         self.named_max_exp_avg_sq[key])

        self.base_schedule_cfg = base_schedule_cfg
        # init scheduler for `self.base_lr`
        self.base_scheduler = utils.init_tensor_scheduler(
            self.base_lr, base_schedule_cfg)

        # init optimzier/scheduler for optionally learnable relative `exp(self.named_log_lrs)`
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer_cfg)
        # NOTE: scheduler.step will be called in mepa evaluator
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler_cfg)
        self.to(device)

    def step(self,
             loss,
             named_weights,
             idx_step,
             high_order,
             allow_unused=False):
        """
        Update parameters out-of-place, using SGD rule.
        Args:
          named_weights (dict): name to parameters dict
          idx_step (int): the index of the inner step
          high_order (bool): if true, will create graph when computing the gradient for updates,
              the high-order gradients will backward through these gradients to named_weights too.
        """
        names, weights = zip(*list(named_weights.items()))
        grads = torch.autograd.grad(loss,
                                    weights,
                                    create_graph=high_order,
                                    allow_unused=allow_unused)
        named_grads = dict(zip(names, grads))
        new_named_weights = dict()
        for n in named_weights:
            grad = named_grads.get(n, None)
            key = n.replace(".", "-")
            if grad is not None:
                lr = (self.base_lr * torch.exp(self.named_log_lrs[key][idx_step])) \
                     if self.learnable_lr else self.base_lr
                if not self.adam_beta:
                    # SGD
                    new_named_weights[n] = named_weights[n] - grad * lr
                else:
                    # Adam
                    beta1, beta2 = self.adam_beta_tensor
                    self.named_steps[key] += 1
                    exp_avg, exp_avg_sq = self.named_exp_avg[
                        key], self.named_exp_avg_sq[key]
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if self.amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        max_exp_avg_sq = self.named_max_exp_avg_sq[key]
                        torch.max(max_exp_avg_sq,
                                  exp_avg_sq,
                                  out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(self.adam_eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(self.adam_eps)
                    bias_correction1 = 1 - beta1**self.named_steps[key]
                    bias_correction2 = 1 - beta2**self.named_steps[key]
                    step_size = lr * math.sqrt(
                        bias_correction2) / bias_correction1
                    if self.learnable_lr:
                        new_named_weights[
                            n] = named_weights[n] - step_size * exp_avg / denom
                    else:
                        new_named_weights[n] = named_weights[n].addcdiv(
                            -step_size.item(), exp_avg, denom)
            else:
                new_named_weights[n] = named_weights[n]
        return new_named_weights

    def update(self, gradients):
        assert self.learnable_lr

        self.zero_grad()
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           self.max_grad_norm)
        # apply the gradients
        self.optimizer.step()

    def on_epoch_start(self, epoch):
        self.logger.info("surrogate base lr: %.5f", self.base_lr)
        if self.learnable_lr:
            all_log_lrs = np.array([
                v.detach().cpu().numpy() for v in self.named_log_lrs.values()
            ])
            min_log_lrs = np.min(all_log_lrs, axis=0)
            max_log_lrs = np.max(all_log_lrs, axis=0)
            self.logger.info(
                "relative surrogate lr: %s",
                "\n\t".join([
                    "step %d: %.4f (log %.4f) - %.4f (log %.4f)" % \
                    (i, np.exp(min_log_lr), min_log_lr,
                     np.exp(max_log_lr), max_log_lr)
                    for i, (min_log_lr, max_log_lr) in enumerate(zip(min_log_lrs, max_log_lrs))]))


class MepaEvaluator(BaseEvaluator):  #pylint: disable=too-many-instance-attributes
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
        "controller_surrogate_steps", "mepa_surrogate_steps", "mepa_samples"
    ]

    def __init__(  #pylint: disable=dangerous-default-value
            self,
            dataset,
            weights_manager,
            objective,
            rollout_type="discrete",
            batch_size=128,
            controller_surrogate_steps=1,
            mepa_surrogate_steps=1,
            derive_surrogate_steps=None,
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
            surrogate_optimizer={
                "type": "SGD",
                "lr": 0.001
            },
            surrogate_scheduler=None,
            schedule_every_batch=False,
            # whether load optimizer/scheduler when loading
            load_optimizer=True,
            load_scheduler=True,
            strict_load_weights_manager=True,
            # advance meta learning configuration
            use_maml_plus=False,
            high_order=False,
            learn_per_weight_step_lr=False,
            use_multi_step_loss=False,
            multi_step_loss_epochs=10,
            multi_step_loss_start=None,
            surrogate_lr_optimizer=None,
            surrogate_lr_scheduler=None,
            report_inner_diagnostics=False,
            report_cont_data_diagnostics=False,
            update_mepa_surrogate_steps=None,
            # mepa samples for `update_evaluator`
            mepa_samples=1,
            disable_step_current=False,
            use_same_surrogate_data=False,
            # If true, `evaluate_rollout` evaluates using the whole controller queue
            # rather a batch during training
            evaluate_with_whole_queue=False,
            update_evaluator_report_perfs=True,
            # data queue configs: (surrogate, mepa, controller)
            data_portion=(0.1, 0.4, 0.5),
            mepa_as_surrogate=False,
            shuffle_data_before_split=False,  # by default not shuffle data before train-val splito
            shuffle_indice_file=None,
            shuffle_data_before_split_seed=None,
            workers_per_queue=2,
            pin_memory_per_queue=True,
            # only work for differentiable controller now
            rollout_batch_size=1,
            # only for rnn data
            bptt_steps=35,
            multiprocess=False,
            schedule_cfg=None,
            switch_epochs=[],
            lr_schedule=None,
            lr_factor=1.0):
        super(MepaEvaluator,
              self).__init__(dataset, weights_manager, objective, rollout_type,
                             schedule_cfg)

        # check rollout type
        if self.rollout_type != "compare":
            expect(
                self.rollout_type == self.weights_manager.rollout_type,
                "the rollout type of evaluator/weights_manager must match, "
                "check the configuration. ({}/{})".format(
                    self.rollout_type,
                    self.weights_manager.rollout_type), ConfigException)
        else:
            # Do not check for now
            pass

        # only maml plus mode support `learn_per_weight_step_lr` and `high_order` config
        if use_maml_plus:
            expect(
                surrogate_optimizer is not None
                and surrogate_optimizer["type"] in {"SGD", "Adam"},
                "maml plus mode only support SGD/Adam surrogate optimizer",
                ConfigException)
        else:
            expect(
                not (learn_per_weight_step_lr or high_order
                     or use_multi_step_loss or report_inner_diagnostics),
                "only maml plus mode support `learn_per_weight_step_lr`, `high_order`"
                ", `use_multi_step_loss`, `report_inner_diagnostics` or "
                "`update_mepa_surrogate_steps` config", ConfigException)

        self._data_type = self.dataset.data_type()
        self._device = self.weights_manager.device
        self.multiprocess = multiprocess

        # configs
        self.batch_size = batch_size
        self.controller_surrogate_steps = controller_surrogate_steps
        if not isinstance(self.controller_surrogate_steps, int):
            if isinstance(self.controller_surrogate_steps, str):
                self.controller_surrogate_steps = tuple(
                    eval(self.controller_surrogate_steps))  #pylint: disable=eval-used
            expect(
                isinstance(self.controller_surrogate_steps, (list, tuple)),
                "`controller_surrogate_steps` must be an integer, or a list/tuple, "
                "or a string eval to a list/tuple", ConfigException)

        self.derive_surrogate_steps = derive_surrogate_steps
        if self.derive_surrogate_steps is None:
            if isinstance(self.controller_surrogate_steps, int):
                self.derive_surrogate_steps = self.controller_surrogate_steps
            else:
                self.derive_surrogate_steps = np.max(
                    self.controller_surrogate_steps)

        self.mepa_surrogate_steps = mepa_surrogate_steps
        if update_mepa_surrogate_steps is not None:
            self.update_mepa_surrogate_steps = update_mepa_surrogate_steps
        else:
            self.update_mepa_surrogate_steps = mepa_surrogate_steps
        expect(
            self.update_mepa_surrogate_steps <= self.mepa_surrogate_steps,
            "`update_mepa_surrogate_steps` should not be bigger than `mepa_surrogate_steps`",
            ConfigException)
        if self.update_mepa_surrogate_steps != self.mepa_surrogate_steps:
            expect(
                report_inner_diagnostics,
                "specify`update_mepa_surrogate_steps`"
                "without `report_inner_diagnostics`==true is meaningless for now",
                ConfigException)
        if multi_step_loss_start is not None:
            assert len(
                multi_step_loss_start) == self.update_mepa_surrogate_steps + 1
            multi_step_loss_start = np.array(multi_step_loss_start)

        # random mepa_surrogate_steps
        if not isinstance(self.mepa_surrogate_steps, int):
            if isinstance(self.mepa_surrogate_steps, str):
                self.mepa_surrogate_steps = tuple(
                    eval(self.mepa_surrogate_steps))  #pylint: disable=eval-used
            expect(
                isinstance(self.mepa_surrogate_steps, (list, tuple)),
                "`mepa_surrogate_steps` must be an integer, or a list/tuple, "
                "or a string eval to a list/tuple", ConfigException)
            expect(
                not use_multi_step_loss, "When `use_multi_step_loss` is true, "
                "cannot use random sampled surrogate steps", ConfigException)

        self.logger.info(
            "mepa surrogate steps: %s; controller surrogate steps: %s; "
            "derive surrogate steps: %s", self.mepa_surrogate_steps,
            self.controller_surrogate_steps, self.derive_surrogate_steps)

        self.evaluate_with_whole_queue = evaluate_with_whole_queue
        self.disable_step_current = disable_step_current
        self.data_portion = data_portion
        self.workers_per_queue = workers_per_queue
        self.pin_memory_per_queue = pin_memory_per_queue
        self.shuffle_data_before_split = shuffle_data_before_split
        self.shuffle_indice_file = shuffle_indice_file
        self.shuffle_data_before_split_seed = shuffle_data_before_split_seed
        self.use_maml_plus = use_maml_plus
        self.high_order = high_order
        self.learn_per_weight_step_lr = learn_per_weight_step_lr
        self.use_multi_step_loss = use_multi_step_loss
        self.multi_step_loss_epochs = multi_step_loss_epochs
        self.multi_step_loss_start = multi_step_loss_start
        self.use_same_surrogate_data = use_same_surrogate_data
        self.mepa_as_surrogate = mepa_as_surrogate
        self.mepa_samples = mepa_samples
        self.rollout_batch_size = rollout_batch_size
        self.schedule_every_batch = schedule_every_batch
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.strict_load_weights_manager = strict_load_weights_manager
        self.report_inner_diagnostics = report_inner_diagnostics
        self.report_cont_data_diagnostics = report_cont_data_diagnostics
        self.update_evaluator_report_perfs = update_evaluator_report_perfs

        self.switch_epochs = switch_epochs
        self.lr_schedule = lr_schedule
        self.lr_factor = lr_factor
        self.mepa_optimizer_cfgs = mepa_optimizer
        self.mepa_scheduler_cfgs = mepa_scheduler

        # rnn specific configs
        self.bptt_steps = bptt_steps

        # initialize optimizers and schedulers
        # do some checks
        expect(
            len(data_portion) in {3, 4}, "`data_portion` should have length 3/4.",
            ConfigException)
        # if self.mepa_surrogate_steps == 0 and self.controller_surrogate_steps == 0:
        #     expect(data_portion[0] == 0,
        #            "Do not waste data, set the first element of `data_portion` to 0 "
        #            "when there are not surrogate steps.", ConfigException)
        if mepa_as_surrogate:
            expect(
                data_portion[0] == 0,
                "`mepa_as_surrogate` is set true, will use mepa valid data as surrogate data "
                "set the first element of `data_portion` to 0.",
                ConfigException)

        # initialize the optimizers and schedulers
        if self.use_maml_plus:
            adam_args = {}
            if surrogate_optimizer["type"] == "Adam":
                adam_args = {
                    k: v
                    for k, v in {
                        "adam_beta": surrogate_optimizer["betas"],
                        "adam_eps": surrogate_optimizer.get("eps", None),
                        "amsgrad": surrogate_optimizer.get("amsgrad", None)
                    }.items() if v is not None
                }
            self.surrogate_optimizer = LearnableLrOutPlaceSGD(
                OrderedDict(self.weights_manager.named_parameters()),
                init_learning_rate=surrogate_optimizer["lr"],
                device=self.weights_manager.device,
                num_inner_steps=self.mepa_surrogate_steps,
                learnable_lr=self.learn_per_weight_step_lr,
                max_grad_norm=5.0,
                base_schedule_cfg=surrogate_scheduler,
                optimizer_cfg=surrogate_lr_optimizer,
                scheduler_cfg=surrogate_lr_scheduler,
                **adam_args)
            self.surrogate_lr_optimizer = self.surrogate_optimizer.optimizer
            self.surrogate_lr_scheduler = self.surrogate_optimizer.scheduler
            self.surrogate_scheduler = self.surrogate_optimizer.base_scheduler
        else:
            self.surrogate_lr_optimizer = None
            self.surrogate_lr_scheduler = None
            self.surrogate_optimizer = utils.init_optimizer(
                self.weights_manager.parameters(), surrogate_optimizer)
            self.surrogate_scheduler = utils.init_scheduler(
                self.surrogate_optimizer, surrogate_scheduler)

        self.mepa_optimizer = utils.init_optimizer(
            self.weights_manager.parameters(), mepa_optimizer)
        self.mepa_scheduler = utils.init_scheduler(self.mepa_optimizer,
                                                   mepa_scheduler)

        # for performance when doing 1-sample ENAS in `update_evaluator`
        if not self.disable_step_current and\
           self.mepa_surrogate_steps == 0 and self.mepa_samples == 1:
            # Will call `step_current_gradients` of weights manager
            self.logger.info(
                "As `mepa_surrogate_steps==0`(ENAS), `mepa_sample==1`, "
                "and `disable_step_current` is not set, "
                "to speed up, will accumulate mepa gradients in-place and call "
                "`super_net.step_current_gradients`.")
            self.mepa_step_current = True
        else:
            self.mepa_step_current = False

        # initialize the data queues
        self._init_data_queues_and_hidden(self._data_type, data_portion,
                                          mepa_as_surrogate)

        # initialize reward criterions used by `get_rollout_reward`
        self._init_criterions(self.rollout_type)

        # for report surrogate loss
        self.epoch_average_meters = defaultdict(utils.AverageMeter)

        # evaluator update steps
        self.step = 0

        # diagnostics
        if self.report_inner_diagnostics:
            self.diag_all_ranges = []
            self.diag_all_ranges_base = []
            self.diag_all_inner_t_accs = []
            self.diag_all_inner_t_losses = []
            self.diag_all_inner_v_accs = []
            self.diag_all_inner_v_losses = []
            self.diag_all_inner_c_accs = []
            self.diag_all_inner_c_losses = []

        # for plateau lr scheduler
        self.plateau_scheduler_loss = []

    @property
    def device(self):
        return self.weights_manager.device

    # ---- APIs ----
    @classmethod
    def supported_data_types(cls):
        """
        Return the supported data types
        """
        return ["image", "sequence"]

    @classmethod
    def supported_rollout_types(cls):
        return ["discrete", "differentiable", "compare", "ofa"]

    def suggested_controller_steps_per_epoch(self):
        return len(self.controller_queue)

    def suggested_evaluator_steps_per_epoch(self):
        return len(self.mepa_queue)

    def evaluate_rollouts(self,
                          rollouts,
                          is_training,
                          portion=None,
                          eval_batches=None,
                          return_candidate_net=False,
                          callback=None):
        """
        Args:
            is_training: If true, only use one data batch from controller/derive queue to evaluate
                (this behavior can be override by `evaluate_with_whole_queue` cfg).
                Otherwise, use the whole controller queue (or `portion` of controller queue if
                `portion` is specified).
                * is_training=True: use `controller_queue`,
                                     the reward will be used to update controller.
                * is_training=False: use `derive_queue`, called by `trainer.derive`,
                                     usually, should not use random data aug.
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
        # support CompareRollout
        if self.rollout_type == "compare":
            eval_rollouts = sum([[r.rollout_1, r.rollout_2] for r in rollouts],
                                [])
        else:
            eval_rollouts = rollouts

        # controller_queue (for updating controller) or derive_queue (for testing rollouts only)
        data_queue = self.controller_queue if is_training else self.derive_queue
        hid_kwargs = self.c_hid_kwargs if is_training else self.d_hid_kwargs
        if not data_queue or not len(data_queue):
            return rollouts

        if is_training and not self.evaluate_with_whole_queue:
            # get one data batch from controller/derive queue
            cont_data = next(data_queue)
            cont_data = utils.to_device(cont_data, self.device)

            # prepare forward keyword arguments for candidate network
            _reward_kwargs = {k: v for k, v in self._reward_kwargs.items()}
            _reward_kwargs.update(hid_kwargs)

            if isinstance(self.controller_surrogate_steps, (tuple, list)):
                # random sample from the range
                num_surrogate_step = np.random.choice(
                    self.controller_surrogate_steps)
            else:
                num_surrogate_step = self.controller_surrogate_steps

            # evaluate these rollouts on one batch of data
            for rollout in eval_rollouts:
                cand_net = self.weights_manager.assemble_candidate(rollout, for_eval=True)
                if return_candidate_net:
                    rollout.candidate_net = cand_net
                # prepare criterions
                criterions = [self._reward_func] + self._report_loss_funcs
                criterions = [
                    partial(func, cand_net=cand_net) for func in criterions
                ]

                # run surrogate steps and eval
                res = self._run_surrogate_steps(partial(self._eval_reward_func,
                                                        data=cont_data,
                                                        cand_net=cand_net,
                                                        criterions=criterions,
                                                        kwargs=_reward_kwargs,
                                                        callback=callback,
                                                        rollout=rollout),
                                                cand_net,
                                                num_surrogate_step,
                                                phase="controller_update")
        else:  # only for test
            # We need to use train mode BN
            # let's make dropout in the eval mode
            with _patch_dropout_forward(self.weights_manager):
                if eval_batches is not None:
                    eval_steps = eval_batches
                else:
                    eval_steps = len(data_queue)
                    if portion is not None:
                        expect(0.0 < portion < 1.0)
                        eval_steps = int(portion * eval_steps)

                for i_rollout, rollout in enumerate(eval_rollouts):
                    print("\r{}/{}".format(i_rollout, len(eval_rollouts)), end="")
                    cand_net = self.weights_manager.assemble_candidate(rollout, for_eval=True)
                    if return_candidate_net:
                        rollout.candidate_net = cand_net
                    # prepare criterions
                    criterions = [self._scalar_reward_func
                                  ] + self._report_loss_funcs
                    criterions = [
                        partial(func, cand_net=cand_net) for func in criterions
                    ]

                    # run surrogate steps and evalaute on queue
                    # NOTE: if virtual buffers, must use train mode here...
                    # if not virtual buffers(virtual parameter only), can use train/eval mode
                    aggregate_fns = [
                        self.objective.aggregate_fn(name, is_training=False)
                        for name in self._all_perf_names
                    ]
                    eval_func = lambda: cand_net.eval_queue(
                        data_queue,
                        criterions=criterions,
                        steps=eval_steps,
                        # NOTE: In parameter-sharing evaluation, let's keep using train-mode BN!!!
                        mode="train",
                        # if test, differentiable rollout does not need to set detach_arch=True too
                        aggregate_fns=aggregate_fns,
                        **hid_kwargs)
                    res = self._run_surrogate_steps(eval_func,
                                                    cand_net,
                                                    self.derive_surrogate_steps,
                                                    phase="controller_test")
                    rollout.set_perfs(OrderedDict(zip(
                        self._all_perf_names, res)))  # res is already flattend

        # support CompareRollout
        if self.rollout_type == "compare":
            num_r = len(rollouts)
            for i_rollout in range(num_r):
                better = eval_rollouts[2 * i_rollout + 1].perf["reward"] > \
                         eval_rollouts[2 * i_rollout].perf["reward"]
                rollouts[i_rollout].set_perfs(
                    OrderedDict([
                        ("compare_result", better),
                    ]))
        return rollouts

    def update_rollouts(self, rollouts):
        """
        Nothing to be done.
        """

    def update_evaluator(self, controller):  #pylint: disable=too-many-branches
        """
        Training meta parameter of the `weights_manager` (shared super network).
        """
        mepa_data = next(self.mepa_queue)
        mepa_data = utils.to_device(mepa_data, self.device)

        all_gradients = defaultdict(float)
        if self.learn_per_weight_step_lr:
            all_opt_gradients = defaultdict(float)
        counts = defaultdict(int)
        report_stats = []

        if isinstance(self.mepa_surrogate_steps, (tuple, list)):
            # random sample from the range
            num_surrogate_step = np.random.choice(self.mepa_surrogate_steps)
        else:
            num_surrogate_step = self.mepa_surrogate_steps

        if self.use_same_surrogate_data:
            surrogate_data_list = [
                next(self.surrogate_queue) for _ in range(num_surrogate_step)
            ]
        holdout_data = next(self.controller_queue) \
                       if self.report_cont_data_diagnostics and self.controller_queue else None

        # sample rollout
        rollouts = controller.sample(n=self.mepa_samples,
                                     batch_size=self.rollout_batch_size)
        num_rollouts = len(rollouts)

        for _ind in range(num_rollouts):
            # surrogate data iterator
            surrogate_iter = iter(surrogate_data_list) if self.use_same_surrogate_data \
                             else self.surrogate_queue

            rollout = rollouts[_ind]
            # assemble candidate net
            cand_net = self.weights_manager.assemble_candidate(rollout)

            # prepare criterions
            if self.update_evaluator_report_perfs:
                # report get_perfs results
                eval_criterions = [self._scalar_reward_func] + self._report_perf_funcs
            else:
                eval_criterions = [self._scalar_reward_func]

            eval_criterions = [
                partial(func, cand_net=cand_net) for func in eval_criterions
            ]

            if not self.use_maml_plus:
                if self.mepa_step_current:
                    # if update in-place, here zero all the grads of weights_manager
                    self.weights_manager.zero_grad()
                else:
                    # otherwise, the meta parameters are updated using
                    # `wm.step(gradients, optimizer)`, do not need to zero grads
                    # During surrogate/inner steps, although the optimizer might
                    # alter parameters other than the ones used by a candidate net due to
                    # non-zeroed grads, these changes will not affect the calculation of
                    # the candidate net as the parameters are **not** used and the changes
                    # are reset once surrogate steps for a candidate net are done
                    pass
                # return gradients if not update in-place
                # here, use loop variable as closure/cell var, this is good for now,
                # as all samples are evaluated sequentially (no parallel/delayed eval)
                gradient_func = lambda: cand_net.gradient(
                    mepa_data,
                    criterion=partial(self._mepa_loss_func, cand_net=cand_net),
                    eval_criterions=eval_criterions,
                    mode="train",
                    return_grads=not self.mepa_step_current,
                    **self.m_hid_kwargs)

                # run surrogate steps and evalaute on queue
                gradients, res = self._run_surrogate_steps(
                    gradient_func,
                    cand_net,
                    num_surrogate_step,
                    phase="mepa_update",
                    queue=surrogate_iter)

                if not self.mepa_step_current:
                    for n, g_v in gradients:
                        all_gradients[n] += g_v
                        counts[n] += 1
            else:  # use_maml_plus
                gradients, opt_gradients, res = self._get_maml_plus_gradient(
                    cand_net,
                    num_surrogate_step,
                    train_iter=surrogate_iter,
                    val_data=mepa_data,
                    eval_criterions=eval_criterions,
                    holdout_data=holdout_data)
                for n, g_v in gradients:
                    all_gradients[n] += g_v
                    counts[n] += 1
                if self.learn_per_weight_step_lr:
                    for n, g_v in opt_gradients:
                        all_opt_gradients[n] += g_v

            # record stats of this arch
            report_stats.append(res)

        if self.schedule_every_batch:
            # schedule learning rate every evaluator step
            self._scheduler_step(self.step)
        self.step += 1

        if self.learn_per_weight_step_lr:
            # maml plus mode, update surrogate optimizer
            all_opt_gradients = {
                k: v / self.mepa_samples
                for k, v in six.iteritems(all_opt_gradients)
            }
            self.surrogate_optimizer.update(all_opt_gradients.items())

        # average the gradients and update the meta parameters
        if not self.mepa_step_current:
            all_gradients = {
                k: v / counts[k]
                for k, v in six.iteritems(all_gradients)
            }
            self.weights_manager.step(all_gradients.items(),
                                      self.mepa_optimizer)
        else:
            # call step_current_gradients; mepa_sample == 1
            self.weights_manager.step_current_gradients(self.mepa_optimizer)

        del all_gradients

        stats_res = OrderedDict(
            zip(self._all_perf_names_update_evaluator, np.mean(report_stats, axis=0)))

        if isinstance(self.mepa_scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.plateau_scheduler_loss.append(stats_res["loss"])
        # return stats
        return stats_res

    def reset_dataloader(self, is_training):
        if is_training:
            for queue in [self.surrogate_queue, self.mepa_queue, self.controller_queue]:
                queue.reset()
        else:
            self.derive_queue.reset()

    def switch(self, epoch):
        # lr_schedule implementation should be improved
        # since the lr would be wrong when loading ckpt
        mepa_optimizer_cfgs = copy.deepcopy(self.mepa_optimizer_cfgs)
        if self.lr_schedule == "constant":
            mepa_optimizer_cfgs["lr"] /= self.lr_factor

        self.mepa_optimizer = utils.init_optimizer(
            self.weights_manager.parameters(), mepa_optimizer_cfgs)
        self.mepa_scheduler = utils.init_scheduler(
            self.mepa_optimizer, self.mepa_scheduler_cfgs)

    def on_epoch_start(self, epoch):
        super(MepaEvaluator, self).on_epoch_start(epoch)
        self.weights_manager.on_epoch_start(epoch)
        self.objective.on_epoch_start(epoch)
        
        if epoch in self.switch_epochs:
            self.switch(epoch)
            self.logger.info(
                "Successfully switching the curriculum, and reset lr to {:.4f} ".format(
                    self.mepa_optimizer_cfgs["lr"]))
        
        if self.use_maml_plus:
            self.surrogate_optimizer.on_epoch_start(epoch)

        if not self.schedule_every_batch:
            # scheduler step is 0-based, epoch of aw_nas components is 1-based
            if isinstance(self.mepa_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.plateau_scheduler_loss:
                    self._scheduler_step(np.mean(self.plateau_scheduler_loss),
                                         log=True)
                    self.plateau_scheduler_loss = []
            else:
                self._scheduler_step(epoch - 1, log=True)

        else:
            self._scheduler_step(self.step, log=True)

        if self.use_multi_step_loss:
            self.logger.info(
                "multi step loss: %s",
                self._get_multi_step_loss_weights(
                    self.update_mepa_surrogate_steps))

    def on_epoch_end(self, epoch):
        super(MepaEvaluator, self).on_epoch_end(epoch)
        self.weights_manager.on_epoch_end(epoch)
        self.objective.on_epoch_end(epoch)
        if self.use_maml_plus:
            self.surrogate_optimizer.on_epoch_end(epoch)

        # optionally write tensorboard info
        if not self.writer.is_none():
            for name, meter in six.iteritems(self.epoch_average_meters):
                if not meter.is_empty():
                    self.writer.add_scalar(name, meter.avg, epoch)

        # reset all meters
        for meter in self.epoch_average_meters.values():
            meter.reset()

        # report and reset inner diagonostics
        if self.report_inner_diagnostics:
            perf_diffs, perf_bases = _summary_inner_diagnostics(
                self.diag_all_inner_t_accs, self.diag_all_inner_t_losses,
                self.diag_all_inner_v_accs, self.diag_all_inner_v_losses,
                self.diag_all_inner_c_accs, self.diag_all_inner_c_losses)
            diags = np.array([
                np.quantile(diffs, [0, 0.5, 1.0], axis=0)
                for diffs in perf_diffs
            ])
            diags_base = [
                np.quantile(perf_base, [0, 0.5, 1.0], axis=0)
                for perf_base in perf_bases
            ]
            self.diag_all_ranges.append(diags)  # 4/6 x 3 x step
            self.diag_all_ranges_base.append(diags_base)  # 4/6 x 3 x step

            str_ = ""
            # diagnostics of base accs
            db_t = diags_base[0]
            str_ += "\t{:16}: {}".format(
                "train base acc", ", ".join([
                    "{:7.3f}+-{:6.3f}".format(db_t[1, step],
                                              db_t[1, step] - db_t[0, step])
                    for step in range(db_t.shape[1])
                ]))
            str_ += "\n\t{:16}: {:7.3f}+-{:6.3f}".format(
                "valid base acc", diags_base[1][1],
                diags_base[1][1] - diags_base[1][0])
            if self.report_cont_data_diagnostics:
                str_ += "\n\t{:16}: {:7.3f}+-{:6.3f}".format(
                    "cont base acc", diags_base[2][1],
                    diags_base[2][1] - diags_base[2][0])
            str_ += "\n"

            # diagnostics of diffs of each inner step
            for name, diag in zip([
                    "train diff acc", "train diff loss", "valid diff acc",
                    "valid diff loss", "cont diff acc", "cont diff loss"
            ], diags):
                str_ += "\t{:16}: ".format(name)
                for step in range(diag.shape[1]):
                    str_ += "{:7.3f}+-{:6.3f}".format(
                        diag[1, step], diag[1, step] - diag[0, step])
                    if step < diag.shape[1] - 1:
                        str_ += ", "
                str_ += "\n"

            self.logger.info("inner step diagnostics:\n%s", str_)
            # reset inner loop diagnostics every epoch
            self.diag_all_inner_t_accs = []
            self.diag_all_inner_t_losses = []
            self.diag_all_inner_v_accs = []
            self.diag_all_inner_v_losses = []
            self.diag_all_inner_c_accs = []
            self.diag_all_inner_c_losses = []

    def save(self, path):
        optimizer_states = {}
        scheduler_states = {}
        for compo_name in ["mepa", "surrogate", "surrogate_lr"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if optimizer is not None:
                optimizer_states[compo_name] = optimizer.state_dict()
            scheduler = getattr(self, compo_name + "_scheduler")
            if scheduler is not None:
                scheduler_states[compo_name] = scheduler.state_dict()
        state_dict = {
            "epoch": self.epoch,
            "step": self.step,
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
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.weights_manager.load_state_dict(
            checkpoint["weights_manager"],
            strict=self.strict_load_weights_manager)

        # load hidden states if exists
        if "hiddens" in checkpoint:
            for compo_name in ["mepa", "surrogate", "controller"]:
                getattr(self, compo_name + "_hiddens").copy_(
                    checkpoint["hiddens"][compo_name])

        optimizer_states = checkpoint["optimizers"]
        scheduler_states = checkpoint["schedulers"]
        for compo_name in ["mepa", "surrogate"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if self.load_optimizer and optimizer is not None and compo_name in optimizer_states:
                optimizer.load_state_dict(optimizer_states[compo_name])
            scheduler = getattr(self, compo_name + "_scheduler")
            if self.load_scheduler and scheduler is not None and compo_name in scheduler_states:
                scheduler.load_state_dict(scheduler_states[compo_name])

        # call `on_epoch_start` for scheduled values
        self.step = checkpoint["step"]
        self.on_epoch_start(checkpoint["epoch"])

    # ---- helper methods ----
    def _get_multi_step_loss_weights(self, surrogate_steps):
        decay_step = self.epoch - 1
        num_w = surrogate_steps + 1
        loss_weights = self.multi_step_loss_start.copy() if self.multi_step_loss_start is not None \
                       else np.ones(shape=(num_w,)) * 1.0 / num_w
        min_value_non_final = 0.03 / (num_w - 1)
        decay_rate = (loss_weights[:-1] -
                      min_value_non_final) / self.multi_step_loss_epochs
        for i in range(num_w - 1):
            loss_weights[i] = np.maximum(
                loss_weights[i] - decay_step * decay_rate[i],
                min_value_non_final)
        loss_weights[-1] = 1 - np.sum(loss_weights[:-1])

        loss_weights = torch.Tensor(loss_weights).to(
            device=self.weights_manager.device)
        return loss_weights

    @staticmethod
    def _candnet_perf_use_param(cand_net,
                                params,
                                data,
                                loss_criterion,
                                forward_kwargs,
                                return_outputs=False):
        # TODO: only support cnn now. because directly use utils.accuracy
        #      should use self._report_loss_funcs instead maybe
        outputs = cand_net.forward_with_params(data[0],
                                               params=params,
                                               **forward_kwargs,
                                               mode="train")
        loss = loss_criterion(data[0], outputs, data[1])
        acc = utils.accuracy(outputs, data[1], topk=(1, ))[0]
        if return_outputs:
            return outputs, loss, acc
        return loss, acc

    def _get_maml_plus_gradient(self,
                                cand_net,
                                surrogate_steps,
                                train_iter,
                                val_data,
                                eval_criterions=None,
                                holdout_data=None):
        if holdout_data is not None:
            holdout_data = (holdout_data[0].to(cand_net.get_device()),
                            holdout_data[1].to(cand_net.get_device()))

        criterion = partial(self._mepa_loss_func, cand_net=cand_net)
        if hasattr(cand_net, "active_named_members"):
            # only get the active named parameters, other parameters
            # will not be used or updated by surrogate steps on this candidate net
            detached_copy_params = dict(
                cand_net.active_named_members(member="parameters",
                                              check_visited=True))
        else:
            detached_copy_params = dict(cand_net.named_parameters())
        if self.use_multi_step_loss:
            multi_step_loss_weights = self._get_multi_step_loss_weights(
                self.update_mepa_surrogate_steps)
        multi_step_losses = []
        use_multi_step_loss = self.use_multi_step_loss and self.epoch < self.multi_step_loss_epochs

        if self.report_inner_diagnostics:
            # diagnostics
            diag_inner_t_accs = []
            diag_inner_t_losses = []
            diag_inner_v_accs = []
            diag_inner_v_losses = []
            if holdout_data is not None:
                diag_inner_c_accs = []
                diag_inner_c_losses = []

        if self.update_mepa_surrogate_steps == 0 or self.use_multi_step_loss or \
           self.report_inner_diagnostics:
            if self.update_mepa_surrogate_steps == 0:
                val_outputs, val_loss_start, val_acc_start = self._candnet_perf_use_param(
                    cand_net,
                    detached_copy_params,
                    val_data,
                    criterion,
                    self.m_hid_kwargs,
                    return_outputs=True)
            else:
                val_loss_start, val_acc_start = self._candnet_perf_use_param(
                    cand_net, detached_copy_params, val_data, criterion,
                    self.m_hid_kwargs)
            val_acc_start = val_acc_start.item()
            if use_multi_step_loss:
                # every step valid loss
                multi_step_losses.append(multi_step_loss_weights[0] *
                                         val_loss_start)
            elif self.update_mepa_surrogate_steps == 0:
                # when `update_mepa_surrogate_steps`==0, maml loss that will be used
                # to update weights is the start valid loss before inner steps
                multi_step_losses.append(val_loss_start)
            val_loss_start = val_loss_start.item()
            if self.report_inner_diagnostics:
                # acc/loss on valid data before inner steps
                diag_inner_v_accs.append(val_acc_start)
                diag_inner_v_losses.append(val_loss_start)
                if self.report_cont_data_diagnostics:
                    c_loss_start, c_acc_start = self._candnet_perf_use_param(
                        cand_net, detached_copy_params, holdout_data,
                        criterion, self.c_hid_kwargs)
                    c_loss_start, c_acc_start = c_loss_start.item(
                    ), c_acc_start.item()
                    diag_inner_c_accs.append(c_acc_start)
                    diag_inner_c_losses.append(c_loss_start)

        for i_step in range(surrogate_steps):
            train_data = next(train_iter)
            train_data = (train_data[0].to(cand_net.get_device()),
                          train_data[1].to(cand_net.get_device()))
            outputs = cand_net.forward_with_params(train_data[0],
                                                   params=detached_copy_params,
                                                   **self.s_hid_kwargs,
                                                   mode="train")
            loss = criterion(train_data[0], outputs, train_data[1])
            detached_copy_params = self.surrogate_optimizer.step(
                loss, detached_copy_params, i_step, self.high_order)
            loss = loss.item()
            outputs = outputs.detach()

            if self.report_inner_diagnostics:
                acc_t_before = utils.accuracy(outputs,
                                              train_data[1],
                                              topk=(1, ))[0].item()
                loss_t_after, acc_t_after = self._candnet_perf_use_param(
                    cand_net, detached_copy_params, train_data, criterion,
                    self.s_hid_kwargs)
                # acc/loss on this batch of train data before/after inner step
                acc_t_after = acc_t_after.item()
                loss_t_after = loss_t_after.item()
                diag_inner_t_accs.append((acc_t_before, acc_t_after))
                diag_inner_t_losses.append((loss, loss_t_after))
            del outputs

            if self.report_inner_diagnostics or i_step == self.update_mepa_surrogate_steps - 1 \
               or (self.use_multi_step_loss and i_step <= self.update_mepa_surrogate_steps - 1):
                # foxfi: for now, use batch mean instead of running statistics for all calc
                if i_step == self.update_mepa_surrogate_steps - 1:
                    val_outputs, val_loss, val_acc = self._candnet_perf_use_param(
                        cand_net,
                        detached_copy_params,
                        val_data,
                        criterion,
                        self.m_hid_kwargs,
                        return_outputs=True)
                else:
                    val_loss, val_acc = self._candnet_perf_use_param(
                        cand_net, detached_copy_params, val_data, criterion,
                        self.m_hid_kwargs)
                val_acc = val_acc.item()
                if use_multi_step_loss and i_step <= self.update_mepa_surrogate_steps - 1:
                    # every step valid loss
                    multi_step_losses.append(
                        multi_step_loss_weights[i_step + 1] * val_loss)
                elif i_step == self.update_mepa_surrogate_steps - 1:
                    # final valid loss
                    multi_step_losses.append(val_loss)
                val_loss = val_loss.item()
                if self.report_inner_diagnostics:
                    diag_inner_v_losses.append(val_loss)
                    diag_inner_v_accs.append(val_acc)
                    if self.report_cont_data_diagnostics:
                        c_loss, c_acc = self._candnet_perf_use_param(
                            cand_net, detached_copy_params, holdout_data,
                            criterion, self.c_hid_kwargs)
                        c_loss, c_acc = c_loss.item(), c_acc.item()
                        diag_inner_c_accs.append(c_acc)
                        diag_inner_c_losses.append(c_loss)

        if self.report_inner_diagnostics:
            # save diagnostics for furthur handling/reporting
            self.diag_all_inner_t_accs.append(diag_inner_t_accs)
            self.diag_all_inner_t_losses.append(diag_inner_t_losses)
            self.diag_all_inner_v_accs.append(diag_inner_v_accs)
            self.diag_all_inner_v_losses.append(diag_inner_v_losses)
            if self.report_cont_data_diagnostics:
                self.diag_all_inner_c_accs.append(diag_inner_c_accs)
                self.diag_all_inner_c_losses.append(diag_inner_c_losses)

        # backward the maml loss
        maml_loss = torch.sum(torch.stack(multi_step_losses))
        cand_net.zero_grad()
        if self.learn_per_weight_step_lr:
            self.surrogate_optimizer.zero_grad()
        maml_loss.backward()

        # collect gradients of cand_net
        grads = [(k, v.grad.clone()) for k, v in cand_net.named_parameters()\
                 if v.grad is not None]

        # collect gradients of surrogate optimizer
        if self.learn_per_weight_step_lr:
            optimizer_grads = [(k, v.grad.clone())
                               for k, v in self.surrogate_optimizer.named_parameters()\
                               if v.grad is not None]
        else:
            optimizer_grads = None

        if eval_criterions:
            eval_res = utils.flatten_list([
                c(val_data[0], val_outputs, val_data[1])
                for c in eval_criterions
            ])
            return grads, optimizer_grads, eval_res
        return grads, optimizer_grads

    def _run_surrogate_steps(self,
                             func,
                             cand_net,
                             surrogate_steps,
                             phase,
                             update_metric=True,
                             queue=None):
        if surrogate_steps <= 0:
            return func()

        surrogate_queue = self.surrogate_queue if queue is None else queue

        kwargs = copy.deepcopy(self.s_hid_kwargs)
        if "differentiable" in self.rollout_type:
            kwargs["detach_arch"] = True

        with cand_net.begin_virtual():
            results = cand_net.train_queue(
                surrogate_queue,
                optimizer=self.surrogate_optimizer,
                criterion=partial(self._mepa_loss_func, cand_net=cand_net),
                eval_criterions=[partial(loss_func, cand_net=cand_net) \
                                 for loss_func in self._report_loss_funcs],
                steps=surrogate_steps,
                aggregate_fns=[self.objective.aggregate_fn(name) for name in self._all_perf_names],
                # **self.s_hid_kwargs
                **kwargs
            )
            if update_metric:
                sur_loss = results[0]
                self.epoch_average_meters["loss/{}/surrogate".format(
                    phase)].update(sur_loss)
                for p_n, p_v in zip(self._perf_names, results[1:]):
                    self.epoch_average_meters["{}/{}/surrogate".format(
                        p_n, phase)].update(p_v)
            return func()

    def _get_hiddens_resetter(self, name):
        def _func():
            getattr(self, name + "_hiddens").zero_()

        _func.__name__ = "{}_hiddens_resetter".format(name)
        return _func

    def _reset_hidden(self):  # not used now
        if self._data_type == "image":
            return
        # reset the hidden states
        [func() for func in self.hiddens_resetter]

    def _init_criterions(self, rollout_type):
        # criterion and forward keyword arguments for evaluating rollout in `evaluate_rollout`

        # support compare rollout
        if rollout_type == "compare":
            # init criterions according to weights manager's rollout type
            rollout_type = self.weights_manager.rollout_type

        if "differentiable" in rollout_type:
            # NOTE: only handle differentiable rollout differently
            self._reward_func = partial(self.objective.get_loss,
                                        add_controller_regularization=True,
                                        add_evaluator_regularization=False)
            self._reward_kwargs = {"detach_arch": False}
            self._scalar_reward_func = lambda *args, **kwargs: \
                utils.get_numpy(self._reward_func(*args, **kwargs))
        else:  # rollout_type in {"discrete", "ofa"} and outer-registered supported rollout types
            self._reward_func = self.objective.get_reward
            self._reward_kwargs = {}
            self._scalar_reward_func = self._reward_func

        self._perf_names = self.objective.perf_names()
        self._all_perf_names = utils.flatten_list(
            ["reward", "loss", self._perf_names])
        self._all_perf_names_update_evaluator = utils.flatten_list(
            ["loss", "reward", self._perf_names])
        # criterion funcs for meta parameter training
        self._mepa_loss_func = partial(self.objective.get_loss,
                                       add_controller_regularization=False,
                                       add_evaluator_regularization=True)
        # criterion funcs for log/report
        self._report_perf_funcs = [
            self.objective.get_perfs
        ]
        self._report_loss_funcs = [
            partial(self.objective.get_loss_item,
                    add_controller_regularization=False,
                    add_evaluator_regularization=False),
            self.objective.get_perfs
        ]
        self._criterions_related_attrs = [
            "_reward_func", "_reward_kwargs", "_scalar_reward_func",
            "_reward_kwargs", "_perf_names", "_mepa_loss_func",
            "_report_perf_funcs", "_report_loss_funcs"
        ]

    def _init_data_queues_and_hidden(self, data_type, data_portion,
                                     mepa_as_surrogate):
        self._dataset_related_attrs = []
        if data_type == "image":
            queue_cfgs = [{
                "split": p[0] if isinstance(p, (list, tuple)) else "train",
                "portion": p[1] if isinstance(p, (list, tuple)) else p,
                "kwargs":
                p[2] if isinstance(p, (list, tuple)) and len(p) > 2 else {},
                "batch_size": self.batch_size # this can be override by p[2]
            } for p in data_portion]
            # image data, do not need to record hidden for each data queue
            self.s_hid_kwargs = {}
            self.c_hid_kwargs = {}
            self.m_hid_kwargs = {}
            self.d_hid_kwargs = {}
        else:  # "sequence"
            # initialize hidden
            self.surrogate_hiddens = self.weights_manager.init_hidden(
                self.batch_size)
            self.mepa_hiddens = self.weights_manager.init_hidden(
                self.batch_size)
            self.controller_hiddens = self.weights_manager.init_hidden(
                self.batch_size)
            self.derive_hiddens = self.weights_manager.init_hidden(
                self.batch_size)

            self.hiddens_resetter = [
                self._get_hiddens_resetter(n)
                for n in ["surrogate", "mepa", "controller", "derive"]
            ]
            queue_cfgs = []
            for callback, portion in zip(self.hiddens_resetter, data_portion):
                queue_cfgs.append({
                    "split":
                    portion[0] if isinstance(portion,
                                             (list, tuple)) else "train",
                    "portion":
                    portion[1] if isinstance(portion,
                                             (list, tuple)) else portion,
                    "kwargs":
                    portion[2]
                    if isinstance(portion, (list, tuple)) and len(portion) > 2 else {},
                    "batch_size":
                    self.batch_size,
                    "bptt_steps":
                    self.bptt_steps,
                    "callback":
                    callback
                })
            self.s_hid_kwargs = {"hiddens": self.surrogate_hiddens}
            self.c_hid_kwargs = {"hiddens": self.controller_hiddens}
            self.m_hid_kwargs = {"hiddens": self.mepa_hiddens}
            self.d_hid_kwargs = {"hiddens": self.derive_hiddens}
            self._dataset_related_attrs += [
                "surrogate_hiddens", "mepa_hiddens", "controller_hiddens", "derive_hiddens"
            ]

        if len(queue_cfgs) == 3:
            self.logger.warn(
                "We suggest explictly specifiying the configuration for "
                "derive_queue. For example, by adding a 4th item \"- "
                "[train_testTransform, [0.8, 1.0], {shuffle: false}]\""
                "into the `data_portion` configuration field")
            # (Backward cfg compatability) if configuration for derive_queue is not specified,
            # use controller queue, and if controller queue split name is train,
            # will try add "_testTransform" suffix
            derive_queue_cfg = copy.deepcopy(queue_cfgs[-1])
            cont_queue_split = queue_cfgs[-1]["split"]
            if isinstance(queue_cfgs[-1]["portion"], float):
                # calculate derive portion as a range
                start_portion = sum(
                    [queue_cfg["portion"] for queue_cfg in queue_cfgs[:-1]
                     if isinstance(queue_cfg["portion"], float) and \
                     queue_cfg["split"] == cont_queue_split])
                derive_portion = [start_portion, queue_cfgs[-1]["portion"] + start_portion]
                derive_queue_cfg["portion"] = derive_portion
            derive_queue_cfg["kwargs"]["shuffle"] = False # do not shuffle
            # do not use distributed sampler
            derive_queue_cfg["kwargs"]["no_distributed_sampler"] = False

            if cont_queue_split == "train":
                # try change split "train" to "train_testTransform"
                if "train_testTransform" in self.dataset.splits():
                    derive_queue_cfg["split"] = "train_testTransform"
            self.logger.info("The configuration for derive_queue is not specified, "
                             "will use split `%s` and set `shuffle=False`/`multiprocess=False`,"
                                 " other cfgs is the same as those of controller_queue.",
                             derive_queue_cfg["split"])
            queue_cfgs.append(derive_queue_cfg)

        self.logger.info("Data queue configurations:\n\t%s", "\n\t".join(
            ["{}: {}".format(queue_name, queue_cfg) for queue_name, queue_cfg in zip(
                ["surrogate", "mepa", "controller", "derive"], queue_cfgs)]))
        self.surrogate_queue, self.mepa_queue, self.controller_queue, self.derive_queue\
            = utils.prepare_data_queues(self.dataset, queue_cfgs,
                                        data_type=self._data_type,
                                        drop_last=self.rollout_batch_size > 1,
                                        shuffle=self.shuffle_data_before_split,
                                        shuffle_seed=self.shuffle_data_before_split_seed,
                                        num_workers=self.workers_per_queue,
                                        pin_memory=self.pin_memory_per_queue,
                                        multiprocess=self.multiprocess,
                                        shuffle_indice_file=self.shuffle_indice_file)

        if mepa_as_surrogate:
            # use mepa data queue as surrogate data queue
            self.surrogate_queue = self.mepa_queue
        # len_surrogate = len(self.surrogate_queue) * self.surrogate_queue.batch_size \
        #                 if self.surrogate_queue else 0
        # self.logger.info(
        #     "Data sizes: surrogate: %s; controller: %d; mepa: %d; derive: %d",
        #     str(len_surrogate) if not mepa_as_surrogate else "(mepa queue)",
        #     len(self.controller_queue) * self.controller_queue.batch_size,
        #     len(self.mepa_queue) * self.mepa_queue.batch_size,
        #     len(self.derive_queue) * self.derive_queue.batch_size)
        self._dataset_related_attrs += [
            "surrogate_queue", "mepa_queue", "controller_queue", "derive_queue",
            "s_hid_kwargs", "c_hid_kwargs", "m_hid_kwargs", "d_hid_kwargs"
        ]

    def _eval_reward_func(self, data, cand_net, criterions, rollout, callback,
                          kwargs):
        res = cand_net.eval_data(data,
                                 criterions=criterions,
                                 mode="train",
                                 **kwargs)
        rollout.set_perfs(OrderedDict(zip(self._all_perf_names, res)))
        if callback is not None:
            callback(rollout)
        # set reward to be the scalar
        rollout.set_perf(utils.get_numpy(rollout.get_perf(name="reward")))
        return res

    def _scheduler_step(self, step, log=False):
        lr_str = ""
        if self.mepa_scheduler is not None:
            self.mepa_scheduler.step(step)
            lr_str += "mepa LR: {:.5f}; ".format(
                self.mepa_optimizer.param_groups[0]['lr'])
        if self.surrogate_scheduler is not None:
            self.surrogate_scheduler.step(step)
            lr_str += "surrogate LR: {:.5f};".format(
                self.surrogate_scheduler.get_lr()[0])
        if self.surrogate_lr_scheduler is not None:
            self.surrogate_lr_scheduler.step(step)
            lr_str += "surrogate lr LR: {:.5f};".format(
                self.surrogate_lr_scheduler.get_lr()[0])
        if log and lr_str:
            self.logger.info("Schedule step %3d: %s", step, lr_str)

    def set_dataset(self, dataset):
        self.dataset = dataset
        self._data_type = self.dataset.data_type()
        if self.multiprocess:
            self.logger.warning(
                "When loading a multiprocess evaluator from a pickle file, "
                "if no process group is initialized, "
                "the data queues cannot be initialized properly, evaluator methods might not work. "
                "However, `evaluator.weights_manager` can be used without problem.")
        else:
            self._init_data_queues_and_hidden(
                self._data_type, self.data_portion, self.mepa_as_surrogate)

    def __setstate__(self, state):
        super(MepaEvaluator, self).__setstate__(state)
        self._init_criterions(self.rollout_type)
        if not hasattr(self, "dataset"):
            self.logger.warning(
                "After load the evaluator from a pickle file, the dataset does not "
                "get loaded automatically, initialize a dataset and call "
                "`set_dataset(dataset)` ")
        else:
            self.set_dataset(self.dataset)

    def __getstate__(self):
        state = super(MepaEvaluator, self).__getstate__()
        if not ((sys.version_info.major >= 3
                 and hasattr(self.dataset, "__reduce__")) or
                (sys.version_info.major == 2
                 and hasattr(self.dataset, "__getinitargs__"))):
            # if dataset has `__getinitargs__` special method defined,
            # we expect a correct and efficient unpickling of this dataset is handled
            # and do not del dataset attribute
            del state["dataset"]
            # dataset can be too large, by default we do not serialize it

            # TODO: load large dataset from disk for multiple times is time- and memory-consuming,
            # can we use multiprocessing shared memory for datasets?

        for attr_name in self._dataset_related_attrs + self._criterions_related_attrs:
            if attr_name in state:
                del state[attr_name]
        return state
