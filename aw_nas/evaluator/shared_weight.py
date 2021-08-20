# -*- coding: utf-8 -*-

import sys
import abc
import copy
from functools import partial
from collections import defaultdict, OrderedDict

import six
import numpy as np
import torch

from aw_nas import utils
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.evaluator.mepa import _patch_dropout_forward

__all__ = ["SharedweightEvaluator", "DiscreteSharedweightEvaluator", "DifferentiableEvaluator"]


class SharedweightEvaluator(
    BaseEvaluator
):  # pylint: disable=too-many-instance-attributes
    """
    A simplified version of shared weights evaluators from `MepaEvaluator`.
    The major difference is that the codes about running surrogate steps (MEPA) are all deleted.

    Methods:
    The two most important methods are `evaluate_rollouts` and `update_evaluator`.
    * evaluate_rollouts: Evaluate rollouts on the validation data split,
        controller queue (usually used when training the controller),
        or derive queue (usually used when derive or test architectures,
                         no random augmentation is used by default).
    * update_evaluator: Update evaluator, or more precisely, the weights manager (supernet),
                        on the training data split (evaluator queue).

    Data queues:
    The evaluator has 3 data queues: the controller/eval/derive queues.
    And one should specificy 2 or 3 items in the `data_portion` configuration.
    By default, if the 3rd item for derive queue is not specified in the `data_portion` list,
    the derive queue uses the same data as the controller queue without random augmentation.
    * The shared weights are optimized on the eval_queue in `update_evaluator()` (train split).
    * The controller's update is inside trainer (not evaluator), and `evaluate_rollouts()` is called
      on the controller_queue (valid split).

    Args:
        * eval_optimizer: The optimizer to update the shared weights.
        * eval_scheduler: The learning rate scheduler.
        * eval_samples: Number of architecture samples when update the shared weights (default 1).
        * data_portion (list of 2/3 items):
          Configuration for controller/evaluator/[optional] derive queues.
          Each item can either be a floating-point number that indicate the portion of
          the training set that is used for this queue, or a list of items:
          [split, portion, kwargs].
          * split: "train" by default, should be one of the keys in the `dataset.splits()` dict
          * portion: In the latter case, the portion could be either a floating-point number or
                     a two-element range.
          * kwargs: Additional keyword arguments that will be passed to DataLoader
    """

    SCHEDULABLE_ATTRS = [
        "eval_samples",
    ]

    def __init__(
        self,
        dataset,
        weights_manager,
        objective,
        rollout_type="discrete",
        batch_size=128,
        eval_optimizer={
            "type": "SGD",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        eval_scheduler={
            "type": "CosineWithRestarts",
            "t_0": 10,
            "eta_min": 0.0001,
            "factor": 2.0,
        },
        schedule_every_batch=False,
        load_optimizer=True,
        load_scheduler=True,
        strict_load_weights_manager=True,
        eval_samples=1,
        disable_step_current=False,
        evaluate_with_whole_queue=False,
        data_portion=(0.5, 0.5),
        shuffle_data_before_split=False,  # by default not shuffle data before train-val split
        shuffle_indice_file=None,
        shuffle_data_before_split_seed=None,
        workers_per_queue=2,
        update_evaluator_report_perfs=True,
        # only work for differentiable controller now
        rollout_batch_size=1,
        # only for rnn data
        bptt_steps=35,
        multiprocess=False,
        schedule_cfg=None,
    ):
        super(SharedweightEvaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type, schedule_cfg
        )

        # check rollout type
        if self.rollout_type != "compare":
            expect(
                self.rollout_type == self.weights_manager.rollout_type,
                "the rollout type of evaluator/weights_manager must match, "
                "check the configuration. ({}/{})".format(
                    self.rollout_type, self.weights_manager.rollout_type
                ),
                ConfigException,
            )
        else:
            # Do not check for now
            pass

        self._data_type = self.dataset.data_type()
        self._device = self.weights_manager.device
        self.multiprocess = multiprocess

        # configs
        self.batch_size = batch_size
        self.evaluate_with_whole_queue = evaluate_with_whole_queue
        self.disable_step_current = disable_step_current
        self.data_portion = data_portion
        self.workers_per_queue = workers_per_queue
        self.shuffle_data_before_split = shuffle_data_before_split
        self.shuffle_indice_file = shuffle_indice_file
        self.shuffle_data_before_split_seed = shuffle_data_before_split_seed
        self.eval_samples = eval_samples
        self.rollout_batch_size = rollout_batch_size
        self.schedule_every_batch = schedule_every_batch
        self.load_optimizer = load_optimizer
        self.load_scheduler = load_scheduler
        self.strict_load_weights_manager = strict_load_weights_manager
        self.update_evaluator_report_perfs = update_evaluator_report_perfs

        # rnn specific configs
        self.bptt_steps = bptt_steps

        # initialize optimizers and schedulers
        # do some checks
        expect(
            len(data_portion) in {2, 3},
            "`data_portion` should have length 2/3.",
            ConfigException,
        )

        self.eval_optimizer = utils.init_optimizer(
            self.weights_manager.parameters(), eval_optimizer
        )
        self.eval_scheduler = utils.init_scheduler(self.eval_optimizer, eval_scheduler)

        # for performance when doing 1-sample ENAS in `update_evaluator`
        if not self.disable_step_current and self.eval_samples == 1:
            # Will call `step_current_gradients` of weights manager
            self.logger.info(
                "As `eval_sample==1` and `disable_step_current` is not set, "
                "to speed up, will accumulate supernet gradients in-place and call "
                "`super_net.step_current_gradients`."
            )
            self.eval_step_current = True
        else:
            self.eval_step_current = False

        # to make pylint happy, actual initialization in _init_criterions method
        self._dataset_related_attrs = []
        self._criterions_related_attrs = []
        self._all_perf_names = None
        self._all_perf_names_update_evaluator = None
        self._reward_func = None
        self._reward_kwargs = None
        self._scalar_reward_func = None
        self._reward_kwargs = None
        self._perf_names = None
        self._eval_loss_func = None
        self._report_loss_funcs = None
        self._report_perf_funcs = None

        # initialize the data queues
        self._init_data_queues_and_hidden(self._data_type, data_portion)
        # initialize reward criterions used by `get_rollout_reward`
        self._init_criterions(self.rollout_type)

        # for report loss
        self.epoch_average_meters = defaultdict(utils.AverageMeter)

        # evaluator update steps
        self.step = 0

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
        return len(self.eval_queue)

    def evaluate_rollouts(
        self,
        rollouts,
        is_training,
        portion=None,
        eval_batches=None,
        return_candidate_net=False,
        callback=None,
    ):
        """
        feed in the rollouts abd acquire the reward signal(valid loss) through self.eval_reward_func
        """

        # support CompareRollout
        if self.rollout_type == "compare":
            eval_rollouts = sum([[r.rollout_1, r.rollout_2] for r in rollouts], [])
        else:
            eval_rollouts = rollouts

        # controller_queue (for updating controller) or derive_queue (for testing rollouts only)
        data_queue = self.controller_queue if is_training else self.derive_queue
        hid_kwargs = self.c_hid_kwargs if is_training else self.d_hid_kwargs
        if not data_queue or not len(data_queue):
            return rollouts

        if is_training and not self.evaluate_with_whole_queue:
            self.objective.set_mode("train")
            # get one data batch from controller/derive queue
            cont_data = next(data_queue)
            cont_data = utils.to_device(cont_data, self.device)

            # prepare forward keyword arguments for candidate network
            _reward_kwargs = {k: v for k, v in self._reward_kwargs.items()}
            _reward_kwargs.update(hid_kwargs)

            # evaluate these rollouts on one batch of data
            for rollout in eval_rollouts:
                cand_net = self.weights_manager.assemble_candidate(rollout)
                if return_candidate_net:
                    rollout.candidate_net = cand_net
                # prepare criterions
                criterions = [self._reward_func] + self._report_loss_funcs
                criterions = [partial(func, cand_net=cand_net) for func in criterions]

                # run eval
                res = self._eval_reward_func(
                    data=cont_data,
                    cand_net=cand_net,
                    criterions=criterions,
                    kwargs=_reward_kwargs,
                    callback=callback,
                    rollout=rollout,
                )

        else:  # only for test
            self.objective.set_mode("eval")
            with _patch_dropout_forward(self.weights_manager):
                if eval_batches is not None:
                    eval_steps = eval_batches
                else:
                    eval_steps = len(data_queue)
                    if portion is not None:
                        expect(0.0 < portion < 1.0)
                        eval_steps = int(portion * eval_steps)

                for rollout in eval_rollouts:
                    cand_net = self.weights_manager.assemble_candidate(rollout)
                    if return_candidate_net:
                        rollout.candidate_net = cand_net
                    # prepare criterions
                    criterions = [self._scalar_reward_func] + self._report_loss_funcs
                    criterions = [partial(func, cand_net=cand_net) for func in criterions]

                    # NOTE: if virtual buffers, must use train mode here...
                    # if not virtual buffers(virtual parameter only), can use train/eval mode
                    aggregate_fns = [
                        self.objective.aggregate_fn(name, is_training=False)
                        for name in self._all_perf_names
                    ]
                    res = cand_net.eval_queue(
                        data_queue,
                        criterions=criterions,
                        steps=eval_steps,
                        # NOTE: In parameter-sharing evaluation, let's keep using train-mode BN!!!
                        mode="train",
                        # if test, differentiable rollout does not need to set detach_arch=True too
                        aggregate_fns=aggregate_fns,
                        **hid_kwargs
                    )

                    rollout.set_perfs(
                        OrderedDict(zip(self._all_perf_names, res))
                    )  # res is already flattend

        # support CompareRollout
        if self.rollout_type == "compare":
            num_r = len(rollouts)
            for i_rollout in range(num_r):
                better = (
                    eval_rollouts[2 * i_rollout + 1].perf["reward"]
                    > eval_rollouts[2 * i_rollout].perf["reward"]
                )
                rollouts[i_rollout].set_perfs(
                    OrderedDict([("compare_result", better),])
                )
        self.objective.set_mode("train")
        return rollouts

    def update_rollouts(self, rollouts):
        """
        Nothing to be done.
        """

    def update_evaluator(self, controller):  # pylint: disable=too-many-branches
        """
        Training meta parameter of the `weights_manager` (shared super network).
        """
        eval_data = next(self.eval_queue)
        eval_data = utils.to_device(eval_data, self.device)

        all_gradients = defaultdict(float)
        counts = defaultdict(int)
        report_stats = []

        # sample rollout
        rollouts = controller.sample(
            n=self.eval_samples, batch_size=self.rollout_batch_size
        )
        num_rollouts = len(rollouts)

        for _ind in range(num_rollouts):

            rollout = rollouts[_ind]
            # assemble candidate net
            cand_net = self.weights_manager.assemble_candidate(rollout)

            # prepare criterions
            if self.update_evaluator_report_perfs:
                # report get_perfs results
                eval_criterions = [self._scalar_reward_func] + self._report_perf_funcs
            else:
                # only report loss and reward
                eval_criterions = [self._scalar_reward_func]
            eval_criterions = [
                partial(func, cand_net=cand_net) for func in eval_criterions
            ]

            if self.eval_step_current:
                # if update in-place, here zero all the grads of weights_manager
                self.weights_manager.zero_grad()
            else:
                # otherwise, the meta parameters are updated using
                # `wm.step(gradients, optimizer)`, do not need to zero grads
                pass

            # return gradients if not update in-place
            gradients, res = cand_net.gradient(
                eval_data,
                criterion=partial(self._eval_loss_func, cand_net=cand_net),
                eval_criterions=eval_criterions,
                mode="train",
                return_grads=not self.eval_step_current,
                **self.m_hid_kwargs
            )

            if not self.eval_step_current:
                for n, g_v in gradients:
                    all_gradients[n] += g_v
                    counts[n] += 1

            # record stats of this arch
            report_stats.append(res)

        self.step += 1
        # average the gradients and update the meta parameters
        if not self.eval_step_current:
            all_gradients = {k: v / counts[k] for k, v in six.iteritems(all_gradients)}
            self.weights_manager.step(all_gradients.items(), self.eval_optimizer)
        else:
            # call step_current_gradients; eval_sample == 1
            self.weights_manager.step_current_gradients(self.eval_optimizer)

        if self.schedule_every_batch:
            # schedule learning rate every evaluator step
            self._scheduler_step(self.step)


        del all_gradients

        stats_res = OrderedDict(zip(self._all_perf_names_update_evaluator,
                                    np.mean(report_stats, axis=0)))

        if isinstance(self.eval_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.plateau_scheduler_loss.append(stats_res["loss"])
        # return stats
        return stats_res

    def on_epoch_start(self, epoch):
        super(SharedweightEvaluator, self).on_epoch_start(epoch)
        self.weights_manager.on_epoch_start(epoch)
        self.objective.on_epoch_start(epoch)


    def on_epoch_end(self, epoch):
        super(SharedweightEvaluator, self).on_epoch_end(epoch)
        self.weights_manager.on_epoch_end(epoch)
        self.objective.on_epoch_end(epoch)
        if not self.writer.is_none():
            for name, meter in six.iteritems(self.epoch_average_meters):
                if not meter.is_empty():
                    self.writer.add_scalar(name, meter.avg, epoch)

        # reset all meters
        for meter in self.epoch_average_meters.values():
            meter.reset()

        # schedule after epoch
        if not self.schedule_every_batch:
            self._scheduler_step(epoch - 1, log=True)

        else:
            self._scheduler_step(self.step, log=True)

    def save(self, path):
        optimizer_states = {}
        scheduler_states = {}
        for compo_name in ["eval"]:
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
            "schedulers": scheduler_states,
        }

        if self._data_type == "sequence":
            hidden_states = {}
            for compo_name in ["eval"]:
                # save hidden states
                hidden = getattr(self, compo_name + "_hiddens")
                hidden_states[compo_name] = hidden
            state_dict["hiddens"] = hidden_states

        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.weights_manager.load_state_dict(
            checkpoint["weights_manager"], strict=self.strict_load_weights_manager
        )

        # load hidden states if exists
        if "hiddens" in checkpoint:
            for compo_name in ["eval"]:
                getattr(self, compo_name + "_hiddens").copy_(
                    checkpoint["hiddens"][compo_name]
                )

        optimizer_states = checkpoint["optimizers"]
        scheduler_states = checkpoint["schedulers"]
        for compo_name in ["eval"]:
            optimizer = getattr(self, compo_name + "_optimizer")
            if (
                self.load_optimizer
                and optimizer is not None
                and compo_name in optimizer_states
            ):
                optimizer.load_state_dict(optimizer_states[compo_name])
            scheduler = getattr(self, compo_name + "_scheduler")
            if (
                self.load_scheduler
                and scheduler is not None
                and compo_name in scheduler_states
            ):
                scheduler.load_state_dict(scheduler_states[compo_name])

        # call `on_epoch_start` for scheduled values
        self.step = checkpoint["step"]
        self.on_epoch_start(checkpoint["epoch"])

    @staticmethod
    def _candnet_perf_use_param(
        cand_net, params, data, loss_criterion, forward_kwargs, return_outputs=False
    ):
        # TODO: only support cnn now. because directly use utils.accuracy
        #      should use self._report_loss_funcs instead maybe
        outputs = cand_net.forward_with_params(
            data[0], params=params, **forward_kwargs, mode="train"
        )
        loss = loss_criterion(data[0], outputs, data[1])
        acc = utils.accuracy(outputs, data[1], topk=(1,))[0]
        if return_outputs:
            return outputs, loss, acc
        return loss, acc

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

    @abc.abstractmethod
    def _init_criterions(self, rollout_type):
        """for complete function implemention this function in `MepaEvaluator` """
        pass

    def _init_data_queues_and_hidden(self, data_type, data_portion):
        if data_type == "image":
            queue_cfgs = [
                {
                    "split": p[0] if isinstance(p, (list, tuple)) else "train",
                    "portion": p[1] if isinstance(p, (list, tuple)) else p,
                    "kwargs": p[2]
                    if isinstance(p, (list, tuple)) and len(p) > 2
                    else {},
                    "batch_size": self.batch_size,  # this can be override by p[2]
                }
                for p in data_portion
            ]
            # image data, do not need to record hidden for each data queue
            self.c_hid_kwargs = {}
            self.m_hid_kwargs = {}
            self.d_hid_kwargs = {}
        else:  # "sequence"
            # initialize hidden
            self.eval_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.controller_hiddens = self.weights_manager.init_hidden(self.batch_size)
            self.derive_hiddens = self.weights_manager.init_hidden(self.batch_size)

            self.hiddens_resetter = [
                self._get_hiddens_resetter(n) for n in ["eval", "controller", "derive"]
            ]
            queue_cfgs = []
            for callback, portion in zip(self.hiddens_resetter, data_portion):
                queue_cfgs.append(
                    {
                        "split": portion[0]
                        if isinstance(portion, (list, tuple))
                        else "train",
                        "portion": portion[1]
                        if isinstance(portion, (list, tuple))
                        else portion,
                        "kwargs": portion[2]
                        if isinstance(portion, (list, tuple)) and len(portion) > 2
                        else {},
                        "batch_size": self.batch_size,
                        "bptt_steps": self.bptt_steps,
                        "callback": callback,
                    }
                )
            self.c_hid_kwargs = {"hiddens": self.controller_hiddens}
            self.m_hid_kwargs = {"hiddens": self.eval_hiddens}
            self.d_hid_kwargs = {"hiddens": self.derive_hiddens}
            self._dataset_related_attrs += [
                "eval_hiddens",
                "controller_hiddens",
                "derive_hiddens",
            ]

        if len(queue_cfgs) == 2:
            self.logger.warn(
                "We suggest explicitly specifying the configuration for "
                'derive_queue. For example, by adding a 3rd item "- '
                '[train_testTransform, [0.8, 1.0], {shuffle: false}]"'
                "into the `data_portion` configuration field"
            )
            # (Backward cfg compatability) if configuration for derive_queue is not specified,
            # use controller queue, and if controller queue split name is train,
            # will try add "_testTransform" suffix
            derive_queue_cfg = copy.deepcopy(queue_cfgs[-1])
            cont_queue_split = queue_cfgs[-1]["split"]
            if isinstance(queue_cfgs[-1]["portion"], float):
                # calculate derive portion as a range
                start_portion = sum(
                    [
                        queue_cfg["portion"]
                        for queue_cfg in queue_cfgs[:-1]
                        if isinstance(queue_cfg["portion"], float)
                        and queue_cfg["split"] == cont_queue_split
                    ]
                )
                derive_portion = [
                    start_portion,
                    queue_cfgs[-1]["portion"] + start_portion,
                ]
                derive_queue_cfg["portion"] = derive_portion
            derive_queue_cfg["kwargs"]["shuffle"] = False  # do not shuffle
            # do not use distributed sampler
            derive_queue_cfg["kwargs"]["no_distributed_sampler"] = False

            if cont_queue_split == "train":
                # try change split "train" to "train_testTransform"
                if "train_testTransform" in self.dataset.splits():
                    derive_queue_cfg["split"] = "train_testTransform"
            self.logger.info(
                "The configuration for derive_queue is not specified, "
                "will use split `%s` and set `shuffle=False`/`multiprocess=False`,"
                " other cfgs is the same as those of controller_queue.",
                derive_queue_cfg["split"],
            )
            queue_cfgs.append(derive_queue_cfg)

        self.logger.info(
            "Data queue configurations:\n\t%s",
            "\n\t".join(
                [
                    "{}: {}".format(queue_name, queue_cfg)
                    for queue_name, queue_cfg in zip(
                        ["eval", "controller", "derive"], queue_cfgs
                    )
                ]
            ),
        )
        (
            self.eval_queue,
            self.controller_queue,
            self.derive_queue,
        ) = utils.prepare_data_queues(
            self.dataset,
            queue_cfgs,
            data_type=self._data_type,
            drop_last=self.rollout_batch_size > 1,
            shuffle=self.shuffle_data_before_split,
            shuffle_seed=self.shuffle_data_before_split_seed,
            num_workers=self.workers_per_queue,
            multiprocess=self.multiprocess,
            shuffle_indice_file=self.shuffle_indice_file,
        )

        self.logger.info(
            "Data sizes: controller: %d; eval: %d; derive: %d",
            len(self.controller_queue) * self.controller_queue.batch_size,
            len(self.eval_queue) * self.eval_queue.batch_size,
            len(self.derive_queue) * self.derive_queue.batch_size,
        )
        self._dataset_related_attrs += [
            "eval_queue",
            "controller_queue",
            "derive_queue",
            "s_hid_kwargs",
            "c_hid_kwargs",
            "m_hid_kwargs",
            "d_hid_kwargs",
        ]

    def _eval_reward_func(self, data, cand_net, criterions, rollout, callback, kwargs):
        res = cand_net.eval_data(data, criterions=criterions, mode="train", **kwargs)
        rollout.set_perfs(OrderedDict(zip(self._all_perf_names, res)))
        if callback is not None:
            callback(rollout)
        # set reward to be the scalar
        rollout.set_perf(utils.get_numpy(rollout.get_perf(name="reward")))
        return res

    def _scheduler_step(self, step, log=False):
        lr_str = ""
        if self.eval_scheduler is not None:
            # self.eval_scheduler.step(step)
            self.eval_scheduler.step()
            lr_str += "eval LR: {:.5f}; ".format(
                self.eval_optimizer.param_groups[0]["lr"]
            )
        if log and lr_str:
            self.logger.info("Schedule step %3d: %s", step + 1, lr_str)

    def set_dataset(self, dataset):
        self.dataset = dataset
        self._data_type = self.dataset.data_type()
        self._init_data_queues_and_hidden(self._data_type, self.data_portion)

    def __setstate__(self, state):
        super(SharedweightEvaluator, self).__setstate__(state)
        self._init_criterions(self.rollout_type)
        if not hasattr(self, "dataset"):
            self.logger.warning(
                "After load the evaluator from a pickle file, the dataset does not "
                "get loaded automatically, initialize a dataset and call "
                "`set_dataset(dataset)` "
            )
        else:
            self.set_dataset(self.dataset)

    def __getstate__(self):
        state = super(SharedweightEvaluator, self).__getstate__()
        if not (
            (sys.version_info.major >= 3 and hasattr(self.dataset, "__reduce__"))
            or (
                sys.version_info.major == 2 and hasattr(self.dataset, "__getinitargs__")
            )
        ):
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


class DiscreteSharedweightEvaluator(SharedweightEvaluator):
    NAME = "discrete_shared_weights"

    def _init_criterions(self, rollout_type):
        # criterion and forward keyword arguments for evaluating rollout in `evaluate_rollout`

        # support compare rollout
        if rollout_type == "compare":
            # init criterions according to weights manager's rollout type
            rollout_type = self.weights_manager.rollout_type

        self._reward_func = self.objective.get_reward
        self._reward_kwargs = {}
        self._scalar_reward_func = self._reward_func

        self._perf_names = self.objective.perf_names()
        self._all_perf_names = utils.flatten_list(["reward", "loss", self._perf_names])
        self._all_perf_names_update_evaluator = utils.flatten_list(
            ["loss", "reward", self._perf_names])
        # criterion funcs for meta parameter training
        self._eval_loss_func = partial(
            self.objective.get_loss,
            add_controller_regularization=False,
            add_evaluator_regularization=True,
        )
        # criterion funcs for log/report
        self._report_perf_funcs = [
            self.objective.get_perfs
        ]
        self._report_loss_funcs = [
            partial(
                self.objective.get_loss_item,
                add_controller_regularization=False,
                add_evaluator_regularization=False,
            ),
            self.objective.get_perfs,
        ]
        self._criterions_related_attrs = [
            "_reward_func",
            "_reward_kwargs",
            "_scalar_reward_func",
            "_reward_kwargs",
            "_perf_names",
            "_eval_loss_func",
            "_report_loss_funcs",
        ]


class DifferentiableEvaluator(SharedweightEvaluator):
    NAME = "differentiable_shared_weights"

    def _init_criterions(self, rollout_type):
        # criterion and forward keyword arguments for evaluating rollout in `evaluate_rollout`

        # support compare rollout
        assert "differentiable" in rollout_type

        # NOTE: only handle differentiable rollout differently
        self._reward_func = partial(
            self.objective.get_loss,
            add_controller_regularization=True,
            add_evaluator_regularization=False,
        )
        self._reward_kwargs = {"detach_arch": False}
        self._scalar_reward_func = lambda *args, **kwargs: utils.get_numpy(
            self._reward_func(*args, **kwargs)
        )

        self._perf_names = self.objective.perf_names()
        self._all_perf_names = utils.flatten_list(["reward", "loss", self._perf_names])
        self._all_perf_names_update_evaluator = utils.flatten_list(
            ["loss", "reward", self._perf_names])
        # criterion funcs for meta parameter training
        self._eval_loss_func = partial(
            self.objective.get_loss,
            add_controller_regularization=False,
            add_evaluator_regularization=True,
        )
        # criterion funcs for log/report
        self._report_perf_funcs = [
            self.objective.get_perfs
        ]
        self._report_loss_funcs = [
            partial(
                self.objective.get_loss_item,
                add_controller_regularization=False,
                add_evaluator_regularization=False,
            ),
            self.objective.get_perfs,
        ]
        self._criterions_related_attrs = [
            "_reward_func",
            "_reward_kwargs",
            "_scalar_reward_func",
            "_reward_kwargs",
            "_perf_names",
            "_eval_loss_func",
            "_report_perf_funcs",
            "_report_loss_funcs",
        ]
