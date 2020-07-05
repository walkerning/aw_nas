# -*- coding: utf-8 -*-

import abc

from aw_nas import Component, utils
from aw_nas.utils.exception import expect, ConfigException

class BaseEvaluator(Component):
    REGISTRY = "evaluator"

    def __init__(self, dataset, weights_manager, objective, rollout_type, schedule_cfg=None):
        super(BaseEvaluator, self).__init__(schedule_cfg)

        self.dataset = dataset
        self.weights_manager = weights_manager
        self.objective = objective
        expect(rollout_type in self.all_supported_rollout_types(),
               "Unsupported `rollout_type`: {}".format(rollout_type),
               ConfigException) # supported rollout types
        self.rollout_type = rollout_type

    @classmethod
    def all_supported_rollout_types(cls):
        return cls.registered_supported_rollouts_() + cls.supported_rollout_types()

    def set_device(self, device):
        if self.weights_manager is not None:
            self.weights_manager.set_device(device)

    @utils.abstractclassmethod
    def supported_data_types(cls):
        """
        Return the supported data types.
        """

    @utils.abstractclassmethod
    def supported_rollout_types(cls):
        """
        Return the supported rollout types.
        """

    def suggested_controller_steps_per_epoch(self): #pylint: disable=invalid-name,no-self-use
        """
        Return the suggested controller steps per epoch for trainer, usually, it will be calculated
        based on data queue length. If your evaluator has no suggestions, just return None.
        """
        return None

    def suggested_evaluator_steps_per_epoch(self): #pylint: disable=invalid-name,no-self-use
        """
        Return the suggested evaluator steps per epoch for trainer, usually, it will be calculated
        based on data queue length. If your evaluator has no suggestions, just return None.
        """
        return None

    @abc.abstractmethod
    def evaluate_rollouts(self, rollouts, is_training, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        """
        Evaluate the reward for these rollouts, and save the reward in the rollout object.

        Args:
            rollout (aw_nas.Rollout): the rollout object.
            is_training (bool): Indicate whether this evaluation is for training or testing.
            portion (float): If specified, can tell the to evaluate this rollout on only this
               portion of data, usually for quick performance glance.
            eval_batches (int): If specified, will ignore `portion`, and only evaluate this rollout
               on these batches of data. (This depends on the acutal implementation of evaluator,
               the evaluator can choose to ignore these arguments. See their documentations.)
            return_candidate_net (bool): If true, the returned rollouts will have its corresponding
                candidate network set.
        """
        return rollouts

    @abc.abstractmethod
    def update_rollouts(self, rollouts):
        """
        Update evaluator rollouts population.
        """

    @abc.abstractmethod
    def update_evaluator(self, controller):
        """
        Do evaluator updates for `steps`. Return an ordered dict of {stat_name: stat_value},
        these stats will be averaged in the trainer, and print out after each epoch.

        Can use the controller to sample new arch for training/updating the evaluator.
        """

    @abc.abstractmethod
    def save(self, path):
        """
        Save the state of the evaluator to `path` on disk.
        """

    @abc.abstractmethod
    def load(self, path):
        """
        Load the state of the evaluator from `path` on disk.
        """
