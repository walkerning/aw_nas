# -*- coding: utf-8 -*-
"""Base class definition of Controller"""

import abc

from aw_nas import Component

class BaseController(Component):
    REGISTRY = "controller"

    def __init__(self, search_space, schedule_cfg=None):
        super(BaseController, self).__init__(schedule_cfg)

        self.search_space = search_space

    @abc.abstractmethod
    def set_mode(self, mode):
        """Set the mode of the controller"""

    @abc.abstractmethod
    def sample(self, n):
        """Sample a architecture rollout, which can be used to assemble an architecture.

        Returns:
            Rollout
        """

    @abc.abstractmethod
    def step(self, rollouts, optimizer):
        """Update the controller state using rollouts, which are already evaluated by the Evaluator.

        Args:
            rollouts (list of Rollout)
        """

    @abc.abstractmethod
    def save(self, path):
        """
        Save the state of the controller to `path` on disk.
        """

    @abc.abstractmethod
    def load(self, path):
        """
        Load the state of the controller from `path` on disk.
        """
