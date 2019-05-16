# -*- coding: utf-8 -*-
"""Base class definition of Controller"""

import abc
import contextlib

from aw_nas import Component

class BaseController(Component):
    REGISTRY = "controller"

    def __init__(self, search_space, mode="eval", schedule_cfg=None):
        super(BaseController, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.mode = mode

    @contextlib.contextmanager
    def begin_mode(self, mode):
        prev_mode = self.mode
        self.set_mode(mode)

        yield

        self.set_mode(prev_mode)

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
    def summary(self, rollouts, prefix="", step=None):
        """Summary the information in these rollouts.

        Use self.logger to log theinformation.
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
