# -*- coding: utf-8 -*-
"""Base class definition of Trainer"""

import os
import abc
import pickle

import torch
import torch.utils.data

from aw_nas import Component
from aw_nas import utils
from aw_nas.utils.exception import expect, ConfigException


class BaseTrainer(Component):
    REGISTRY = "trainer"

    def __init__(self, controller, evaluator, rollout_type, schedule_cfg=None):
        super(BaseTrainer, self).__init__(schedule_cfg)

        self.controller = controller
        self.evaluator = evaluator
        expect(rollout_type in self.all_supported_rollout_types(),
               "Unsupported `rollout_type`: {}".format(rollout_type),
               ConfigException) # supported rollout types
        self.rollout_type = rollout_type

        self.is_setup = False
        self.epoch = 1
        self.save_every = None
        self.save_controller_every = None
        self.train_dir = None
        self.interleave_report_every = None

    @classmethod
    def all_supported_rollout_types(cls):
        return cls.registered_supported_rollouts_() + cls.supported_rollout_types()

    # ---- virtual APIs to be implemented in subclasses ----
    @utils.abstractclassmethod
    def supported_rollout_types(cls):
        pass

    @abc.abstractmethod
    def train(self):
        """
        Do the actual searching task of your trainer.
        """

    @abc.abstractmethod
    def test(self):
        """
        Derive an arch and give its performance on test dataset.

        .. note::
        This is the performance of an architecture Without training from scratch.
        Basically, we hope the test performance of an arch can be a surrogate for
        the true performance when it's carefully tuned from scratch.
        """

    @abc.abstractmethod
    def derive(self, n, steps=None, out_file=None):
        """
        Derive some architectures.
        """

    @abc.abstractmethod
    def save(self, path):
        """
        Save the trainer state to disk.
        """

    @abc.abstractmethod
    def load(self, path):
        """
        Load the trainer state from disk.
        """

    # ---- some helper methods ----
    def on_epoch_start(self, epoch):
        """
        Default implementation: dispatch `on_epoch_start` call to each sub-components.
        """
        super(BaseTrainer, self).on_epoch_start(epoch)
        self.controller.on_epoch_start(epoch)
        self.evaluator.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        """
        Default implementation: dispatch `on_epoch_end` call to each sub-components.
        """
        super(BaseTrainer, self).on_epoch_end(epoch)
        self.controller.on_epoch_end(epoch)
        self.evaluator.on_epoch_end(epoch)

    def final_save(self):
        """
        Pickle dump the controller/evaluator directly. Usually, evaluator use dataset,
        it will failed to be pickled if no handling is specified using `__getsate__`,
        in that case, will fallback to call `evaluator.save`.

        The dumped checkpoint can be loaded directly using `model = torch.load(checkpoint)`,
        without instantiate the correct class with correct configuration first.
        This checkpoint is convenient for test/usage.

        Visualization writer is not kept after save/load, so take care when these checkpoints
        are used in the middle of training process that has visualization writer. Better using
        the checkpoints dumped by `maybe_save` when finetuning.
        """
        if self.train_dir:
            # final saving
            dir_ = utils.makedir(os.path.join(self.train_dir, "final"))
            rank = os.environ.get("LOCAL_RANK")
            if rank is None or rank == '0':
                try:
                    torch.save(self.controller, os.path.join(dir_, "controller.pt"))
                except pickle.PicklingError as e:
                    self.logger.warning("Final saving: torch.save(controller) fail, "
                                        "fallback to call `controller.save`: %s", e)
                    self.controller.save(os.path.join(dir_, "controller"))
                try:
                    torch.save(self.evaluator, os.path.join(dir_, "evaluator.pt"))
                except pickle.PicklingError as e:
                    self.logger.warning("Final saving: torch.save(evaluator) fail, "
                                        "fallback to call `evaluator.save`: %s", e)
                    self.evaluator.save(os.path.join(dir_, "evaluator.pt"))
                self.logger.info("Final Saving: Dump controller/evaluator to directory %s", dir_)

    def maybe_save(self):
        rank = os.environ.get("LOCAL_RANK")
        if self.save_every is not None and self.train_dir:
            assert self.save_controller_every is not None, \
                ("when save-every is not None, save-controller every should also not be None,"
                 " since its default value should be save-every")
            if self.epoch % self.save_every == 0:
                if rank is None or rank == "0":
                    self.controller.save(self._save_path("controller"))
                    self.evaluator.save(self._save_path("evaluator"))
                    self.save(self._save_path("trainer"))
                    self.logger.info("Epoch %3d: Save all checkpoints to directory %s",
                                    self.epoch, self._save_path())
            elif self.epoch % self.save_controller_every == 0:
                self.controller.save(self._save_path("controller"))
                self.logger.info("Epoch %3d: Save Controller to directory %s",
                                 self.epoch, self._save_path())

    def _save_path(self, name=""):
        if self.train_dir is None:
            return None
        dir_ = utils.makedir(os.path.join(self.train_dir, str(self.epoch)))
        return os.path.join(dir_, name)

    def setup(self, load=None, save_every=None, save_controller_every=None,train_dir=None, writer=None, load_components=None,
              interleave_report_every=None):
        """
        Setup the scaffold: saving/loading/visualization settings.
        """
        if load is not None:
            all_components = ("controller", "evaluator", "trainer")
            load_components = all_components\
                              if load_components is None else load_components
            expect(set(load_components).issubset(all_components), "Invalid `load_components`")

            if "controller" in load_components:
                path = os.path.join(load, "controller")
                self.logger.info("Load controller from %s", path)
                try:
                    self.controller.load(path)
                except Exception as e:
                    self.logger.error("Controller not loaded! %s", e)
            if "evaluator" in load_components:
                path = os.path.join(load, "evaluator")
                # if os.path.exists(path):
                self.logger.info("Load evaluator from %s", path)
                try:
                    self.evaluator.load(path)
                except Exception as e:
                    self.logger.error("Evaluator not loaded: %s", e)
            if "trainer" in load_components:
                path = os.path.join(load, "trainer")
                # if os.path.exists(path):
                self.logger.info("Load trainer from %s", path)
                try:
                    self.load(path)
                except Exception as e:
                    self.logger.error("Trainer not loaded: %s", e)

        self.save_every = save_every
        self.save_controller_every = save_controller_every
        self.train_dir = utils.makedir(train_dir) if train_dir is not None else train_dir
        if writer is not None:
            self.setup_writer(writer.get_sub_writer("trainer"))
            self.controller.setup_writer(writer.get_sub_writer("controller"))
            self.evaluator.setup_writer(writer.get_sub_writer("evaluator"))
        self.interleave_report_every = interleave_report_every
        self.is_setup = True
