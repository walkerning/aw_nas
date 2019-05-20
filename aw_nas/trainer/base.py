# -*- coding: utf-8 -*-
"""Base class definition of Trainer"""

import os
import abc
import six

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from aw_nas import Component
from aw_nas import utils


class BaseTrainer(Component):
    REGISTRY = "trainer"

    def __init__(self, controller, weights_manager, dataset, schedule_cfg=None):
        super(BaseTrainer, self).__init__(schedule_cfg)

        self.controller = controller
        self.weights_manager = weights_manager
        self.dataset = dataset

        self.is_setup = False
        self.epoch = 1
        self.save_every = None
        self.train_dir = None

    # ---- virtual APIs to be implemented in subclasses ----
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
    def derive(self, n):
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

    def on_epoch_start(self, epoch):
        """
        Default implementaion: dispatch `on_epoch_start` call to each sub-components.
        """
        super(BaseTrainer, self).on_epoch_start(epoch)
        self.controller.on_epoch_start(epoch)
        self.weights_manager.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        """
        Default implementaion: dispatch `on_epoch_end` call to each sub-components.
        """
        super(BaseTrainer, self).on_epoch_end(epoch)
        self.controller.on_epoch_end(epoch)
        self.weights_manager.on_epoch_end(epoch)

    # ---- some helper methods ----
    def maybe_save(self):
        if self.save_every is not None and self.train_dir and self.epoch % self.save_every == 0:
            self.controller.save(self._save_path("controller"))
            self.weights_manager.save(self._save_path("weights_manager"))
            self.save(self._save_path("trainer"))
            self.logger.info("Epoch %3d: Save all checkpoints to directory %s",
                             self.epoch, self._save_path())

    def _save_path(self, name=""):
        if self.train_dir is None:
            return None
        dir_ = utils.makedir(os.path.join(self.train_dir, str(self.epoch)))
        return os.path.join(dir_, name)

    def setup(self, load=None, save_every=None, train_dir=None, writer=None, load_components=None):
        """
        Setup the scaffold: saving/loading/visualization settings.
        """
        if load is not None:
            all_components = ("controller", "weights_manager", "trainer")
            load_components = all_components\
                              if load_components is None else load_components
            assert set(load_components).issubset(all_components)

            if "controller" in load_components:
                path = os.path.join(load, "controller")
                self.logger.info("Load controller from %s", path)
                self.controller.load(path)
            if "weights_manager" in load_components:
                path = os.path.join(load, "weights_manager")
                self.logger.info("Load weights_manager from %s", path)
                self.weights_manager.load(path)
            if "trainer" in load_components:
                path = os.path.join(load, "trainer")
                self.logger.info("Load trainer from %s", path)
                self.load(path)

        self.save_every = save_every
        self.train_dir = utils.makedir(train_dir) if train_dir is not None else train_dir
        if writer is not None:
            self.setup_writer(writer.get_sub_writer("trainer"))
            self.controller.setup_writer(writer.get_sub_writer("controller"))
            self.weights_manager.setup_writer(writer.get_sub_writer("weights_manager"))
        self.is_setup = True

    def prepare_data_queues(self, splits, queue_cfg_lst):
        """
        Further partition the dataset splits, prepare different data queues.

        Example::
        @TODO: doc
        """
        dset_splits = splits
        dset_sizes = {n: len(d) for n, d in six.iteritems(dset_splits)}
        used_portions = {n: 0. for n in splits}
        queues = []
        for cfg in queue_cfg_lst: # all the queues interleave sub-dataset
            batch_size = cfg["batch_size"]
            split = cfg["split"]
            portion = cfg["portion"]

            if portion == 0:
                queues.append(None)
                continue

            used_portion = used_portions[split]
            size = dset_sizes[split]

            kwargs = {
                "batch_size": batch_size,
                "pin_memory": True,
                "num_workers": 2,
                "sampler": SubsetRandomSampler(list(range(int(size*used_portion),
                                                          int(size*(used_portion+portion)))))
            }
            queue = utils.get_inf_iterator(torch.utils.data.DataLoader(
                dset_splits[split], **kwargs))
            used_portions[split] += portion
            queues.append(queue)

        for n, portion in used_portions.items():
            if portion != 1.:
                self.logger.warn("Dataset split %s is not fully used (%.2f %%)! "
                                 "Check the `data_queues` section of the configuration.",
                                 n, portion*100)
        return queues

    def get_new_candidates(self, n):
        """
        Get new candidate networks by calling controller and weights_manager.

        Args:
            n (int): Number of candidate networks to get.

        Returns:
            List(aw_nas.Rollout): Rollouts with candidate nets generated.
        """
        rollouts = self.controller.sample(n)
        for r in rollouts:
            candidate_net = self.weights_manager.assemble_candidate(r)
            r.candidate_net = candidate_net
        return rollouts
