# -*- coding: utf-8 -*-
"""
Asynchronous orchestration of controller (sample architecture) and evaluator (eval architecture).
"""

from __future__ import print_function
from __future__ import division

import os
import abc
import signal
import random
import string
from datetime import datetime
import multiprocessing
try:
    import queue
except Exception:
    import Queue as queue

from aw_nas import utils
from aw_nas.utils import logger as _logger
from aw_nas.base import Component
from aw_nas.trainer.base import BaseTrainer
from aw_nas.utils.exception import expect, ConfigException

__all__ = ["AsyncTrainer"]


class BaseDispatcher(Component):
    REGISTRY = "dispatcher"

    def __init__(self, schedule_cfg=None):
        super(BaseDispatcher, self).__init__(schedule_cfg)

    def start_eval_rollouts(self, rollouts):
        return [self.start_eval_rollout(r) for r in rollouts]

    @abc.abstractmethod
    def start_eval_rollout(self, rollout):
        """
        Should return immediately.
        """

    @abc.abstractmethod
    def get_finished_rollouts(self, timeout=None):
        """
        Get the rollouts whose evaluation finished.
        Should block untill any evaluation finished or timeout (if not None) is reached.
        """

    @abc.abstractmethod
    def init(self, evaluator, ckpt_dir):
        """
        Initialize the dispatcher.
        """

    @property
    def parallelism(self):
        """
        Return the current available parallelism.
        """


class MultiprocessDispatcher(BaseDispatcher):
    NAME = "multiprocess"

    def __init__(self, gpu_ids=(0,)):
        super(MultiprocessDispatcher, self).__init__()
        self.gpu_ids = gpu_ids
        self._inited = False
        self.evaluator = None
        self.req_queue = None
        self.ans_queue = None
        self.workers = []
        self.ckpt_dir = None

    @property
    def parallelism(self):
        return len(self.gpu_ids)

    def _worker(self, gpu_id):
        self.evaluator.set_device(str(gpu_id))
        while 1:
            rollout = self.req_queue.get()
            if self.ckpt_dir:
                random_salt = "".join([random.choice(string.ascii_letters + string.digits)
                                       for n in range(16)])
                ckpt_subdir = "{time}-{gpu}-{salt}".format(
                    time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), gpu=gpu_id, salt=random_salt)
                # handle output
                rollout.set_ckpt_path(os.path.join(self.ckpt_dir, ckpt_subdir))
            evaled_rollout = self.evaluator.evaluate_rollouts([rollout], is_training=True)
            self.ans_queue.put(evaled_rollout)

    def init(self, evaluator, ckpt_dir):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        print("checkpoint dir: ", self.ckpt_dir)
        self.evaluator = evaluator
        self.req_queue = multiprocessing.Queue()
        self.ans_queue = multiprocessing.Queue()
        self._register_signal_handler()
        for gpu_id in self.gpu_ids:
            worker_p = multiprocessing.Process(target=self._worker, args=(gpu_id,))
            self.workers.append(worker_p)
            worker_p.start()
        self._inited = True

    def start_eval_rollout(self, rollout):
        assert self._inited, "Dispatcher must be inited first"
        self.req_queue.put(rollout)

    def get_finished_rollouts(self, timeout=None):
        all_rollouts = []
        try:
            rollouts = self.ans_queue.get(timeout=timeout)
            all_rollouts += rollouts
        except queue.Empty:
            # timeout
            return []
        else:
            while 1:
                try:
                    rollouts = self.ans_queue.get_nowait()
                    all_rollouts += rollouts
                except queue.Empty:
                    break
            return all_rollouts

    def _register_signal_handler(self):
        ori_sigint_handler = signal.getsignal(signal.SIGINT)
        def signal_handler(sig, frame):
            print("Receive sigint, stop sub processes...")
            for worker_p in self.workers:
                if worker_p is not None:
                    print("Stoping process {}...".format(worker_p.pid))
                    worker_p.terminate()
            print("Stop sub processses finished.")
            ori_sigint_handler(sig, frame)
        signal.signal(signal.SIGINT, signal_handler)


class AsyncTrainer(BaseTrainer):
    """
    Async NAS searcher.
    """

    NAME = "async"

    SCHEDULABLE_ATTRS = []

    def __init__(self,
                 controller, evaluator, rollout_type="discrete",
                 dispatcher_type="multiprocess",
                 dispatcher_cfg=None,
                 log_timeout=600.,
                 parallelism=4,
                 steps=1000,
                 schedule_cfg=None):
        super(AsyncTrainer, self).__init__(controller, evaluator, rollout_type, schedule_cfg)

        expect(self.rollout_type == self.controller.rollout_type == \
               self.evaluator.rollout_type,
               "the rollout type of trainer/controller/evaluator must match, "
               "check the configuration. ({}/{}/{})".format(
                   self.rollout_type, self.controller.rollout_type,
                   self.evaluator.rollout_type), ConfigException)

        # configurations
        self.steps = steps
        self.parallelism = parallelism
        self.log_timeout = log_timeout

        d_cls = BaseDispatcher.get_class_(dispatcher_type)
        self.dispatcher = d_cls(**(dispatcher_cfg or {}))

        self._finished = 0
        self._remain_steps = steps
        self.save_every = None

    @property
    def _now_step(self):
        return self.steps - self._remain_steps

    # ---- APIs ----
    @classmethod
    def supported_rollout_types(cls):
        return ["mutation"]

    def train(self):
        num_nofinish = 0
        last_save_step = 0
        init_rollouts = self.controller.sample(n=self.parallelism)
        for i in range(self.parallelism):
            self.dispatcher.start_eval_rollout(init_rollouts[i])

        while not self._finished:
            finished_rollouts = self.dispatcher.get_finished_rollouts(timeout=self.log_timeout)
            if not finished_rollouts:
                num_nofinish += 1
                self.logger.info("No rollout finished in the past %d seconds",
                                 num_nofinish * self.log_timeout)
            else:
                num_nofinish = 0
                self.controller.step(finished_rollouts, None)
                for r in finished_rollouts:
                    self.logger.info("Rollout %s evaluation finished.", r)
                num_new = len(finished_rollouts)
                new_rollouts = self.controller.sample(n=num_new)
                for r in new_rollouts:
                    self.logger.info("Rollout %s put into ready queue.", r)
                    self.dispatcher.start_eval_rollout(r)
                self._remain_steps -= len(finished_rollouts)
                if self.save_every and self._now_step > (last_save_step + 1) * self.save_every:
                    self._save_all()

    def setup(self, load=None, save_every=None, train_dir=None, writer=None, load_components=None,
              interleave_report_every=None):
        # TODO: handle load components
        assert train_dir is not None, \
            ("You'd better provide a path using `--train-dir` to save all the checkpoint "
             "when using async trainer")

        self.train_dir = train_dir
        ckpt_dir = utils.makedir(os.path.join(train_dir, "checkpoints"))
        self.dispatcher.init(self.evaluator, ckpt_dir)
        current_avail_parallelism = self.dispatcher.parallelism
        self.logger.info("Arch rollout parallelism: %d; Current available dispatcher "
                         "parallelism: %d.",
                         self.parallelism, current_avail_parallelism)
        if self.parallelism > self.dispatcher.parallelism:
            self.logger.warning("Arch rollout parallelism (%d) is configuredbigger "
                                "than available dispatcher/evaluation parallelism (%d).",
                                self.parallelism, current_avail_parallelism)

        self.save_every = save_every

    def test(self):
        pass

    def derive(self, n, steps=None, out_file=None):
        pass

    def _save_path(self, name=""):
        if self.train_dir is None:
            return None
        dir_ = utils.makedir(os.path.join(self.train_dir, str(self._now_step)))
        return os.path.join(dir_, name)

    def _save_all(self):
        if self.train_dir is not None:
            self.controller.save(self._save_path("controller"))
            self.evaluator.save(self._save_path("evaluator"))
            self.logger.info("Step %3d: Save all checkpoints to directory %s",
                             self._now_step, self._save_path())

    def save(self, path):
        # No state of trainer need to be saved?
        pass

    def load(self, path):
        pass

    @classmethod
    def get_default_config_str(cls):
        all_str = super(AsyncTrainer, cls).get_default_config_str()
        # Possible dispatcher configs
        all_str += utils.component_sample_config_str("dispatcher", prefix="#   ") + "\n"
        return all_str


_LOGGER = _logger.getChild("ray_dispatcher")
try:
    from aw_nas.trainer.ray_dispatcher import RayDispatcher
except ImportError as e:
    _LOGGER.warn(
        ("Error importing module aw_nas.evaluator.ray_dispatcher: {}\n"
         "Should install ray package first.").format(e))
