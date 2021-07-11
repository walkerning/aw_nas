# -*- coding: utf-8 -*-
"""
Asynchronous orchestration of controller (sample architecture) and evaluator (eval architecture).
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import abc
import time
import signal
import random
import string
import logging
import functools
from datetime import datetime
try:
    import queue
except Exception:
    import Queue as queue

import imageio
import numpy as np
import torch
from torch import multiprocessing

from aw_nas import utils
from aw_nas.utils import logger as _logger
from aw_nas.utils import log
from aw_nas.base import Component
from aw_nas.common import BaseRollout
from aw_nas.trainer.base import BaseTrainer
from aw_nas.utils.exception import expect, ConfigException

__all__ = ["AsyncTrainer"]


class SignalObject(object):
    MAX_TERMINATE_CALLED = 3

    def __init__(self, shutdown_callback):
        self.terminate_called = 0
        self.shutdown_callback = shutdown_callback


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

    @abc.abstractmethod
    def shutdown(self):
        """
        Shutdown the dispatcher.
        """

    @abc.abstractmethod
    def stop(self):
        """
        Stop the dispatcher, try to be elegant.
        """

    @property
    def parallelism(self):
        """
        Return the current available parallelism.
        """

class MultiprocessDispatcher(BaseDispatcher):
    NAME = "multiprocess"

    STOP_WAIT_SECS = 10

    def __init__(self, gpu_ids=(0,)):
        super(MultiprocessDispatcher, self).__init__()
        self.gpu_ids = gpu_ids
        self._inited = False
        self.evaluator = None
        self.stop_event = None
        self.req_queue = None
        self.ans_queue = None
        self.workers = []
        self.ckpt_dir = None

    @property
    def parallelism(self):
        return len(self.gpu_ids)

    @staticmethod
    def _worker(evaluator, gpu_id, ckpt_dir, stop_event, req_queue, ans_queue):
        # ignore SIGINT in the worker process
        # better using stop event to gracefully shutdown the worker process
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # do not ignore the SIGTERM, as the main process would try calling proc.terminate()
        # to force shutdown the workers

        # redirect logging output to log file
        import os
        os.environ["AWNAS_LOG_LEVEL"] = "error" # set worker log level to ERROR

        worker_pid = os.getpid()
        log_file = os.path.join(ckpt_dir, "worker_pid{}_gpu{}.log".format(worker_pid, gpu_id))
        [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
        logging.basicConfig(filename=log_file, level=log.LEVEL,
                            format="worker_pid{} ".format(worker_pid) + log.LOG_FORMAT)

        # set evaluator device
        evaluator.set_device("cuda:{}".format(gpu_id))
        while not stop_event.is_set():
            rollout = req_queue.get()
            if ckpt_dir:
                random_salt = "".join([random.choice(string.ascii_letters + string.digits)
                                       for n in range(16)])
                ckpt_subdir = "{time}-{gpu}-{salt}".format(
                    time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), gpu=gpu_id, salt=random_salt)
                # handle output
                if hasattr(rollout, "set_ckpt_path"):
                    rollout.set_ckpt_path(os.path.join(ckpt_dir, ckpt_subdir))
            # FIXME: `evaluator.on_epoch_start` would not be called in worker processes,
            # if scheduling is needed in evaluator, need to pass in token through the req queue
            # upon every epoch's start
            evaled_rollout = evaluator.evaluate_rollouts([rollout], is_training=True)
            ans_queue.put(evaled_rollout)

    def init(self, evaluator, ckpt_dir):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.logger.info("checkpoint dir: %s", self.ckpt_dir)
        self.evaluator = evaluator
        self.stop_event = multiprocessing.Event()
        self.req_queue = multiprocessing.Queue()
        self.ans_queue = multiprocessing.Queue()
        self._register_signal_handler()
        backup_handlers = _logger.handlers
        _logger.handlers = [logging.NullHandler()]
        for gpu_id in self.gpu_ids:
            worker_p = multiprocessing.Process(target=self._worker, args=(
                self.evaluator, gpu_id, self.ckpt_dir,
                self.stop_event, self.req_queue, self.ans_queue))
            self.workers.append(worker_p)
            worker_p.start()
        _logger.handlers = backup_handlers
        self._inited = True

    def start_eval_rollout(self, rollout):
        if self.stop_event.is_set():
            # dispatcher has been stopped
            return
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

    def shutdown(self):
        if not self.workers:
            return

        for worker_p in self.workers:
            if worker_p is not None and worker_p.is_alive():
                self.logger.info("Stopping process {}...".format(worker_p.pid))
                worker_p.terminate()

    def stop(self):
        if self.stop_event is None or not self.workers:
            return

        # set event, which will be checked in the worker processes
        self.stop_event.set()
        wait_until = time.time() + self.STOP_WAIT_SECS

        # wait up to STOP_WAIT_SECS for the workers to stop
        self.logger.info("Wait for %d seconds for %d workers to stop.", self.STOP_WAIT_SECS, len(self.workers))
        for worker_p in self.workers:
            join_secs = max(0.0, min(wait_until - time.time(), self.STOP_WAIT_SECS))
            worker_p.join(join_secs)

        # terminate workers that do not end
        num_terminated = 0
        num_failed = 0
        while self.workers:
            worker_p = self.workers.pop()
            if worker_p.is_alive():
                worker_p.terminate()
                num_terminated += 1
            else:
                exitcode = worker_p.exitcode
                if exitcode:
                    num_failed += 1

        self.logger.info("%d processes failed (exit code != 0); call terminate on %d processes.",
                         num_failed, num_terminated)

        # clean up queues
        self._cleanup_queue(self.req_queue)
        self._cleanup_queue(self.ans_queue)

    @staticmethod
    def _cleanup_queue(queue_):
        while 1:
            try:
                queue_.get(block=False)
            except queue.Empty:
                break
        queue_.close()
        queue_.join_thread()

    def _register_signal_handler(self):
        # the signal handling of the main prorcess is registered in the trainer
        pass

    # def _register_signal_handler(self):
    #     ori_sigint_handler = signal.getsignal(signal.SIGINT)
    #     def signal_handler(sig, frame):
    #         print("Receive sigint, stop sub processes...")
    #         for worker_p in self.workers:
    #             if worker_p is not None:
    #                 print("Stoping process {}...".format(worker_p.pid))
    #                 worker_p.terminate()
    #         print("Stop sub processses finished.")
    #         ori_sigint_handler(sig, frame)
    #     signal.signal(signal.SIGINT, signal_handler)


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
                 # log_timeout=600.,
                 log_timeout=20.,
                 parallelism=4,
                 # steps=1000,
                 steps=40,
                 num_epochs=25,
                 derive_samples=10,
                 schedule_cfg=None):
        super(AsyncTrainer, self).__init__(controller, evaluator, rollout_type, schedule_cfg)

        expect(self.rollout_type == self.controller.rollout_type == \
               self.evaluator.rollout_type,
               "the rollout type of trainer/controller/evaluator must match, "
               "check the configuration. ({}/{}/{})".format(
                   self.rollout_type, self.controller.rollout_type,
                   self.evaluator.rollout_type), ConfigException)

        # must use spawn init method to enable CUDA context init in subprocesses
        torch.multiprocessing.set_start_method('spawn')

        # configurations
        self.steps = steps
        self.num_epochs = num_epochs
        self.parallelism = parallelism
        self.log_timeout = log_timeout
        self.derive_samples = derive_samples

        d_cls = BaseDispatcher.get_class_(dispatcher_type)
        self.dispatcher = d_cls(**(dispatcher_cfg or {}))

        self.epoch = 0
        self.last_epoch = 0
        self._ended = False
        self._to_issue = self._remain_steps = steps
        self.save_every = None

    @property
    def _now_step(self):
        return self.steps - self._remain_steps

    def _register_signals(self):
        signal_object = SignalObject(self.dispatcher.stop)
        self._init_signal(
            signal.SIGINT, signal_object, KeyboardInterrupt, self._shutdown_signal_handler)
        self._init_signal(
            signal.SIGTERM, signal_object, KeyboardInterrupt, self._shutdown_signal_handler)

    def _shutdown_signal_handler(
            self, signal_object, exception_class, signal_num, current_stack_frame):
        self._ended = True
        signal_object.terminate_called += 1
        print("Shutdown signal received {}/{}!".format(
            signal_object.terminate_called, signal_object.MAX_TERMINATE_CALLED))
        if signal_object.terminate_called == signal_object.MAX_TERMINATE_CALLED:
            # if termination is called for enough times, raise exception and stop the program
            raise exception_class()
        signal_object.shutdown_callback()
        sys.exit(0)

    @staticmethod
    def _init_signal(signal_num, signal_object, exception_class, handler):
        handler = functools.partial(handler, signal_object, exception_class)
        signal.signal(signal_num, handler)
        signal.siginterrupt(signal_num, False)

    # ---- APIs ----
    @classmethod
    def supported_rollout_types(cls):
        return list(BaseRollout.all_classes_().keys())

    def train(self):
        num_nofinish = 0
        self.controller.set_mode("train")
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            if self._ended:
                break
            self.on_epoch_start(epoch)
            self._to_issue = self._remain_steps = self.steps

            init_rollouts = self.controller.sample(n=min(self.parallelism, self._to_issue))
            for rollout in init_rollouts:
                self.logger.info("Rollout %s put into ready queue.", rollout)
                self.dispatcher.start_eval_rollout(rollout)
            self._to_issue -= len(init_rollouts)

            while not self._ended:
                finished_rollouts = self.dispatcher.get_finished_rollouts(timeout=self.log_timeout)
                if not finished_rollouts:
                    num_nofinish += 1
                    self.logger.info("No rollout finished in the past %d s."
                            " %d rollouts(s) to be sampled.",
                                     num_nofinish * self.log_timeout,
                                     self._to_issue)
                else:
                    num_nofinish = 0
                    self.controller.step(finished_rollouts, None, perf_name="reward")
                    for r in finished_rollouts:
                        self.logger.info("Rollout %s evaluation finished.", r)
                    num_new = len(finished_rollouts)
                    # get evaluation results of num_new rollouts
                    self._remain_steps -= num_new
                    # print("epoch: {}; to issue: {}; to get: {}".format(
                    # epoch, self._to_issue, self._remain_steps))
                    if self._remain_steps <= 0:
                        break
                    new_rollouts = self.controller.sample(n=min(num_new, self._to_issue))
                    for r in new_rollouts:
                        self.logger.info("Rollout %s put into ready queue.", r)
                        self.dispatcher.start_eval_rollout(r)
                    # issue len(new_rollouts) new rollouts for evaluation
                    self._to_issue -= len(new_rollouts)
            if self.save_every and self.epoch % self.save_every == 0:
                self._save_all()

        # stop the dispatcher
        self.dispatcher.stop()

    def setup(self, load=None, save_every=None, save_controller_every=None, train_dir=None, writer=None, load_components=None,
              interleave_report_every=None):
        # TODO: handle load components
        assert train_dir is not None, \
            ("You'd better provide a path using `--train-dir` to save all the checkpoint "
             "when using async trainer")

        super(AsyncTrainer, self).setup(load, save_every, save_controller_every, train_dir, writer, load_components,
                interleave_report_every)
        self.train_dir = train_dir
        ckpt_dir = utils.makedir(os.path.join(train_dir, "checkpoints"))
        self.dispatcher.init(self.evaluator, ckpt_dir)
        self._register_signals() # register signal handlers for clean up
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
        """
        Derive and test, plot the best arch among the arch samples.

        Copied from SimpleTrainer.
        Currently, the main process is used for deriving.
        """
        rollouts = self.derive(n=self.derive_samples)

        rewards = [r.get_perf("reward") or 0. for r in rollouts]
        mean_rew = np.mean(rewards)
        idx = np.argmax(rewards)
        other_perfs = {n: [r.perf.get(n, 0.) for r in rollouts] for n in rollouts[0].perf}

        save_path = self._save_path("rollout/cell")
        if save_path is not None:
            # NOTE: If `train_dir` is None, the image will not be saved to tensorboard too
            fnames = rollouts[idx].plot_arch(save_path, label="epoch {}".format(self.epoch))
            if not self.writer.is_none() and fnames is not None:
                for cg_n, fname in fnames:
                    image = imageio.imread(fname)
                    self.writer.add_image("genotypes/{}".format(cg_n), image, self.epoch,
                                          dataformats="HWC")

        self.logger.info("TEST Epoch %3d: Among %d sampled archs: "
                         "BEST (in reward): %.5f (mean: %.5f); Performance: %s",
                         self.epoch, self.derive_samples, rewards[idx], mean_rew,
                         "; ".join(["{}: {} (mean {:.5f})".format(
                             n, "{:.5f}".format(
                                 other_perfs[n][idx]) if other_perfs[n][idx] is not None else None,
                             np.mean([0])) for n in rollouts[0].perf]))
        self.logger.info("Saved this arch to %s.\nGenotype: %s",
                         save_path, rollouts[idx].genotype)
        self.controller.summary(rollouts, log=True, log_prefix="Rollouts Info: ", step=self.epoch)
        return rollouts

    def derive(self, n, steps=None):
        with self.controller.begin_mode("eval"):
            rollouts = self.controller.sample(n)
            for i_sample in range(n):
                rollouts[i_sample] = self.evaluator.evaluate_rollouts([rollouts[i_sample]],
                                                                      is_training=False,
                                                                      eval_batches=steps)[0]
                print("Finish test {}/{}\r".format(i_sample+1, n), end="")
        return rollouts

    def _save_path(self, name=""):
        if self.train_dir is None:
            return None
        dir_ = utils.makedir(os.path.join(self.train_dir, str(self.epoch)))
        return os.path.join(dir_, name)

    def _save_all(self):
        if self.train_dir is not None:
            self.controller.save(self._save_path("controller"))
            # FIXME: do evaluator really need to be saved, since evaluator is not updated
            self.evaluator.save(self._save_path("evaluator"))
            self.logger.info("Step %3d: Save all checkpoints to directory %s",
                             self.epoch, self._save_path())

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
        ("Cannot import module aw_nas.evaluator.ray_dispatcher: {}\n"
         "Should install ray package first.").format(e))
