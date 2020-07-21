# -*- coding: utf-8 -*-
import os
import time
import random
import string
import signal
import multiprocessing
from datetime import datetime

import psutil
import ray

from aw_nas.utils.exception import expect, ConfigException
from aw_nas.trainer.async_trainer import BaseDispatcher


class KillSignal(ray.experimental.signal.Signal):
    pass

@ray.remote
class Killer(object):
    def send_kill(self):
        ray.experimental.signal.send(KillSignal())
        print("finished sending kill signals, "
              "please wait for some seconds for all these tasks to exit")

class RayDispatcher(BaseDispatcher):
    NAME = "ray"

    def __init__(self, redis_addr=None):
        super(RayDispatcher, self).__init__()

        expect(redis_addr is not None, "Redis address must be specified", ConfigException)
        self.redis_addr = redis_addr
        ray.init(redis_address=redis_addr)
        self.killer = Killer.remote() # create the killer actor
        self.evaluator = None
        self.evaluate_func = None
        self.ckpt_dir = None
        self.executing_ids = set()

    def get_evaluate_func(self):
        @ray.remote(num_gpus=1)
        def evaluate_func(rollout, killer):
            # TODO: use subprocess to run?
            gpus = ray.get_gpu_ids()
            gpu_str = ",".join(map(str, gpus))
            # unset cuda visible devices, do not use this env, use ordinal consistently
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            self.evaluator.set_device(gpu_str)
            if self.ckpt_dir:
                random_salt = "".join([random.choice(string.ascii_letters + string.digits)
                                       for n in range(16)])
                ckpt_subdir = "{time}-{gpu}-{salt}".format(
                    time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    gpu=gpu_str, salt=random_salt)
                # handle output
                rollout.set_ckpt_path(os.path.join(self.ckpt_dir, ckpt_subdir))
            return_dct = multiprocessing.Manager().dict()
            def _evaluate_wrapper(rollout, return_dct):
                rollout = self.evaluator.evaluate_rollouts([rollout], is_training=True)[0]
                return_dct["_"] = rollout
            proc = multiprocessing.Process(target=_evaluate_wrapper, args=(rollout, return_dct))
            proc.start()
            # wait for proc finish or killed
            while 1:
                time.sleep(10)
                if proc.is_alive():
                    sigs = ray.experimental.signal.receive([killer], timeout=1)
                    if sigs:
                        print("ray task: receive kill signal from killer, "
                              "kill the working processes")
                        process = psutil.Process(proc.pid)
                        for c_proc in process.children(recursive=True):
                            c_proc.kill()
                        process.kill()
                        exit_status = 1
                        break
                else:
                    exit_status = proc.exitcode
                    break
            if exit_status != 0:
                return None
            return return_dct["_"]
        return evaluate_func

    def init(self, evaluator, ckpt_dir):
        # self.evaluate_func = ray.remote(evaluator.evalute, num_gpus=1)
        self.evaluator = evaluator
        self.ckpt_dir = ckpt_dir
        self.evaluate_func = self.get_evaluate_func()
        self._register_signal_handler()

    def stop(self):
        print("Stop ray dispatcher...")
        self.killer.send_kill.remote()

    def shutdown(self):
        print("Shutdown ray dispatcher...")
        self.killer.send_kill.remote()

    def start_eval_rollout(self, rollout):
        res_id = self.evaluate_func.remote(rollout, self.killer)
        self.executing_ids.add(res_id)

    def get_finished_rollouts(self, timeout=None):
        ready_ids, _ = ray.wait(list(self.executing_ids), timeout=timeout)
        f_rollouts = ray.get(ready_ids)
        self.executing_ids -= set(ready_ids)
        if None in f_rollouts:
            # sigint
            print("Found None in the finished rollout! Interrupted!")
            raise KeyboardInterrupt()
        return f_rollouts

    @property
    def parallelism(self):
        # FIXME: temp. This returns the #available gpus on the current node, not correct
        return ray.worker._global_node.get_resource_spec().num_gpus

    def _register_signal_handler(self):
        pass
        # ori_sigint_handler = signal.getsignal(signal.SIGINT)
        # def signal_handler(sig, frame):
        #     print("Receive sigint, sending kill signal...")
        #     self.killer.send_kill.remote()
        # signal.signal(signal.SIGINT, signal_handler)
