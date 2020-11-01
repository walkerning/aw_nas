# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
MultiShotEvaluator: Curve fitting to estimate reward at targeted FLOPs
Copyright (c) 2019 Xuefei Ning, Junbo Zhao
"""

import abc
import copy
import collections
import math

import numpy as np

from scipy.optimize import curve_fit
from aw_nas.common import BaseRollout
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.weights_manager.base import BaseWeightsManager


class BaseMultiShotEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset,
        weights_manager,
        objective,
        rollout_type,
        sub_evaluators=[],
        schedule_cfg=None,
    ):
        super(BaseMultiShotEvaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type, schedule_cfg=schedule_cfg
        )
        self.sub_evaluator_cfgs = sub_evaluators
        self.num_sub_evaluators = len(self.sub_evaluator_cfgs)
        self.sub_evaluators = []
        for sub_eva_cfg in self.sub_evaluator_cfgs:
            wm_type = sub_eva_cfg["sub_weights_manager_type"]
            wm_cfg = sub_eva_cfg["sub_weights_manager_cfg"]
            wm_device = sub_eva_cfg.get("device", weights_manager.device)
            sub_wm = BaseWeightsManager.get_class_(wm_type)(
                weights_manager.search_space, device=wm_device, **wm_cfg
            )
            eva_type = sub_eva_cfg["sub_evaluator_type"]
            eva_cfg = sub_eva_cfg["sub_evaluator_cfg"]
            eva_ckpt_path = sub_eva_cfg.get("ckpt_path", None)
            sub_eva = BaseEvaluator.get_class_(eva_type)(
                dataset, sub_wm, objective, **eva_cfg
            )
            if eva_ckpt_path is not None:
                sub_eva.load(eva_ckpt_path)
            self.sub_evaluators.append(sub_eva)

    @classmethod
    def supported_data_types(cls):
        # just return all possible data types
        # since this is a hyper-evaluator
        return ["image", "sequence"]

    @classmethod
    def supported_rollout_types(cls):
        # just return all possible rollout types
        # since this is a hyper-evaluator
        return list(BaseRollout.all_classes_().keys())

    def evaluate_rollouts(
        self,
        rollouts,
        is_training,
        portion=None,
        eval_batches=None,
        return_candidate_net=False,
        callback=None,
    ):
        sub_eva_perfs = []
        for sub_eva in self.sub_evaluators:
            rollouts = sub_eva.evaluate_rollouts(
                rollouts,
                is_training,
                portion=portion,
                eval_batches=eval_batches,
                return_candidate_net=False,
                callback=callback,
            )
            r_perfs = [copy.deepcopy(r.perf) for r in rollouts]
            sub_eva_perfs.append(r_perfs)
        r_comb_perfs = [
            self._combine_multi_perfs(r_perfs) for r_perfs in zip(*sub_eva_perfs)
        ]
        for i, rollout in enumerate(rollouts):
            rollout.multi_perf = [copy.deepcopy(sub_eva_perfs[j][i]) for j in range(self.num_sub_evaluators)]
        for rollout, r_comb_perf in zip(rollouts, r_comb_perfs):
            # use update instead of assignment, to keep track of the `predicted_score` field
            # that is set by predictor-based controller
            rollout.perf.update(r_comb_perf)
        return rollouts

    def update_rollouts(self, rollouts):
        pass

    def update_evaluator(self, controller):
        pass

    def save(self, path):
        for i_eva, sub_eva in enumerate(self.sub_evaluators):
            sub_eva.save("{}_sub{}".format(path, i_eva))

    def load(self, path):
        for i_eva, sub_eva in enumerate(self.sub_evaluators):
            sub_eva.load("{}_sub{}".format(path, i_eva))

    # ---- helper methods ----
    @abc.abstractmethod
    def _combine_multi_perfs(self, perfs):
        res = collections.OrderedDict()
        # An unmeaningful example
        res["reward"], res["acc_clean"], res["acc_adv"], res["flops"] = (
            0.5 * (perfs[0]["acc_clean"] + perfs[0]["acc_adv"]),
            perfs[0]["acc_clean"],
            perfs[0]["acc_adv"],
            perfs[0]["flops"],
        )
        return res


class MultiEvaluator(BaseMultiShotEvaluator):
    NAME = "multi_shot_evaluator"

    def __init__(
        self,
        dataset,
        weights_manager,
        objective,
        rollout_type,
        sub_evaluators=[],
        target_flops=1500.0e6,
        fit_function_type="log_power",
        schedule_cfg=None,
    ):
        super(MultiEvaluator, self).__init__(
            dataset,
            weights_manager,
            objective,
            rollout_type,
            sub_evaluators,
            schedule_cfg,
        )
        self.target_flops = float(target_flops)
        self.fit_function_type = fit_function_type
        self.fit_function = getattr(self, fit_function_type)
        assert callable(self.fit_function)

    # ---- candidate functions ----
    @staticmethod
    def pow2(x, c, a):
        return c - a * x ** (-a)

    @staticmethod
    def pow3(x, c, a, alpha):
        return c - a * x ** (-alpha)

    @staticmethod
    def vapor_pressure(x, a, b, c):
        return a + b / x + c * np.log(x)

    @staticmethod
    def log_log_linear(x, a, b):
        return np.log(a * np.log(x) + b)

    @staticmethod
    def ilog2(x, c, a):
        return c - a / np.log(x)

    @staticmethod
    def log_power(x, a, b, c):
        return a / (1 + pow(x, b) * math.exp(c))

    # ---- helper methods ----
    def _combine_multi_perfs(self, perfs):
        acc_cleans, acc_advs, flops = zip(
            *[(perf["acc_clean"], perf["acc_adv"], perf["flops"]) for perf in perfs]
        )

        reward_ = [
            0.5 * (acc_clean + acc_adv)
            for acc_clean, acc_adv in zip(acc_cleans, acc_advs)
        ]
        acc_cleans, acc_advs, flops, reward_ = (
            np.array(acc_cleans),
            np.array(acc_advs),
            np.array(flops),
            np.array(reward_),
        )

        target_acc_clean = self.fit(
            self.fit_function, flops, acc_cleans, self.target_flops
        )
        target_acc_adv = self.fit(self.fit_function, flops, acc_advs, self.target_flops)

        target_reward = self.fit(self.fit_function, flops, reward_, self.target_flops)
        res = collections.OrderedDict()
        res["reward"], res["acc_clean"], res["acc_adv"], res["flops"] = (
            target_reward,
            target_acc_clean,
            target_acc_adv,
            self.target_flops,
        )
        return res

    @staticmethod
    def fit(func, flops, rewards, target_flops):
        popt, _ = curve_fit(
            func,
            flops,
            rewards,
            maxfev=500000000,
            bounds=([0.0, -np.inf, -np.inf], [np.inf, 0, np.inf]),
        )
        target_reward = func(target_flops, *popt)
        return target_reward
