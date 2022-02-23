# -*- coding: utf-8 -*-

import os
import copy
import pickle
import warnings

import numpy as np
from scipy.stats import stats
from torch.utils.data import Dataset, DataLoader

from aw_nas import utils
from aw_nas.common import BaseRollout
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.controller.base import BaseController
from aw_nas.evaluator.arch_network import ArchNetwork

class ArchDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self._len = len(self.data)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

def train_predictor(logger, train_loader, val_loader, model, epochs, cfg):
    for i_epoch in range(1, epochs + 1):
        avg_loss = train_epoch(logger, train_loader, model, i_epoch, cfg)
        logger.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        corr, _ = valid_epoch(logger, train_loader, model, cfg)
        logger.info("Train: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, corr))
        if val_loader is not None:
            val_corr, _ = valid_epoch(logger, val_loader, model, cfg)
            logger.info("Valid: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, val_corr))
    return avg_loss, corr, val_corr if val_loader is not None else None

def valid_epoch(logger, val_loader, model, cfg, funcs=[]):
    model.eval()
    all_scores = []
    true_accs = []
    for _, (archs, accs) in enumerate(val_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    funcs_res = [func(true_accs, all_scores) for func in funcs]
    return corr, funcs_res

def train_epoch(logger, train_loader, model, epoch, cfg):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, accs) in enumerate(train_loader):
        archs = np.array(archs)
        accs = np.array(accs)
        n = len(archs)
        if cfg["compare"]:
            n_max_pairs = int(cfg["max_compare_ratio"] * n)
            acc_diff = np.array(accs)[:, None] - np.array(accs)
            acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
            ex_thresh_inds = np.where(acc_abs_diff_matrix > cfg["compare_threshold"])
            ex_thresh_num = len(ex_thresh_inds[0])
            if ex_thresh_num > n_max_pairs:
                keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
                ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
            archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], \
                                           (acc_diff > 0)[ex_thresh_inds]
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)
        if step % cfg["report_freq"] == 0:
            n_pair_per_batch = (cfg["batch_size"] * (cfg["batch_size"] - 1)) // 2
            logger.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if cfg["compare"] else ""))
    return objs.avg


class PredictorBasedController(BaseController):
    NAME = "predictor-based"

    def __init__(self, search_space, device, rollout_type, mode="eval",
                 inner_controller_type=None,
                 inner_controller_cfg=None,
                 arch_network_type="pointwise_comparator",
                 arch_network_cfg=None,

                 # how to use the inner controller and arch network to sample new archs
                 inner_sample_n=1,
                 inner_samples=1,
                 inner_steps=200,
                 inner_report_freq=50,
                 predict_batch_size=512,
                 inner_random_init=True,
                 inner_iter_random_init=True,
                 inner_enumerate_search_space=False, # DEPRECATED
                 inner_enumerate_sample_ratio=None, # DEPRECATED
                 min_inner_sample_ratio=10,

                 # how to train the arch network
                 begin_train_num=0,
                 predictor_train_cfg={
                     "epochs": 200,
                     "num_workers": 2,
                     "batch_size": 128,
                     "compare": True,
                     "max_compare_ratio": 4,
                     "compare_threshold": 0.,
                     "report_freq": 50,
                     "train_valid_split": None,
                     "n_cross_valid": None,
                 },
                 training_on_load=False, # force retraining on load
                 pretrained_predictor_path: str = None, # load pretrained predictor
                 schedule_cfg=None):
        super(PredictorBasedController, self).__init__(
            search_space, rollout_type, mode, schedule_cfg)

        expect(inner_controller_type is not None, "Must specificy inner controller type",
               ConfigException)

        self.device = device
        self.predictor_train_cfg = predictor_train_cfg
        self.inner_controller_reinit = True
        self.inner_sample_n = inner_sample_n
        self.inner_samples = inner_samples
        self.inner_steps = inner_steps
        self.inner_report_freq = inner_report_freq
        self.inner_random_init = inner_random_init
        self.inner_iter_random_init = inner_iter_random_init
        self.inner_enumerate_search_space = inner_enumerate_search_space
        if inner_enumerate_search_space:
            warnings.warn("The `inner_enumerate_search_space` option is DEPRECATED. "
                          "Use inner_controller, and set `inner_samples`, `inner_steps` "
                          "accordingly", warnings.DeprecationWarning)
        self.inner_enumerate_sample_ratio = inner_enumerate_sample_ratio
        self.min_inner_sample_ratio = min_inner_sample_ratio
        self.predict_batch_size = predict_batch_size
        self.begin_train_num = begin_train_num
        self.training_on_load = training_on_load

        # initialize the inner controller
        inner_controller_cfg = inner_controller_cfg or {}
        tmp_r_type = inner_controller_cfg.get("rollout_type", None)
        if tmp_r_type is not None:
            expect(tmp_r_type == rollout_type,
                   "If specified, inner_controller's `rollout_type` must match "
                   "the outer `rollout_type`",
                   ConfigException)
            inner_controller_cfg.pop("rollout_type", None)
            inner_controller_cfg.pop("mode", None)

        self.inner_controller_type = inner_controller_type
        self.inner_controller_cfg = inner_controller_cfg
        # if not self.inner_controller_reinit:
        self.inner_controller = BaseController.get_class_(self.inner_controller_type)(
            self.search_space, self.device,
            rollout_type=self.rollout_type, **self.inner_controller_cfg)
        # else:
        #     self.inner_controller = None
        # Currently, we do not use controller with parameters to be optimized (e.g. RL-learned RNN)
        self.inner_cont_optimizer = None

        # initialize the predictor
        arch_network_cfg = arch_network_cfg or {}
        expect(arch_network_type in ["pointwise_comparator", "dynamic_ensemble_pointwise_comparator"],
               "only support pointwise_comparator arch network for now", ConfigException)
        model_cls = ArchNetwork.get_class_(arch_network_type)
        self.model = model_cls(self.search_space, **arch_network_cfg)
        self.model.to(self.device)

        self.gt_rollouts = []
        self.gt_arch_scores = []
        self.num_gt_rollouts = 0
        # self.train_loader = None
        # self.val_loader = None
        self.is_predictor_trained = False

        if pretrained_predictor_path:
            self.init_load_predictor(pretrained_predictor_path)

    def _predict_rollouts(self, rollouts):
        num_r = len(rollouts)
        cur_ind = 0
        while cur_ind < num_r:
            end_ind = min(num_r, cur_ind + self.predict_batch_size)
            padded_archs = self._pad_archs([r.arch for r in rollouts[cur_ind:end_ind]])
            scores = self.model.predict(padded_archs).cpu().data.numpy()
            for r, score in zip(rollouts[cur_ind:end_ind], scores):
                r.set_perf(score, name="predicted_score")
            cur_ind = end_ind
        return rollouts

    def _pad_archs(self, archs):
        if hasattr(self.search_space, "pad_archs"):
            return self.search_space.pad_archs(archs)
        return archs

    # ---- APIs ----
    def set_mode(self, mode):
        if self.inner_controller is not None:
            self.inner_controller.set_mode(mode)
        self.mode = mode

    def set_device(self, device):
        if self.inner_controller is not None:
            self.inner_controller.set_device(device)
        self.device = device

    def sample(self, n=1, batch_size=1):
        """Sample architectures based on the current predictor"""

        if self.mode == "eval":
            # return the best n rollouts that are evaluted by ground-truth evaluator
            self.logger.info("Return the best {} rollouts in the population".format(n))
            all_gt_arch_scores= sum(self.gt_arch_scores, [])
            all_rollouts = sum(self.gt_rollouts, [])
            best_inds = np.argpartition([item[1] for item in all_gt_arch_scores], -n)[-n:]
            # all_rollouts, all_scores = zip(
            #     *[(r, r.get_perf("reward")) for rs in self.gt_rollouts for r in rs])
            # best_inds = np.argpartition(all_scores, -n)[-n:]
            return [all_rollouts[ind] for ind in best_inds]

        if not self.is_predictor_trained:
            # if predictor is not trained, random sample from search space
            return [self.search_space.random_sample() for _ in range(n)]

        if n % self.inner_sample_n != 0:
            self.logger.warn("sample number %d cannot be divided by inner_sample_n %d",
                             n, self.inner_sample_n)

        # the arch rollouts that have already evaled, avoid sampling them
        already_evaled_r_set = sum(self.gt_rollouts, [])
        # nb101, nb201 420k, 15k, small. forward 1~2min max
        if self.inner_enumerate_search_space:
            if self.inner_enumerate_sample_ratio is not None:
                assert n % self.inner_sample_n == 0

            max_num = None if self.inner_enumerate_sample_ratio is None \
                      else n * self.inner_enumerate_sample_ratio
            iter_ = self.search_space.batch_rollouts(batch_size=self.predict_batch_size,
                                                     shuffle=True,
                                                     max_num=max_num)
            scores = []
            all_rollouts = []
            num_ignore = 0
            for rollouts in iter_:
                # remove the rollouts that is already evaled
                ori_len_ = len(rollouts)
                rollouts = [rollout for rollout in rollouts if rollout not in already_evaled_r_set]
                num_ignore += ori_len_ - len(rollouts)
                all_rollouts = all_rollouts + self._predict_rollouts(rollouts)
                scores = scores + [i.perf["predicted_score"] for i in rollouts]

            if self.inner_sample_n is not None:
                num_iters = n // self.inner_sample_n
                rs_per_s = len(scores) // num_iters
                scores = np.array(scores)[:rs_per_s * num_iters]
                inds = np.argpartition(scores.reshape([num_iters, rs_per_s]),
                                       -self.inner_sample_n, axis=1)[:, -self.inner_sample_n:]
                # inds: (num_iters, self.inner_sample_n)
                best_inds = (inds + rs_per_s * np.arange(num_iters)[:, None]).reshape(-1)
                self.logger.info("Random sample %d archs (max num %d), ignore %d already evaled archs, "
                                 "and choose %d archs per %d archs with highest predict scores",
                                 len(scores), max_num, num_ignore, self.inner_sample_n, rs_per_s)
            else:
                # finally: ranking, and get the first n archs. train_cellss_pkl.py `sample` function
                best_inds = np.argpartition(scores, -n)[-n:]
                self.logger.info("Random sample %d archs (max num %d), ignore %d already evaled archs, "
                                 "and choose %d archs with highest predict scores",
                                 len(scores), max_num, num_ignore, n)
            return [all_rollouts[i] for i in best_inds]

        # if self.inner_controller_reinit:
        self.inner_controller = BaseController.get_class_(self.inner_controller_type)(
            self.search_space, self.device, mode=self.mode,
            rollout_type=self.rollout_type, **self.inner_controller_cfg)
        if hasattr(self.inner_controller, "set_init_population"):
            self.logger.info("re-evaluating %d rollouts using the current predictor",
                             self.num_gt_rollouts)
            # set the init population of the inner controller
            # re-evaluate rollouts using the current predictor
            for rollouts in self.gt_rollouts:
                rollouts = self._predict_rollouts(rollouts)

            if not self.inner_random_init:
                self.inner_controller.set_init_population(
                    sum(self.gt_rollouts, []), perf_name="predicted_score")

        # inner_sample_n: how many archs to sample every iter
        num_iter = (n + self.inner_sample_n - 1) // self.inner_sample_n
        sampled_rollouts = []
        sampled_scores = []
        # the number, mean and max predicted scores of current sampled archs
        cur_sampled_mean_max = (0, 0, 0)
        i_iter = 1
        while i_iter <= num_iter:
            # random init
            if self.inner_iter_random_init \
               and hasattr(self.inner_controller, "reinit"):
                if i_iter > 1:
                    # might use gt rollouts as the init population if `inner_random_init=true`
                    # so, do not call reinit when i_iter == 1
                    if (not isinstance(self.inner_iter_random_init, int)) or \
                       self.inner_iter_random_init == 1 or \
                       i_iter % self.inner_iter_random_init == 1:
                        # if `inner_iter_random_init` is a integer
                        # only reinit every `inner_iter_random_init` iterations.
                        # `inner_iter_random_init==True` is the same as `inner_iter_random_init==1`,
                        # and means that every iter (besides iter 1) would call `reinit`
                        self.inner_controller.reinit()


            new_per_step_meter = utils.AverageMeter()

            # a list with length self.inner_sample_n
            best_rollouts = []
            best_scores = []
            num_to_sample = min(n - (i_iter - 1) * self.inner_sample_n, self.inner_sample_n)
            iter_r_set = []
            iter_s_set = []
            sampled_r_set = sampled_rollouts
            for i_inner in range(1, self.inner_steps + 1):
                # self.inner_controller.on_epoch_begin(i_inner)
                # while 1:
                #     rollouts = self.inner_controller.sample(self.inner_samples)
                #     # remove the duplicate rollouts
                #     # *fixme* FIXME: local minimum problem exists!
                #     # random sample is one way, or do not use the best as the init?
                #     # Add a test to test the whole dataset...
                #     # grond-truth evaled, decided rollouts
                #     # rollouts = [r for r in rollouts
                #     #             if r not in already_evaled_r_set \
                #     #             and r not in sampled_r_set]
                #     # and r not in iter_r_set

                #     if not rollouts:
                #         print("all conflict, resample")
                #         continue
                #     else:
                #         # print("sampled {}".format(i_inner))
                #         break
                rollouts = self.inner_controller.sample(self.inner_samples)
                rollouts = self._predict_rollouts(rollouts)
                self.inner_controller.step(rollouts, self.inner_cont_optimizer,
                                           perf_name="predicted_score")

                # keep the `num_to_sample` archs with highest scores
                step_scores = [r.get_perf(name="predicted_score") for r in rollouts]
                new_rollouts = [r for r in rollouts
                                if r not in already_evaled_r_set \
                                and r not in sampled_r_set
                                and r not in iter_r_set]
                new_step_scores = [r.get_perf(name="predicted_score") for r in new_rollouts]
                new_per_step_meter.update(len(new_rollouts))
                best_rollouts += new_rollouts
                best_scores += new_step_scores
                iter_r_set += rollouts
                iter_s_set += step_scores

                if len(best_scores) > num_to_sample:
                    keep_inds = np.argpartition(best_scores, -num_to_sample)[-num_to_sample:]
                    best_rollouts = [best_rollouts[ind] for ind in keep_inds]
                    best_scores = [best_scores[ind] for ind in keep_inds]
                if i_inner % self.inner_report_freq == 0:
                    self.logger.info(
                        ("Iter %d (to sample %d) (already sampled %d mean %.5f, best %.5f); "
                         "Step %d: sample %d step mean %.5f best %.5f: {} "
                         "(iter mean %.5f, best %.5f). AVG new/step: %.3f").format(
                             ", ".join(["{:.5f}".format(s) for s in best_scores])),
                        i_iter, num_to_sample,
                        cur_sampled_mean_max[0], cur_sampled_mean_max[1], cur_sampled_mean_max[2],
                        i_inner, len(rollouts), np.mean(step_scores), np.max(step_scores),
                        np.mean(iter_s_set), np.max(iter_s_set), new_per_step_meter.avg)
            if new_per_step_meter.sum < num_to_sample * self.min_inner_sample_ratio:
                # rerun this iter, also reinit!
                self.logger.info("Cannot find %d (num_to_sample x min_inner_sample_ratio)"
                                 " (%d x %d) new rollouts in one run of the inner controller"
                                 "Re-init the controller and re-run this iteration.",
                                 num_to_sample * self.min_inner_sample_ratio,
                                 num_to_sample, self.min_inner_sample_ratio)
                continue

            i_iter += 1
            assert len(best_scores) == num_to_sample
            sampled_rollouts += best_rollouts
            sampled_scores += best_scores
            cur_sampled_mean_max = (
                len(sampled_scores), np.mean(sampled_scores), np.max(sampled_scores))

        return sampled_rollouts

    def step(self, rollouts, optimizer, perf_name):
        """Train the predictor, using the ground-truth evaluations"""
        self.gt_rollouts.append(rollouts)
        if perf_name != "reward":
            # set an attribute to each rollout
            [setattr(r, "gt_perf_name", perf_name) for r in rollouts]

        new_archs, new_perfs = zip(*[(r.arch, r.get_perf(perf_name)) for r in rollouts])
        self.gt_arch_scores.append(list(zip(self._pad_archs(new_archs), new_perfs)))

        self.num_gt_rollouts += len(rollouts)
        if self.num_gt_rollouts < self.begin_train_num:
            self.logger.info("num ground-truth rollouts (%d) smaller than %d",
                             self.num_gt_rollouts, self.begin_train_num)
            return 0

        return self.prepare_data_and_train_predictor()

    def prepare_data_and_train_predictor(self):
        # *TODO*: different ways of utilizing multi-stage data
        # weight? finetune with smaller lr? multi-stage sampling?

        # TODO: maybe use some for validation, and *early stop using validation set*
        # TODO: cross valid and use an ensemble
        # val_loader = DataLoader(
        #     valid_data, batch_size=self.predictor_train_cfg["batch_size"],
        #     shuffle=True, pin_memory=True, num_workers=self.predictor_train_cfg["num_workers"],
        #     collate_fn=lambda items: list(zip(*items)))
        gt_arch_scores = copy.deepcopy(self.gt_arch_scores)
        tv_split = self.predictor_train_cfg.get("train_valid_split", None)

        if tv_split is not None and tv_split < 1:
            val_arch_scores = []
            train_arch_scores = []
            for arch_scores in gt_arch_scores:
                num_stage_arch = len(arch_scores)
                num_stage_train_arch = int(tv_split * num_stage_arch)
                val_arch_scores.append(arch_scores[num_stage_train_arch:])
                train_arch_scores.append(arch_scores[:num_stage_train_arch])
            all_val_data = sum(val_arch_scores, []) # *TODO*: other methods to use multi-stage data
            val_loader = DataLoader(
                ArchDataset(all_val_data), batch_size=self.predictor_train_cfg["batch_size"],
                shuffle=True, pin_memory=True, num_workers=self.predictor_train_cfg["num_workers"],
                collate_fn=lambda items: list(zip(*items)))
        else:
            val_loader = None
            train_arch_scores = gt_arch_scores

        # construct the train loader
        all_train_data = sum(train_arch_scores, []) # *TODO*: other methods to use multi-stage data
        self.logger.info("Number of data: train {} val {}".format(
            len(all_train_data), len(all_val_data) if val_loader is not None else 0))
        train_loader = DataLoader(
            ArchDataset(all_train_data), batch_size=self.predictor_train_cfg["batch_size"],
            shuffle=True, pin_memory=True, num_workers=self.predictor_train_cfg["num_workers"],
            collate_fn=lambda items: list(zip(*items)))

        loss, corr, val_corr = train_predictor(
            self.logger, train_loader, val_loader, self.model,
            self.predictor_train_cfg["epochs"], self.predictor_train_cfg)

        self.is_predictor_trained = True
        return loss

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        # TODO: report the validation information of the trained predictor
        pass

    def save(self, path):
        # save the evaled rollouts, predictor, controller
        with open("{}_rollouts.pkl".format(path), "wb") as f:
            pickle.dump(self.gt_rollouts, f)
        if self.inner_controller is not None:
            self.inner_controller.save("{}_controller".format(path))
        if self.is_predictor_trained:
            # only save when the predictor is trained
            self.model.save("{}_predictor".format(path))

    def init_load_predictor(self, path):
        assert os.path.exists(path)
        self.model.load(path)
        self.logger.info("Initial load predictor from {}.".format(os.path.abspath(path)))

    def load(self, path):
        # load the evaled rollouts, predictor, controller
        rollout_path = "{}_rollouts.pkl".format(path)
        if os.path.exists(rollout_path):
            with open(rollout_path, "rb") as f:
                self.gt_rollouts = pickle.load(f)
            for rollouts in self.gt_rollouts:
                # save the perf_name instead of the gt_arch_scores
                archs, perfs = zip(*[(r.arch, r.get_perf(getattr(r, "gt_perf_name", "reward")))
                                     for r in rollouts])
                self.gt_arch_scores.append(list(zip(self._pad_archs(archs), perfs)))
            self.num_gt_rollouts = sum([len(rollouts) for rollouts in self.gt_rollouts])

        inner_controller_path = "{}_controller".format(path)
        if os.path.exists(inner_controller_path):
            self.inner_controller.load(inner_controller_path)

        predictor_path = "{}_predictor".format(path)
        if os.path.exists(predictor_path):
            self.model.load(predictor_path)
            self.is_predictor_trained = True
        if self.training_on_load:
            self.logger.info(("`training_on_load` set, re-training a predictor."
                              " Current number of gt rollouts: %d. `begin_train_num`: %d"),
                             self.num_gt_rollouts, self.begin_train_num)
            self.prepare_data_and_train_predictor()
            self.is_predictor_trained = True

    @classmethod
    def supported_rollout_types(cls):
        return list(BaseRollout.all_classes_().keys())
