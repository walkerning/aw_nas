# -*- coding: utf-8 -*-
# pylint: disable-all

from io import StringIO
import os
import sys
import copy
import shutil
import logging
import argparse
import random
import pickle

import yaml
import setproctitle
from scipy.stats import stats

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from aw_nas import utils
from aw_nas.common import get_search_space, rollout_from_genotype_str
from aw_nas.evaluator.arch_network import ArchNetwork


def _get_float_format(lst, fmt):
    if isinstance(lst, (float, np.float32, np.float64)):
        return fmt.format(lst)
    if isinstance(lst, (list, tuple)):
        return "[" + ", ".join([_get_float_format(item, fmt) for item in lst]) + "]"
    return "{}".format(lst)

class CellSSDataset(Dataset):
    def __init__(self, data, minus=None, div=None):
        self.data = data
        self._len = len(self.data)
        self.minus = minus
        self.div = div

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.minus is not None:
            data = (data[0], data[1] - self.minus)
        if self.div is not None:
            data = (data[0], data[1] / self.div)
        return data

def _cal_stage_probs(avg_stage_scores, stage_prob_power):
    probs = []
    prob = 1.
    for i_decision, avg_score_stage in enumerate(avg_stage_scores):
        power = stage_prob_power[i_decision] if isinstance(stage_prob_power, (tuple, list)) else stage_prob_power
        avg_score_stage = (avg_score_stage[0] ** power, avg_score_stage[1] ** power)
        cur_prob = prob * avg_score_stage[0] / (avg_score_stage[0] + avg_score_stage[1])
        probs.append(cur_prob)
        prob *= avg_score_stage[1] / (avg_score_stage[0] + avg_score_stage[1])
    probs.append(prob)
    return probs

def make_pair_pool(train_stages, args, stage_epochs):
    num_stages = len(train_stages)
    stage_pairs_list = []
    diff_threshold = args.diff_threshold
    all_train_stages = sum(train_stages, [])
    stage_lens = [len(stage_data) for stage_data in train_stages]
    stage_start_inds = [0] + list(np.cumsum(stage_lens[:-1]))
    for i_stage in range(num_stages):
        stage_data = train_stages[i_stage]
        if i_stage in {0} and num_stages == 4: # try special handling for 4-stage
            beyond_stage_data = sum([train_stages[j_stage] for j_stage in range(i_stage + 2, num_stages)], [])
            other_start_ind = stage_start_inds[i_stage + 2]
        else:
            beyond_stage_data = sum([train_stages[j_stage] for j_stage in range(i_stage, num_stages)], [])
            other_start_ind = stage_start_inds[i_stage]

        epoch = stage_epochs[i_stage]
        # stage_perf
        stage_perf = np.array([d[1][epoch] for d in stage_data])
        beyond_stage_perf = np.array([d[1][epoch] for d in beyond_stage_data])
        diff = beyond_stage_perf - stage_perf[:, None]
        acc_abs_diff_matrix = np.triu(np.abs(diff), 1) - np.tril(np.ones(diff.shape), 0)
        indexes = np.where(acc_abs_diff_matrix > diff_threshold[i_stage])
        pairs = (stage_start_inds[i_stage] + indexes[0], other_start_ind + indexes[1], (diff > 0)[indexes])
        stage_pairs_list.append(pairs)
        logging.info("Num pairs using the perfs of stage {} (epoch {}): {}/{}".format(
            i_stage, epoch, len(pairs[0]),
            stage_lens[i_stage] * (stage_lens[i_stage] - 1) / 2 + stage_lens[i_stage] * sum(stage_lens[i_stage+1:], 0)))
    return all_train_stages, stage_pairs_list

def train_multi_stage_pair_pool(all_stages, pairs_list, model, i_epoch, args):
    objs = utils.AverageMeter()
    model.train()

    # try get through all the pairs
    pairs_pool = list(zip(*[np.concatenate(items) for items in zip(*pairs_list)]))
    num_pairs = len(pairs_pool)
    logging.info("Number of pairs: {}".format(num_pairs))
    np.random.shuffle(pairs_pool)
    num_batch = num_pairs // args.batch_size

    for i_batch in range(num_batch):
        archs_1_inds, archs_2_inds, better_lst = list(zip(
            *pairs_pool[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]))
        loss = model.update_compare(np.array([all_stages[idx][0] for idx in archs_1_inds]),
                                    np.array([all_stages[idx][0] for idx in archs_2_inds]), better_lst)
        objs.update(loss, args.batch_size)
        if i_batch % args.report_freq == 0:
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}".format(
                i_epoch, i_batch, num_batch, objs.avg))
    return objs.avg

def train_multi_stage(train_stages, model, epoch, args, avg_stage_scores, stage_epochs):
    # TODO: multi stage
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()

    num_stages = len(train_stages)
    # must specificy `stage_probs` or `stage_prob_power`
    stage_probs = getattr(args, "stage_probs", None)
    if stage_probs is None:
        stage_probs = _cal_stage_probs(avg_stage_scores, args.stage_prob_power)
    stage_accept_pair_probs = getattr(args, "stage_accept_pair_probs", [1.0] * num_stages)

    stage_lens = [len(stage_data) for stage_data in train_stages]
    for i, len_ in enumerate(stage_lens):
        if len_ == 0:
            n_j = num_stages - i - 1
            for j in range(i + 1, num_stages):
                stage_probs[j] += stage_probs[i] / float(n_j)
            stage_probs[i] = 0
    # diff_threshold = getattr(args, "diff_threshold", [0.08, 0.04, 0.02, 0.0])
    stage_single_probs = getattr(args, "stage_single_probs", None)
    if stage_single_probs is not None:
        stage_probs = np.array([single_prob * len_ for single_prob, len_ in zip(stage_single_probs, stage_lens)])
        stage_probs = stage_probs / stage_probs.sum()
    logging.info("Epoch {:d}: Stage probs {}".format(epoch, stage_probs))

    diff_threshold = args.diff_threshold
    for step in range(args.num_batch_per_epoch):
        pair_batch = []
        i_pair = 0
        while 1:
            stage_1, stage_2 = np.random.choice(np.arange(num_stages), size=2, p=stage_probs)
            d_1 = train_stages[stage_1][np.random.randint(0, stage_lens[stage_1])]
            d_2 = train_stages[stage_2][np.random.randint(0, stage_lens[stage_2])]
            min_stage = min(stage_2, stage_1)
            if np.random.rand() > stage_accept_pair_probs[min_stage]:
                continue
            # max_stage = stage_2 + stage_1 - min_stage
            # if max_stage - min_stage >= 2:
            #     better = stage_2 > stage_1
            # else:
            min_epoch = stage_epochs[min_stage]
            diff_21 = d_2[1][min_epoch] - d_1[1][min_epoch]
            # print(stage_1, stage_2, diff_21, diff_threshold)
            if np.abs(diff_21) > diff_threshold[min_stage]:
                # if the difference is larger than the threshold of the min stage, this pair count
                better = diff_21 > 0
            else:
                continue
            pair_batch.append((d_1[0], d_2[0], better))
            i_pair += 1
            if i_pair == args.batch_size:
                break
        archs_1, archs_2, better_lst = zip(*pair_batch)
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.update_compare(archs_1, archs_2, better_lst)
        objs.update(loss, n_diff_pairs)
        if step % args.report_freq == 0:
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}".format(
                epoch, step, args.num_batch_per_epoch, objs.avg))
    return objs.avg

def train_multi_stage_listwise(train_stages, model, epoch, args, avg_stage_scores, stage_epochs, score_train_stages=None):
    # TODO: multi stage
    objs = utils.AverageMeter()
    n_listlength_meter = utils.AverageMeter()
    model.train()

    num_stages = len(train_stages)

    stage_lens = [len(stage_data) for stage_data in train_stages]
    stage_sep_inds = [np.arange(stage_len) for stage_len in stage_lens]
    sample_acc_temp = getattr(args, "sample_acc_temp", None)
    if sample_acc_temp is not None:
        stage_sep_probs = []
        for i_stage, stage_data in enumerate(train_stages):
            perfs = np.array([item[1][stage_epochs[i_stage]] for item in train_stages[i_stage]])
            perfs = perfs / sample_acc_temp
            exp_perfs = np.exp(perfs - np.max(perfs))
            stage_sep_probs.append(exp_perfs / exp_perfs.sum())
    else:
        stage_sep_probs = None
    stage_single_probs = getattr(args, "stage_single_probs", None)
    assert stage_single_probs is not None
    if stage_single_probs is not None:
        stage_probs = np.array([single_prob * len_ for single_prob, len_ in zip(stage_single_probs, stage_lens)])
        stage_probs = stage_probs / stage_probs.sum()
    logging.info("Epoch {:d}: Stage probs {}".format(epoch, stage_probs))

    num_stage_samples_avg = np.zeros(num_stages)
    train_stages = np.array(train_stages)

    listwise_compare = getattr(args, "listwise_compare", False)
    if listwise_compare:
        assert args.list_length == 2

    for step in range(args.num_batch_per_epoch):
        num_stage_samples = np.random.multinomial(args.list_length, stage_probs)
        num_stage_samples = np.minimum(num_stage_samples, stage_lens)
        true_ll = np.sum(num_stage_samples)
        n_listlength_meter.update(true_ll, args.batch_size)
        num_stage_samples_avg += num_stage_samples
        stage_inds = [np.array([np.random.choice(
            stage_sep_inds[i_stage], size=(sz), replace=False,
            p=None if stage_sep_probs is None else stage_sep_probs[i_stage]) for _ in range(args.batch_size)])
                      if sz > 0 else np.zeros((args.batch_size, 0), dtype=np.int) for i_stage, sz in enumerate(num_stage_samples)]
        sorted_stage_inds = [s_stage_inds[np.arange(args.batch_size)[:, None], np.argsort(np.array(np.array(train_stages[i_stage])[s_stage_inds][:, :, 1].tolist())[:, :, stage_epochs[i_stage]], axis=1)] if s_stage_inds.shape[1] > 1 else s_stage_inds for i_stage, s_stage_inds in enumerate(stage_inds)]
        archs = np.concatenate([np.array(train_stages[i_stage])[s_stage_inds][:, :, 0] for i_stage, s_stage_inds in enumerate(sorted_stage_inds) if s_stage_inds.size > 0], axis=1)
        archs = archs[:, ::-1] # order: from best to worst
        assert archs.ndim == 2
        archs = np.array(archs.tolist()) # (batch_size, list_length, num_cell_groups, node_or_op, decisions)
        if listwise_compare:
            loss = model.update_compare(archs[:, 0], archs[:, 1], np.zeros(archs.shape[0]))
        else:
            loss = model.update_argsort(archs, idxes=None, first_n=getattr(args, "score_list_length", None), is_sorted=True)
        objs.update(loss, args.batch_size)
        if step % args.report_freq == 0:
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f} (mean ll: {:.1f}; {})".format(
                epoch, step, args.num_batch_per_epoch, objs.avg, n_listlength_meter.avg, (num_stage_samples_avg / (step + 1)).tolist()))
    return objs.avg

def train(train_loader, model, epoch, args):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    n_eq_pairs_meter = utils.AverageMeter()
    model.train()

    margin_diff_coeff = getattr(args, "margin_diff_coeff", None)
    eq_threshold = getattr(args, "eq_threshold", None)
    eq_pair_ratio = getattr(args, "eq_pair_ratio", 0)
    if eq_threshold is not None:
        assert eq_pair_ratio > 0
        assert eq_threshold <= args.compare_threshold
    for step, (archs, all_accs) in enumerate(train_loader):
        archs = np.array(archs)
        n = len(archs)
        use_checkpoint = getattr(args, "use_checkpoint", 3)
        accs = all_accs[:, use_checkpoint]
        if args.compare:
            if getattr(args, "compare_split", False):
                n_pairs = len(archs) // 2
                accs = np.array(accs)
                acc_diff_lst = accs[n_pairs:2*n_pairs] - accs[:n_pairs]
                keep_inds = np.where(np.abs(acc_diff_lst) > args.compare_threshold)[0]
                better_lst = (np.array(accs[n_pairs:2*n_pairs] - accs[:n_pairs]) > 0)[keep_inds]
                archs_1 = np.array(archs[:n_pairs])[keep_inds]
                archs_2 = np.array(archs[n_pairs:2*n_pairs])[keep_inds]
            else:
                n_max_pairs = int(args.max_compare_ratio * n * (1 - eq_pair_ratio))
                acc_diff = np.array(accs)[:, None] - np.array(accs)
                acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
                ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
                ex_thresh_num = len(ex_thresh_inds[0])
                if ex_thresh_num > n_max_pairs:
                    if args.choose_pair_criterion == "diff":
                        keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
                    elif args.choose_pair_criterion == "random":
                        keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
                    ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
                archs_1, archs_2, better_lst, acc_diff_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds], acc_diff[ex_thresh_inds]
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            if eq_threshold is None:
                if margin_diff_coeff is not None:
                    margin = np.abs(acc_diff_lst) * margin_diff_coeff
                    loss = model.update_compare(archs_1, archs_2, better_lst, margin=margin)
                else:
                    loss = model.update_compare(archs_1, archs_2, better_lst)
            else:
                # drag close the score of arch pairs whose true acc diffs are below args.eq_threshold
                n_eq_pairs = int(args.max_compare_ratio * n * eq_pair_ratio)
                below_eq_thresh_inds = np.where(acc_abs_diff_matrix < eq_threshold)
                below_eq_thresh_num = len(below_eq_thresh_inds[0])
                if below_eq_thresh_num > n_eq_pairs:
                    keep_inds = np.random.choice(np.arange(below_eq_thresh_num), n_eq_pairs, replace=False)
                    below_eq_thresh_inds = (below_eq_thresh_inds[0][keep_inds], below_eq_thresh_inds[1][keep_inds])
                eq_archs_1, eq_archs_2, below_acc_diff_lst = \
                    archs[below_eq_thresh_inds[1]], archs[below_eq_thresh_inds[0]], acc_abs_diff_matrix[below_eq_thresh_inds]
                if margin_diff_coeff is not None:
                    margin = np.concatenate((
                        np.abs(acc_diff_lst),
                        np.abs(below_acc_diff_lst))) * margin_diff_coeff
                else:
                    margin = None
                better_pm_lst = np.concatenate((2 * better_lst - 1, np.zeros(len(eq_archs_1))))
                n_eq_pairs_meter.update(float(len(eq_archs_1)))
                loss = model.update_compare_eq(np.concatenate((archs_1, eq_archs_1)),
                                               np.concatenate((archs_2, eq_archs_2)),
                                               better_pm_lst, margin=margin)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d}){}".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch,
                    "; eq pairs: {.3d}".format(n_eq_pairs_meter.avg) if eq_threshold is not None else "")
                if args.compare else ""))
    return objs.avg

# ---- test funcs ----
def kat1(true_scores, predict_scores):
    ind = np.argmax(true_scores)
    return list(np.argsort(predict_scores)[::-1]).index(ind) + 1

def katn(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1] + 1
    num_archs = len(ranks)
    # katn for each number
    katns = np.zeros(num_archs)
    passed_set = set()
    cur_ind = 0
    for k in range(1, num_archs+1):
        if k in passed_set:
            katns[k - 1] = katns[k - 2]
        else:
            while ranks[cur_ind] != k:
                passed_set.add(ranks[cur_ind])
                cur_ind += 1
            katns[k - 1] = cur_ind + 1
    ratios = [0.01, 0.05, 0.1, 0.5]
    ks = [0, 4, 9] + [int(r * num_archs) - 1 for r in ratios]
    return [(k + 1, float(k + 1) / num_archs, int(katns[k]), float(katns[k]) / num_archs)
            for k in ks]

def natk(true_scores, predict_scores):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    reorder_predict_scores = predict_scores[predict_inds]
    reorder_true_scores = true_scores[predict_inds]
    ranks = np.argsort(reorder_true_scores)[::-1] + 1
    num_archs = len(ranks)
    # natk for each number
    natks = np.zeros(num_archs)
    passed_set = set()
    cur_ind = 0
    for k in range(1, num_archs+1):
        if k in passed_set:
            natks[k - 1] = natks[k - 2]
        else:
            while ranks[cur_ind] != k:
                passed_set.add(ranks[cur_ind])
                cur_ind += 1
            natks[k - 1] = cur_ind + 1
    ratios = [0.01, 0.05, 0.1, 0.5]
    ks = [0, 4, 9] + [int(r * num_archs) - 1 for r in ratios]
    return [(k + 1, float(k + 1) / num_archs, int(natks[k]), float(natks[k]) / num_archs)
            for k in ks]

def patk(true_scores, predict_scores):
    return [(item[0], item[1], item[3]) for item in test_xk(true_scores, predict_scores, ratios=[0.01, 0.05, 0.1, 0.2, 0.5])]

def test_xp(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    # calculate precision at each point
    cur_inds = np.zeros(num_archs)
    passed_set = set()
    for i_rank, rank in enumerate(ranks):
        cur_inds[i_rank] = (cur_inds[i_rank - 1] if i_rank > 0 else 0) + \
                           int(i_rank in passed_set) + int(rank <= i_rank)
        passed_set.add(rank)
    patks = cur_inds / (np.arange(num_archs) + 1)
    THRESH = 100
    p_corrs = []
    for prec in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        k = np.where(patks[THRESH:] >= prec)[0][0] + THRESH
        arch_inds = ranks[:k][ranks[:k] < k]
        # stats.kendalltau(arch_inds, np.arange(len(arch_inds)))
        p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return p_corrs

def test_xk(true_scores, predict_scores, ratios=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0]):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks
# ---- END test funcs ----


def pairwise_valid(val_loader, model, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    model.eval()
    true_accs = []
    all_archs = []
    for step, (archs, accs) in enumerate(val_loader):
        all_archs += list(archs)
        true_accs += list(accs[:, -1])

    num_valid = len(true_accs)
    pseudo_scores = np.zeros(num_valid)
    indexes = model.argsort_list(all_archs, batch_size=512)
    pseudo_scores[indexes] = np.arange(num_valid)

    corr = stats.kendalltau(true_accs, pseudo_scores).correlation
    funcs_res = [func(true_accs, all_scores) for func in funcs]
    return corr, funcs_res

def sample_batchify(search_space, model, ratio, K, args, conflict_archs=None):
    model.eval()
    inner_sample_n = args.sample_batchify_inner_sample_n
    ss = search_space
    assert K % inner_sample_n == 0
    num_iter = K // inner_sample_n
    want_samples_per_iter = int(ratio * inner_sample_n)
    logging.info("Sample {}. REPEAT {}: Sample {} archs based on the predicted score across {} archs".format(
        K, num_iter, inner_sample_n, want_samples_per_iter))
    sampled_rollouts = []
    sampled_scores = []
    # the number, mean and max predicted scores of current sampled archs
    cur_sampled_mean_max = (0, 0, 0)
    i_iter = 1
    # num_steps = (ratio * K + args.batch_size - 1) // args.batch_size
    _r_cls = ss.random_sample().__class__
    conflict_rollouts = [_r_cls(arch, info={}, search_space=search_space) for arch in conflict_archs or []]
    inner_report_freq = 10
    judget_conflict = False
    while i_iter <= num_iter:
        # # random init
        # if self.inner_iter_random_init \
        #    and hasattr(self.inner_controller, "reinit"):
        #     self.inner_controller.reinit()

        new_per_step_meter = utils.AverageMeter()

        # a list with length self.inner_sample_n
        best_rollouts = []
        best_scores = []
        num_to_sample = inner_sample_n
        iter_r_set = []
        iter_s_set = []
        sampled_r_set = sampled_rollouts
        # for i_inner in range(1, num_steps+1):
        i_inner = 0
        while new_per_step_meter.sum < want_samples_per_iter:
            i_inner += 1
            rollouts = [search_space.random_sample() for _ in range(args.batch_size)]
            batch_archs = [r.arch for r in rollouts]
            step_scores = list(model.predict(batch_archs).cpu().data.numpy())
            if judget_conflict:
                new_inds, new_rollouts = zip(*[(i, r) for i, r in enumerate(rollouts)
                                               if r not in conflict_rollouts
                                               and r not in sampled_r_set
                                               and r not in iter_r_set])
                new_step_scores = [step_scores[i] for i in new_inds]
                iter_r_set += new_rollouts
                iter_s_set += new_step_scores
            else:
                new_rollouts = rollouts
                new_step_scores = step_scores
            new_per_step_meter.update(len(new_rollouts))
            best_rollouts += new_rollouts
            best_scores += new_step_scores
            # iter_r_set += rollouts
            # iter_s_set += step_scores

            if len(best_scores) > num_to_sample:
                keep_inds = np.argpartition(best_scores, -num_to_sample)[-num_to_sample:]
                best_rollouts = [best_rollouts[ind] for ind in keep_inds]
                best_scores = [best_scores[ind] for ind in keep_inds]
            if i_inner % inner_report_freq == 0:
                logging.info(
                    ("Seen %d/%d Iter %d (to sample %d) (already sampled %d mean %.5f, best %.5f); "
                     "Step %d: sample %d step mean %.5f best %.5f: {} "
                     # "(iter mean %.5f, best %.5f).
                     "AVG new/step: %.3f").format(
                         ", ".join(["{:.5f}".format(s) for s in best_scores])),
                    new_per_step_meter.sum, want_samples_per_iter,
                    i_iter, num_to_sample,
                    cur_sampled_mean_max[0], cur_sampled_mean_max[1], cur_sampled_mean_max[2],
                    i_inner, len(rollouts), np.mean(step_scores), np.max(step_scores),
                    #np.mean(iter_s_set), np.max(iter_s_set),
                    new_per_step_meter.avg)
        # if new_per_step_meter.sum < num_to_sample * 10:
        #         # rerun this iter, also reinit!
        #         self.logger.info("Cannot find %d (num_to_sample x min_inner_sample_ratio)"
        #                          " (%d x %d) new rollouts in one run of the inner controller"
        #                          "Re-init the controller and re-run this iteration.",
        #                          num_to_sample * self.min_inner_sample_ratio,
        #                          num_to_sample, self.min_inner_sample_ratio)
        #         continue

        i_iter += 1
        assert len(best_scores) == num_to_sample
        sampled_rollouts += best_rollouts
        sampled_scores += best_scores
        cur_sampled_mean_max = (
            len(sampled_scores), np.mean(sampled_scores), np.max(sampled_scores))

    return [r.genotype for r in sampled_rollouts]

def sample(search_space, model, N, K, args, from_genotypes=None, conflict_archs=None):
    model.eval()
    logging.info("Sample {} archs based on the predicted score across {} archs".format(K, N))
    # ugly
    if from_genotypes is None:
        remain_to_sample = N
        all_archs = []
        all_rollouts = []
        conflict_archs = conflict_archs or []
        while remain_to_sample > 0:
            logging.info("sample {}".format(remain_to_sample))
            while 1:
                rollouts = [search_space.random_sample() for _ in range(remain_to_sample)]
                archs = np.array([r.arch for r in rollouts]).tolist()
                if len(np.unique(archs, axis=0)) == remain_to_sample:
                    break
                loging.info("Resample ...")
            conflict = conflict_archs + all_archs
            remain_to_sample = 0
            indexes = []
            for i, r in enumerate(archs):
                if r in conflict:
                    remain_to_sample += 1
                else:
                    indexes.append(i)
            archs = [archs[i] for i in indexes]
            rollouts = [rollouts[i] for i in indexes]
            all_archs = all_archs + archs
            all_rollouts = all_rollouts + rollouts
    else:
        # rollouts = [rollout_from_genotype_str(geno_str, search_space) for geno_str in from_genotypes]
        all_rollouts = [rollout_from_genotype_str(geno_str, search_space) for geno_str in from_genotypes]
    rollouts = all_rollouts
    logging.info("len sampled: {}".format(len(rollouts)))
    archs = [r.arch for r in rollouts]
    num_batch = (len(archs) + args.batch_size - 1) // args.batch_size
    all_scores = []
    for i_batch in range(num_batch):
        batch_archs = archs[i_batch * args.batch_size: min((i_batch + 1) * args.batch_size, N)]
        scores = list(model.predict(batch_archs).cpu().data.numpy())
        all_scores += scores

    all_scores = np.array(all_scores)
    sorted_inds = np.argsort(all_scores)[::-1][:K]
    # np.where(np.triu(all_scores[sorted_inds] == all_scores[sorted_inds][:, None]).astype(float) - np.eye(N))
    return [rollouts[ind].genotype for ind in sorted_inds]

def valid(val_loader, model, args, funcs=[]):
    if not callable(getattr(model, "predict", None)):
        assert callable(getattr(model, "compare", None))
        corrs, funcs_res = zip(*[
            pairwise_valid(val_loader, model, pv_seed, funcs)
            for pv_seed in getattr(
                    args, "pairwise_valid_seeds", [1, 12, 123]
            )])
        funcs_res = np.mean(funcs_res, axis=0)
        logging.info("pairwise: ", corrs)
        return np.mean(corrs), true_accs, p_scores, funcs_res

    model.eval()
    all_scores = []
    true_accs = []

    for step, (archs, accs) in enumerate(val_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs[:, -1])

    corr = stats.kendalltau(true_accs, all_scores).correlation
    if args.valid_true_split or args.valid_score_split:
        if args.valid_true_split:
            valid_true_split = ["null"] + [float(x) for x in args.valid_true_split.split(",")]
        else:
            valid_true_split = ["null"]
        if args.valid_score_split:
            valid_score_split = ["null"] + [float(x) for x in args.valid_score_split.split(",")]
        else:
            valid_score_split = ["null"]
        tas_list = []
        sas_list = []
        num_unique_tas_list = []
        num_unique_sas_list = []
        for vts in valid_true_split:
            tas = true_accs if vts == "null" else (np.array(true_accs) / vts).astype(int)
            tas_list.append(tas)
            num_unique_tas_list.append(len(np.unique(tas)))
        for vss in valid_score_split:
            sas = all_scores if vss == "null" else (np.array(all_scores) / vss).astype(int)
            sas_list.append(sas)
            num_unique_sas_list.append(len(np.unique(sas)))

        str_ = "{:15}   {}".format("true/score split", "   ".join(["{:5} ({:4})".format(
            split, num_uni) for split, num_uni in zip(valid_score_split, num_unique_sas_list)]))
        for i_tas, tas in enumerate(tas_list):
            s_str_ = "{:15}   ".format("{:5} ({:4})".format(valid_true_split[i_tas], num_unique_tas_list[i_tas])) + \
                     "   ".join(["{:12.4f}".format(stats.kendalltau(tas, sas).correlation) for sas in sas_list])
            str_ = str_ + "\n" + s_str_
        logging.info("Valid kd matrix:\n" + str_)
    funcs_res = [func(true_accs, all_scores) for func in funcs]
    return corr, funcs_res


def main(argv):
    parser = argparse.ArgumentParser(prog="train_cellss_pkl.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--report-freq", default=200, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--train-dir", default=None, help="Save train log/results into TRAIN_DIR")
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--test-only", default=False, action="store_true")
    parser.add_argument("--test-funcs", default=None, help="comma-separated list of test funcs")
    parser.add_argument("--load", default=None, help="Load comparator from disk.")
    parser.add_argument("--sample", default=None, type=int)
    parser.add_argument("--sample-batchify-inner-sample-n", default=None, type=int)
    parser.add_argument("--sample-to-file", default=None, type=str)
    parser.add_argument("--sample-from-file", default=None, type=str)
    parser.add_argument("--sample-conflict-file", default=None, type=str, action="append")
    parser.add_argument("--sample-ratio", default=10, type=float)
    parser.add_argument("--sample-output-dir", default="./sample_output/")
    # parser.add_argument("--data-fname", default="cellss_data.pkl")
    # parser.add_argument("--data-fname", default="cellss_data_round1_999.pkl")
    parser.add_argument("--data-fname", default="enas_data_round1_980.pkl")
    parser.add_argument("--addi-train", default=[], action="append", help="additional train data")
    parser.add_argument("--addi-train-only", action="store_true", default=False)
    parser.add_argument("--addi-valid", default=[], action="append", help="additional valid data")
    parser.add_argument("--addi-valid-only", action="store_true", default=False)
    parser.add_argument("--valid-true-split", default=None)
    parser.add_argument("--valid-score-split", default=None)
    parser.add_argument("--enas-ss", default=True, action="store_true")
    args = parser.parse_args(argv)

    setproctitle.setproctitle("python train_cellss_pkl.py config: {}; train_dir: {}; cwd: {}"\
                              .format(args.cfg_file, args.train_dir, os.getcwd()))

    # log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        # if training, setting up log file, backup config file
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        log_file = os.path.join(args.train_dir, "train.log")
        logging.getLogger().addFile(log_file)

        # copy config file
        backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
        shutil.copyfile(args.cfg_file, backup_cfg_file)
    else:
        backup_cfg_file = args.cfg_file

    # cuda
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = %d" % args.gpu)
    else:
        logging.info("no GPU available, use CPU!!")

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Load pkl cache from cellss_data.pkl")
    data_fname = args.data_fname
    with open(data_fname, "rb") as rf:
        data = pickle.load(rf)
    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f)


    logging.info("Config: %s", cfg)

    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    # search space
    if args.enas_ss:
        ss_cfg_str = """
search_space_type: cnn
search_space_cfg:
  cell_layout: null
  num_cell_groups: 2
  num_init_nodes: 2
  num_layers: 8
  num_node_inputs: 2
  num_steps: 4
  reduce_cell_groups:
  - 1
  shared_primitives:
  - skip_connect
  - sep_conv_3x3
  - sep_conv_5x5
  - avg_pool_3x3
  - max_pool_3x3
        """
    else:
        ss_cfg_str = """
search_space_cfg:
  cell_layout: null
  num_cell_groups: 2
  num_init_nodes: 2
  num_layers: 8
  num_node_inputs: 2
  num_steps: 4
  reduce_cell_groups:
  - 1
  shared_primitives:
  - none
  - max_pool_3x3
  - avg_pool_3x3
  - skip_connect
  - sep_conv_3x3
  - sep_conv_5x5
  - dil_conv_3x3
  - dil_conv_5x5
search_space_type: cnn
        """
    
    ss_cfg = yaml.load(StringIO(ss_cfg_str))
    search_space = get_search_space(ss_cfg["search_space_type"], **ss_cfg["search_space_cfg"])
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    if args.load is not None:
        logging.info("Load %s from %s", arch_network_type, args.load)
        model.load(args.load)
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # init data loaders
    if hasattr(args, "train_size"):
        train_valid_split = args.train_size
    else:
        train_valid_split = int(getattr(args, "train_valid_split", 0.6) * len(data))
    train_data = data[:train_valid_split]
    valid_data = data[train_valid_split:]
    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        _num = len(train_data)
        train_data = train_data[:int(_num * args.train_ratio)]
        logging.info("Train dataset ratio: %.3f", args.train_ratio)
    if args.addi_train:
        if args.addi_train_only:
            train_data = []
        for addi_train_fname in args.addi_train:
            with open(addi_train_fname, "rb") as rf:
                addi_train_data = pickle.load(rf)
            train_data += addi_train_data
    if args.addi_valid:
        if args.addi_valid_only:
            valid_data = []
        for addi_fname in args.addi_valid:
            with open(addi_fname, "rb") as rf:
                addi_valid_data = pickle.load(rf)
            valid_data += addi_valid_data
    num_train_archs = len(train_data)
    logging.info("Number of architectures: train: %d; valid: %d", num_train_archs, len(valid_data))
    train_data = CellSSDataset(train_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    valid_data = CellSSDataset(valid_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list([np.array(x) for x in zip(*items)]))
    val_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list([np.array(x) for x in zip(*items)]))

    if args.test_funcs is not None:
        test_func_names = args.test_funcs.split(",")
        test_funcs = [globals()[func_name] for func_name in test_func_names]
    else:
        test_funcs = []

    # init test
    if not arch_network_type == "pairwise_comparator" or args.test_only:
        corr, func_res = valid(val_loader, model, args, funcs=test_funcs)

        if args.sample is not None:
            if args.sample_from_file:
                logging.info("Read genotypes from: {}".format(args.sample_from_file))
                with open(args.sample_from_file, "r") as rf:
                    from_genotypes = yaml.load(rf)
                assert len(from_genotypes) == args.sample * int(args.sample_ratio)
            else:
                from_genotypes = None
            if args.sample_conflict_file:
                conflict_archs = []
                for scf in args.sample_conflict_file:
                    conflict_archs += pickle.load(open(scf, "rb"))
            else:
                conflict_archs = None
            if args.sample_batchify_inner_sample_n is not None:
                # do not support multi-stage now
                genotypes = sample_batchify(search_space, model, args.sample_ratio, args.sample, args, conflict_archs=conflict_archs)
            else:
                genotypes = sample(search_space, model, args.sample * int(args.sample_ratio), args.sample, args, from_genotypes=from_genotypes, conflict_archs=conflict_archs)
            if args.sample_to_file:
                with open(args.sample_to_file, "w") as wf:
                    yaml.dump([str(geno) for geno in genotypes], wf)
            else:
                with open("./final_template.yaml", "r") as rf:
                    logging.info("Load final template config file!")
                    template_cfg = yaml.load(rf)
                for i, genotype in enumerate(genotypes):
                    sample_cfg = copy.deepcopy(template_cfg)
                    sample_cfg["final_model_cfg"]["genotypes"] = str(genotype)
                    if not os.path.exists(args.sample_output_dir):
                        os.makedirs(args.sample_output_dir)
                    with open(os.path.join(args.sample_output_dir, "{}.yaml".format(i)), "w") as wf:
                        yaml.dump(sample_cfg, wf)

        if args.test_funcs is None:
            logging.info("INIT: kendall tau {:.4f}".format(corr))
        else:
            logging.info("INIT: kendall tau {:.4f};\n\t{}".format(
                corr,
                "\n\t".join(["{}: {}".format(name, _get_float_format(res, "{:.4f}"))
                             for name, res in zip(test_func_names, func_res)])))

    if args.test_only:
        return

    _multi_stage = getattr(args, "multi_stage", False)
    _multi_stage_pair_pool = getattr(args, "multi_stage_pair_pool", False)
    if _multi_stage:
        all_perfs = np.array([item[1] for item in train_data.data])
        all_inds = np.arange(all_perfs.shape[0])

        stage_epochs = getattr(args, "stage_epochs", [0, 1, 2, 3])
        num_stages = len(stage_epochs)

        default_stage_nums = [all_perfs.shape[0] // num_stages] * (num_stages - 1) + \
            [all_perfs.shape[0] - all_perfs.shape[0] // num_stages * (num_stages - 1)]
        stage_nums = getattr(args, "stage_nums", default_stage_nums)

        assert np.sum(stage_nums) == all_perfs.shape[0]
        logging.info("Stage nums: {}".format(stage_nums))

        stage_inds_lst = []
        for i_stage in range(num_stages):
            max_stage_ = np.max(all_perfs[all_inds, stage_epochs[i_stage]])
            min_stage_ = np.min(all_perfs[all_inds, stage_epochs[i_stage]])
            logging.info("Stage {}, epoch {}: min {:.2f} %; max {:.2f}% (range {:.2f} %)".format(
                i_stage, stage_epochs[i_stage], min_stage_ * 100, max_stage_ * 100, (max_stage_ - min_stage_) * 100))
            sorted_inds = np.argsort(all_perfs[all_inds, stage_epochs[i_stage]])
            stage_inds, all_inds = all_inds[sorted_inds[:stage_nums[i_stage]]],\
                                            all_inds[sorted_inds[stage_nums[i_stage]:]]
            stage_inds_lst.append(stage_inds)
        train_stages = [[train_data.data[ind] for ind in _stage_inds]
                        for _stage_inds in stage_inds_lst]
        avg_score_stages = []
        for i_stage in range(num_stages - 1):
            avg_score_stages.append((all_perfs[stage_inds_lst[i_stage], stage_epochs[i_stage]].sum(),
                                     np.sum([all_perfs[stage_inds_lst[j_stage], stage_epochs[i_stage]].sum()
                                             for j_stage in range(i_stage+1, num_stages)])))
        if _multi_stage_pair_pool:
            all_stages, pairs_list = make_pair_pool(train_stages, args, stage_epochs)

        total_eval_time = all_perfs.shape[0] * all_perfs.shape[1]
        multi_stage_eval_time = sum([(stage_epochs[i_stage] + 1) * len(_stage_inds)
                                     for i_stage, _stage_inds in enumerate(stage_inds_lst)])
        logging.info("Percentage of evaluation time: {:.2f} %".format(float(multi_stage_eval_time) / total_eval_time * 100))

    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        if _multi_stage:
            if _multi_stage_pair_pool:
                avg_loss = train_multi_stage_pair_pool(all_stages, pairs_list, model, i_epoch, args)
            else:
                if getattr(args, "use_listwise", False):
                    avg_loss = train_multi_stage_listwise(train_stages, model, i_epoch, args, avg_score_stages, stage_epochs)
                else:
                    avg_loss = train_multi_stage(train_stages, model, i_epoch, args, avg_score_stages, stage_epochs)
        else:
            avg_loss = train(train_loader, model, i_epoch, args)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        train_corr, train_func_res = valid(
            train_loader, model, args,
            funcs=test_funcs)
        if args.test_funcs is not None:
            for name, res in zip(test_func_names, train_func_res):
                logging.info("Train: Epoch {:3d}: {}: {}".format(i_epoch, name, _get_float_format(res, "{:.4f}")))
        logging.info("Train: Epoch {:3d}: train kd {:.4f}".format(i_epoch, train_corr))
        corr, func_res = valid(
            val_loader, model, args,
            funcs=test_funcs)
        if args.test_funcs is not None:
            for name, res in zip(test_func_names, func_res):
                logging.info("Valid: Epoch {:3d}: {}: {}".format(i_epoch, name, _get_float_format(res, "{:.4f}")))
        logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, corr))
        if args.save_every is not None and i_epoch % args.save_every == 0:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
