# -*- coding: utf-8 -*-
# pylint: disable-all

import sys
import os
import random
import logging
import shutil
from typing import Tuple

import pickle
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import stats
import numpy as np


def prepare(args) -> Tuple[str, torch.device]:
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream = sys.stdout, level = logging.INFO,
                        format = log_format, datefmt = "%m/%d %I:%M:%S %p")

    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        backup_cfg_file = _make_dir(args.train_dir, args.cfg_file)
    else:
        backup_cfg_file = args.cfg_file
    
    device = _set_device(args.gpu)
    
    if args.seed is not None:
        _set_seed(args.seed)

    return backup_cfg_file, device


def valid(val_loader, model, args, save_path: str = None):
    """
    Test the predictor on the validation set.
    """
    model.eval()
    all_scores = []
    all_low_fidelity = []
    all_real_accs = []
    for step, (archs, real_accs, low_fidelity) in enumerate(val_loader):
        archs = np.array(archs)
        low_fidelity = np.array(low_fidelity)
        real_accs = np.array(real_accs)
        n = len(archs)
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        all_low_fidelity += list(low_fidelity)
        all_real_accs += list(real_accs)
    
    low_fidelity_corr = stats.kendalltau(all_low_fidelity, all_scores).correlation
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {"real_acc": all_real_accs, "score": all_scores}
        with open(os.path.join(save_path, "output.pkl"), "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk


def nb301_valid(val_loader, model, args, save_path: str = None):
    """
    Test the predictor trained on the NAS-Bench-301 search space on the validation set.
    """
    model.eval()
    all_scores = []
    all_real_accs = []
    for step, (archs, real_accs) in enumerate(val_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        n = len(archs)
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        all_real_accs += list(real_accs)
    
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)
    
    if save_path:
        data = {"real_acc": all_real_accs, "score": all_scores}
        with open(os.path.join(save_path, "output.pkl"), "wb") as wf:
            pickle.dump(data, wf)

    return real_corr, patk


def compare_data(archs, f_accs, accs, args):
    if None in f_accs:
        # some archs only have half-time acc
        n_max_pairs = int(args.max_compare_ratio * n)
        n_max_inter_pairs = int(args.inter_pair_ratio * n_max_pairs)
        half_inds = np.array([ind for ind, acc in enumerate(accs) if acc is None])
        mask = np.zeros(n)
        mask[half_inds] = 1
        final_inds = np.where(1 - mask)[0]

        half_eche = h_accs[half_inds]
        final_eche = h_accs[final_inds]
        half_acc_diff = final_eche[:, None] - half_eche # (num_final, num_half)
        assert (half_acc_diff >= 0).all() # should be >0
        half_ex_thresh_inds = np.where(np.abs(half_acc_diff) > getattr(
            args, "half_compare_threshold", 2 * args.compare_threshold))
        half_ex_thresh_num = len(half_ex_thresh_inds[0])
        if half_ex_thresh_num > n_max_inter_pairs:
            # random choose
            keep_inds = np.random.choice(np.arange(half_ex_thresh_num), n_max_inter_pairs, replace = False)
            half_ex_thresh_inds = (half_ex_thresh_inds[0][keep_inds], half_ex_thresh_inds[1][keep_inds])
        inter_archs_1, inter_archs_2, inter_better_lst = \
                    archs[half_inds[half_ex_thresh_inds[1]]], archs[final_inds[half_ex_thresh_inds[0]]], \
                    (half_acc_diff > 0)[half_ex_thresh_inds]
        n_inter_pairs = len(inter_better_lst)

        # only use intra pairs in the final echelon
        n_intra_pairs = n_max_pairs - n_inter_pairs
        accs = np.array(accs)[final_inds]
        archs = archs[final_inds]
        acc_diff = np.array(accs)[:, None] - np.array(accs)
        acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
        ex_thresh_num = len(ex_thresh_inds[0])
        if ex_thresh_num > n_intra_pairs:
            if args.choose_pair_criterion == "diff":
                keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_intra_pairs)[-n_intra_pairs:]
            elif args.choose_pair_criterion == "random":
                keep_inds = np.random.choice(np.arange(ex_thresh_num), n_intra_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
        archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]],\
                                       archs[ex_thresh_inds[0]],\
                                       (acc_diff > 0)[ex_thresh_inds]
        archs_1, archs_2, better_lst = np.concatenate((inter_archs_1, archs_1)),\
                                       np.concatenate((inter_archs_2, archs_2)),\
                                       np.concatenate((inter_better_lst, better_lst))
    else:
        if getattr(args, "compare_split", False):
            n_pairs = len(archs) // 2
            accs = np.array(accs)
            acc_diff_lst = accs[n_pairs:2*n_pairs] - accs[:n_pairs]
            keep_inds = np.where(np.abs(acc_diff_lst) > args.compare_threshold)[0]
            better_lst = (np.array(accs[n_pairs:2*n_pairs] - accs[:n_pairs]) > 0)[keep_inds]
            archs_1 = np.array(archs[:n_pairs])[keep_inds]
            archs_2 = np.array(archs[n_pairs:2*n_pairs])[keep_inds]
        else:
            n_max_pairs = int(args.max_compare_ratio * len(archs))
            acc_diff = np.array(accs)[:, None] - np.array(accs)
            acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
            ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
            ex_thresh_num = len(ex_thresh_inds[0])
            if ex_thresh_num > n_max_pairs:
                if args.choose_pair_criterion == "diff":
                    keep_inds = np.argpartition(
                            acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
                    ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
                elif args.choose_pair_criterion == "random":
                    keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
                    ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
            archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]

    return archs_1, archs_2, better_lst


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
        p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return p_corrs


def test_xk(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks


def _make_dir(train_dir: str, cfg_file: str):
    # if training, setting up log file, backup config file
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    log_file = os.path.join(train_dir, "train.log")
    logging.getLogger().addFile(log_file)

    # copy config file
    backup_cfg_file = os.path.join(train_dir, "config.yaml")
    shutil.copyfile(cfg_file, backup_cfg_file)
    
    return backup_cfg_file


def _set_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = %d" % gpu)
        device = torch.device("cuda")
    else:
        logging.info("no GPU available, use CPU!!")
        device = torch.device("cpu")
    return device


def _set_seed(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def mtl_valid(val_loader, model, args, save_path: str = None):
    """
    Test the predictor on the validation set. 

    Returns:
        low_fidelity_corr (Dict[str, float]): The Kendall's Tau correlation between different
            low-fidelity experts' predicted scores and the actual low-fidelity information.
        real_corr (float): The Kendall's Tau correlation between the predicted scores and the
            actual performance.
        patk: P@topk.
    """
    model.eval()
    all_scores = []
    all_real_accs = []
    all_low_fidelity = {low_fidelity: [] for low_fidelity in args.low_fidelity_type}
    all_low_fidelity_scores = {low_fidelity: [] for low_fidelity in args.low_fidelity_type}

    for step, (archs, real_accs, low_fidelity_perfs) in enumerate(val_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        
        n = len(archs)

        scores, auxiliary_scores_lst = model.mtl_predict(archs)
        scores = list(scores.cpu().data.numpy())
        all_scores += scores
        all_real_accs += list(real_accs)

        for i, low_fidelity in enumerate(args.low_fidelity_type):
            all_low_fidelity[low_fidelity] += [perf[low_fidelity] for perf in low_fidelity_perfs]
            all_low_fidelity_scores[low_fidelity] += list(auxiliary_scores_lst[i].cpu().data.numpy())

    low_fidelity_corr = {
        low_fidelity: stats.kendalltau(all_low_fidelity[low_fidelity], 
                                       all_low_fidelity_scores[low_fidelity]).correlation
        for low_fidelity in args.low_fidelity_type
    }
    real_corr = stats.kendalltau(all_real_accs, all_scores).correlation
    patk = test_xk(all_real_accs, all_scores)

    if save_path:
        data = {
            "real_acc": all_real_accs, 
            "score": all_scores, 
            "low_fidelity": all_low_fidelity,
            "low_fidelity_score": all_low_fidelity_scores
        }
        with open(os.path.join(save_path, "output.pkl"), "wb") as wf:
            pickle.dump(data, wf)

    return low_fidelity_corr, real_corr, patk
