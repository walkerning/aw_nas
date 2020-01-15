# -*- coding: utf-8 -*-
# pylint: disable-all

from io import StringIO
import os
import sys
import shutil
import logging
import argparse
import random
import pickle

import yaml
from scipy.stats import stats

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork


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

def train_multi_stage(train_stages, model, epoch, args, avg_stage_scores):
    # TODO: multi stage
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()

    # must specificy `stage_probs` or `stage_prob_power`
    stage_probs = getattr(args, "stage_probs", None)
    if stage_probs is None:
        stage_probs = _cal_stage_probs(avg_stage_scores, args.stage_prob_power)
    logging.info("Epoch {:d}: Stage probs {}".format(epoch, stage_probs))

    stage_lens = [len(stage_data) for stage_data in train_stages]
    # diff_threshold = getattr(args, "diff_threshold", [0.08, 0.04, 0.02, 0.0])
    diff_threshold = args.diff_threshold
    for step in range(args.num_batch_per_epoch):
        # num_stage_samples = np.multinomial(args.batch_size, stage_probs)
        # sampled_stage_inds = [np.random.choice(np.arange(stage_lens[i_stage]), num_sample, replace=False)
        #                       for i_stage, (stage_data, num_sample) in enumerate(zip(train_stages, num_stage_samples))]
        
        pair_batch = []
        i_pair = 0
        while 1:
            stage_1, stage_2 = np.random.choice(np.arange(4), size=2, p=stage_probs)
            d_1 = train_stages[stage_1][np.random.randint(0, stage_lens[stage_1])]
            d_2 = train_stages[stage_2][np.random.randint(0, stage_lens[stage_2])]
            min_stage = min(stage_2, stage_1)
            diff_21 = d_2[1][min_stage] - d_1[1][min_stage]
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
    
def train(train_loader, model, epoch, args):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
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
                n_max_pairs = int(args.max_compare_ratio * n)
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
                archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))
    return objs.avg

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
    funcs_res = [func(true_accs, all_scores) for func in funcs]
    return corr, funcs_res

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

def main(argv):
    parser = argparse.ArgumentParser(prog="train_cellss_pkl.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--report_freq", default=200, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--train-dir", default=None, help="Save train log/results into TRAIN_DIR")
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--test-only", default=False, action="store_true")
    parser.add_argument("--test-funcs", default=None, help="comma-separated list of test funcs")
    parser.add_argument("--load", default=None, help="Load comparator from disk.")
    args = parser.parse_args(argv)

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
    with open("cellss_data.pkl", "rb") as rf:
        data = pickle.load(rf)
    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f)


    logging.info("Config: %s", cfg)

    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
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
    train_valid_split = int(getattr(args, "train_valid_split", 0.6) * len(data))
    train_data = data[:train_valid_split]
    valid_data = data[train_valid_split:]
    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        _num = len(train_data)
        train_data = train_data[:int(_num * args.train_ratio)]
        logging.info("Train dataset ratio: %.3f", args.train_ratio)
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

    # init test
    if not arch_network_type == "pairwise_comparator" or args.test_only:
        if args.test_funcs is not None:
            test_func_names = args.test_funcs.split(",")
        corr, func_res = valid(val_loader, model, args,
                               funcs=[globals()[func_name] for func_name in test_func_names]
                               if args.test_funcs is not None else [])
        if args.test_funcs is None:
            logging.info("INIT: kendall tau {:.4f}".format(corr))
        else:
            logging.info("INIT: kendall tau {:.4f};\n\t{}".format(
                corr,
                "\n\t".join(["{}: {}".format(name, res)
                             for name, res in zip(test_func_names, func_res)])))

    if args.test_only:
        return

    _multi_stage = getattr(args, "multi_stage", False)
    if _multi_stage:
        all_perfs = np.array([item[1] for item in train_data.data])
        all_inds = np.arange(all_perfs.shape[0])

        default_stage_nums = [all_perfs.shape[0] // 4] * 3 + [all_perfs.shape[0] - all_perfs.shape[0] // 4 * 3]
        stage_nums = getattr(args, "stage_nums", default_stage_nums)
        assert np.sum(stage_nums) == all_perfs.shape[0]
        logging.info("Stage nums: {}".format(stage_nums))

        stage_inds_lst = []
        for i_stage in range(4):
            sorted_inds = np.argsort(all_perfs[all_inds, i_stage])
            stage_inds, all_inds = all_inds[sorted_inds[:stage_nums[i_stage]]],\
                                            all_inds[sorted_inds[stage_nums[i_stage]:]]
            stage_inds_lst.append(stage_inds)
        train_stages = [[train_data.data[ind] for ind in stage_inds] for stage_inds in stage_inds_lst]
        avg_score_stages = []
        for i_stage in range(3):
            avg_score_stages.append((all_perfs[stage_inds_lst[i_stage], i_stage].sum(),
                                     np.sum([all_perfs[stage_inds_lst[j_stage], i_stage].sum() for j_stage in range(i_stage+1, 4)])))

    for i_epoch in range(1, args.epochs + 1):
        if _multi_stage:
            avg_loss = train_multi_stage(train_stages, model, i_epoch, args, avg_score_stages)
        else:
            avg_loss = train(train_loader, model, i_epoch, args)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        corr, _ = valid(val_loader, model, args)
        logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, corr))
        if args.save_every is not None and i_epoch % args.save_every == 0:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
