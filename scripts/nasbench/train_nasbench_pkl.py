# -*- coding: utf-8 -*-
# pylint: disable-all
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
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from nasbench import api

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork


class NasBench101Dataset(Dataset):
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
            data = (data[0], data[1] - self.minus, data[2] - self.minus)
        if self.div is not None:
            data = (data[0], data[1] / self.div, data[2] / self.div)
        return data

def train(train_loader, model, epoch, args):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, f_accs, h_accs) in enumerate(train_loader):
        n = len(archs)
        if getattr(args, "use_half", False):
            accs = h_accs
        else:
            accs = f_accs
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
                archs = np.array(archs)
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

def pairwise_valid(val_loader, model, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    model.eval()
    true_accs = []
    all_archs = []
    for step, (archs, accs, _) in enumerate(val_loader):
        all_archs += list(archs)
        true_accs += list(accs)

    num_valid = len(true_accs)
    pseudo_scores = np.zeros(num_valid)
    indexes = model.argsort_list(all_archs, batch_size=512)
    pseudo_scores[indexes] = np.arange(num_valid)

    corr = stats.kendalltau(true_accs, pseudo_scores).correlation
    return corr

def valid(val_loader, model, args):
    if not callable(getattr(model, "predict", None)):
        assert callable(getattr(model, "compare", None))
        corrs = [pairwise_valid(val_loader, model, pv_seed)
                 for pv_seed in getattr(
                         args, "pairwise_valid_seeds", [1, 12, 123]
                 )]
        logging.info("pairwise: ", corrs)
        return np.mean(corrs)

    model.eval()
    all_scores = []
    true_accs = []
    for step, (archs, accs, _) in enumerate(val_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    return corr

def main(argv):
    parser = argparse.ArgumentParser(prog="train_nasbench_pkl.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--report_freq", default=200, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--save-every", default=10, type=int)
    args = parser.parse_args(argv)

    # log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    log_file = os.path.join(args.train_dir, "train.log")
    logging.getLogger().addFile(log_file)

    # copy config file
    backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
    shutil.copyfile(args.cfg_file, backup_cfg_file)

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

    if not os.path.exists("./nasbench_7v.pkl"):
        # init nasbench search space, might take several minutes
        search_space = get_search_space("nasbench-101")
        # sort according to hash, direct group without using get_metrics_from_spec
        fixed_statistics = list(search_space.nasbench.fixed_statistics.items())
        # only handle archs with 7 nodes for efficient batching
        fixed_statistics = [stat for stat in fixed_statistics
                            if stat[1]["module_adjacency"].shape[0] == 7]
        logging.info("Number of arch data: {}".format(len(fixed_statistics)))
        valid_ratio = 0.1
        num_valid = int(len(fixed_statistics) * valid_ratio)
        train_data = []
        for key, f_metric in fixed_statistics[:-num_valid]:
            arch = (f_metric["module_adjacency"], search_space.op_to_idx(f_metric["module_operations"]))
            metrics = search_space.nasbench.computed_statistics[key]
            valid_acc = np.mean([metrics[108][i]["final_validation_accuracy"] for i in range(3)])
            half_valid_acc = np.mean([metrics[108][i]["halfway_validation_accuracy"]
                                      for i in range(3)])
            train_data.append((arch, valid_acc, half_valid_acc))

        valid_data = []
        for key, f_metric in fixed_statistics[-num_valid:]:
            arch = (f_metric["module_adjacency"], search_space.op_to_idx(f_metric["module_operations"]))
            metrics = search_space.nasbench.computed_statistics[key]
            valid_acc = np.mean([metrics[108][i]["final_validation_accuracy"] for i in range(3)])
            half_valid_acc = np.mean([metrics[108][i]["halfway_validation_accuracy"]
                                      for i in range(3)])
            valid_data.append((arch, valid_acc, half_valid_acc))

        with open("nasbench_7v.pkl", "wb") as wf:
            pickle.dump(train_data, wf)
        with open("nasbench_7v_valid.pkl", "wb") as wf:
            pickle.dump(valid_data, wf)
    else:
        search_space = get_search_space("nasbench-101", load_nasbench=False)
        logging.info("Load pkl cache from nasbench_7v.pkl and nasbench_7v_valid.pkl")
        with open("nasbench_7v.pkl", "rb") as rf:
            train_data = pickle.load(rf)
        with open("nasbench_7v_valid.pkl", "rb") as rf:
            valid_data = pickle.load(rf)

    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f)

    logging.info("Config: %s", cfg)

    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # init nasbench data loaders
    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        _num = len(train_data)
        train_data = train_data[:int(_num * args.train_ratio)]
        logging.info("Train dataset ratio: %.3f", args.train_ratio)
    logging.info("Number of architectures: train: %d; valid: %d", len(train_data), len(valid_data))
    train_data = NasBench101Dataset(train_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    valid_data = NasBench101Dataset(valid_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list(zip(*items)))

    # init test
    if not arch_network_type == "pairwise_comparator":
        # the comparison-based `argsort_list` would be very slow
        # if the partial order is totally violated, cauz in this case 
        # the score is far from meaningful to be utilized to get relative balanced split.
        # for now, we omit the initial correlation testing in this pairwise case
        # 3~4 s for pointwise. How about stop evaluate small segments, which is
        # not efficient, e.g. when len(inds) < N just do not do recursion.
        # threshold 1: 450s; threshold 50: 70s; threshold 100: 60s
        corr = valid(val_loader, model, args)
        logging.info("INIT: kendall tau {:.4f}".format(corr))

    for i_epoch in range(1, args.epochs + 1):
        avg_loss = train(train_loader, model, i_epoch, args)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        corr = valid(val_loader, model, args)
        logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, corr))
        if i_epoch % args.save_every == 0:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
