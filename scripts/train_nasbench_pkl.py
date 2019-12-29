# -*- coding: utf-8 -*-
# pylint: disable-all
import os
import sys
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
from aw_nas.evaluator.arch_network import PointwiseComparator


class NasBench101Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self._len = len(self.data)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.data[idx]

def train(train_loader, model, epoch):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, accs) in enumerate(train_loader):
        n = len(archs)
        if args.compare:
            n_max_pairs = args.max_compare_ratio * n
            acc_diff = np.array(accs)[:, None] - np.array(accs)
            acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
            ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
            ex_thresh_num = len(ex_thresh_inds[0])
            if ex_thresh_num > n_max_pairs:
                if args.choose_pair_criterion == "diff":
                    keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
                elif args.choose_pair_criterion == "random":
                    keep_inds = np.random.choice(np.arange(ex_thresh_num()), n_max_pairs, replace=False)
                ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
            archs = np.array(archs)
            archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]
            # # this list comprehension spend more than 3 times than `update_compare`
            # pairs, abs_diff = zip(*[
            #     ((archs[i], archs[j], accs[j] > accs[i]), acc_diff_matrix[j, i])
            #     for i in range(len(archs)) for j in range(i)
            #     if acc_diff_matrix[j, i] > args.compare_threshold])
            # if len(pairs) > n_max_pairs:
            #     if args.choose_pair_criterion == "random":
            #         pairs = list(np.random.choice(pairs, size=n_max_pairs, replace=False))
            #     elif args.choose_pair_criterion == "diff":
            #         inds = np.argpartition(abs_diff, -n_max_pairs)[-n_max_pairs:]
            #         pairs = [pairs[ind] for ind in inds]
            # archs_1, archs_2, better_lst = zip(*pairs)
            # archs_1, archs_2, better_lst = zip(*[
            #     (archs[i], archs[j], accs[j] > accs[i])
            #     for i in range(len(archs)) for j in range(i)
            #     if np.abs(accs[j] - accs[i]) > args.compare_threshold])
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

def valid(val_loader, model):
    model.eval()
    all_scores = []
    true_accs = []
    for step, (archs, accs) in enumerate(val_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    return corr

parser = argparse.ArgumentParser()
parser.add_argument("cfg_file")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch-size", default=512, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--report_freq", default=200, type=int)
parser.add_argument("--compare", default=False, action="store_true")
parser.add_argument("--compare-threshold", default=0.01, type=float)
parser.add_argument("--max-compare-ratio", default=4, type=int)
parser.add_argument("--choose-pair-criterion", default="diff", choices=["diff", "random"])
parser.add_argument("--eval-batch-size", default=None, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--train-dir", required=True)
parser.add_argument("--save-every", default=10, type=int)
args = parser.parse_args()

# log
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m/%d %I:%M:%S %p")
if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)
log_file = os.path.join(args.train_dir, "train.log")
logging.getLogger().addFile(log_file)

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
    fixed_statistics = list(search_space.nasbench.fixed_statistics.values())
    fixed_statistics = [stat for stat in fixed_statistics if stat["module_adjacency"].shape[0] == 7] # FIXME: temporary
    print("Number of arch data: {}".format(len(fixed_statistics)))
    valid_ratio = 0.1
    num_valid = int(len(fixed_statistics) * valid_ratio)
    train_data = []
    for f_metric in fixed_statistics[:-num_valid]:
        arch = (f_metric["module_adjacency"], search_space.op_to_idx(f_metric["module_operations"]))
        spec = api.ModelSpec(f_metric["module_adjacency"], f_metric["module_operations"])
        metrics = search_space.nasbench.get_metrics_from_spec(spec)
        valid_acc = np.mean([metrics[1][108][i]["final_validation_accuracy"] for i in range(3)])
        train_data.append((arch, valid_acc))

    valid_data = []
    for f_metric in fixed_statistics[-num_valid:]:
        arch = (f_metric["module_adjacency"], search_space.op_to_idx(f_metric["module_operations"]))
        spec = api.ModelSpec(f_metric["module_adjacency"], f_metric["module_operations"])
        metrics = search_space.nasbench.get_metrics_from_spec(spec)
        valid_acc = np.mean([metrics[1][108][i]["final_validation_accuracy"] for i in range(3)])
        valid_data.append((arch, valid_acc))
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

# init nasbench data loaders
train_data = NasBench101Dataset(train_data)
valid_data = NasBench101Dataset(valid_data)
train_loader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers,
    collate_fn=lambda items: list(zip(*items)))
val_loader = DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers,
    collate_fn=lambda items: list(zip(*items)))

with open(args.cfg_file, "r") as cfg_f:
    cfg = yaml.load(cfg_f)

model = PointwiseComparator(search_space, **cfg)
model.to(device)

# init test
corr = valid(val_loader, model)
logging.info("INIT: kendall tau {:.4f}".format(corr))

for i_epoch in range(1, args.epochs + 1):
    avg_loss = train(train_loader, model, i_epoch)
    logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
    corr = valid(val_loader, model)
    logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, corr))
    if i_epoch % args.save_every == 0:
        save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
        model.save(save_path)
        logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))
