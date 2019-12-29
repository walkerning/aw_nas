# -*- coding: utf-8 -*-
import sys
import logging
import argparse
import yaml
import random
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
    def __init__(self, nb_search_space, fixed_statistics=None):
        self.ss = nb_search_space
        self.nasbench = self.ss.nasbench
        self.multi_fidelity = self.ss.multi_fidelity
        if fixed_statistics is not None:
            self.fixed_statistics = fixed_statistics
        else:
            self.fixed_statistics = list(self.nasbench.fixed_statistics.values())

    def __len__(self):
        return len(self.fixed_statistics)

    def __getitem__(self, idx):
        f_metric = self.fixed_statistics[idx]
        arch = (f_metric["module_adjacency"], self.ss.op_to_idx(f_metric["module_operations"]))
        spec = api.ModelSpec(f_metric["module_adjacency"], f_metric["module_operations"])
        metrics = self.nasbench.get_metrics_from_spec(spec)
        valid_acc = np.mean([metrics[1][108][i]["final_validation_accuracy"] for i in range(3)])
        return arch, valid_acc

def train(train_loader, model, epoch):
    objs = utils.AverageMeter()
    model.train()
    for step, (archs, accs) in enumerate(train_loader):
        if args.compare:
            archs_1, archs_2, better_lst = zip(*[(archs[i], archs[j], accs[j] > accs[i])
                                                 for i in range(len(archs)) for j in range(i)
                                                 if np.abs(accs[j] - accs[i]) > args.compare_threshold])
            loss = model.update_compare(archs_1, archs_2, better_lst)
        else:
            loss = model.update_predict(archs, accs)
        n = len(archs)
        objs.update(loss, n)
        if step % args.report_freq == 0:
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}".format(epoch, step, len(train_loader), objs.avg))
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
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument("--epochs", default=100, type=int)    
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--report_freq", default=100, type=int)
parser.add_argument("--compare", default=False, action="store_true")
parser.add_argument("--compare-threshold", default=0., type=float)
parser.add_argument("--eval-batch-size", default=None, type=int)
parser.add_argument("--seed", default=None, type=int)
args = parser.parse_args()

# cuda
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('GPU device = %d' % args.gpu)
else:
    logging.info('no GPU available, use CPU!!')

if args.seed is not None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init nasbench search space, might take several minutes
search_space = get_search_space("nasbench-101")
fixed_statistics = list(search_space.nasbench.fixed_statistics.values())
fixed_statistics = [stat for stat in fixed_statistics if stat["module_adjacency"].shape[0] == 7] # FIXME: temporary
print("Number of arch data: {}".format(len(fixed_statistics)))

# init nasbench data loaders
valid_ratio = 0.1
num_valid = int(len(fixed_statistics) * valid_ratio)
train_data = NasBench101Dataset(search_space, fixed_statistics[:-num_valid])
valid_data = NasBench101Dataset(search_space, fixed_statistics[-num_valid:])
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

for i_epoch in range(args.epochs):
    avg_loss = train(train_loader, model, i_epoch)
    print("Epoch {:03d}: train loss {:.4f}".format(i_epoch, avg_loss))
    corr = valid(val_loader, model)
    print("valid {:03d}: kendall tau {:.4f}".format(i_epoch, corr))
