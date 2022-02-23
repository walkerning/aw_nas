# -*- coding: utf-8 -*-
# pylint: disable-all

"""
Train the dynamic ensemble architecture-performance predictor on 
NAS-Bench-201 / NAS-Bench-301 / NDS-ResNet/ResNeXT-A.

Copyright (c) 2022 Junbo Zhao, Xuefei Ning
"""


import os
import sys
import logging
import argparse
import pickle
import yaml
from typing import List

import numpy as np
import setproctitle
from torch.utils.data import DataLoader

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork

from utils import compare_data, prepare, mtl_valid, nb301_valid
from dataset import MultiLFArchDataset, NasBench301Dataset


def pretrain(train_loader, model, epoch, args, arch_network_type):
    """
    In the first step, train different low-fidelity experts on different types of low-fidelity information.
    """
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()

    model.train()
    for step, data in enumerate(train_loader):
        archs, _, low_fidelity_perfs = data
        archs = np.array(archs)
        n = len(archs)
        
        data_lst = []
        for low_fidelity in args.low_fidelity_type:
            lf_lst = np.array([perf_lst[low_fidelity] for perf_lst in low_fidelity_perfs])
            archs_1, archs_2, better_lst = compare_data(archs, lf_lst, lf_lst, args)
            data_lst.append((archs_1, archs_2, better_lst))
            n_diff_pairs = len(better_lst)
                
        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.mtl_update_compare(data_lst)
        objs.update(loss, n_diff_pairs)
        
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))

    return objs.avg


def train(train_loader, model, epoch, args, arch_network_type):
    """
    In the second step, finetune the entire predictor on the actual performance data.
    """
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    
    model.train()
    for step, data in enumerate(train_loader):
        archs, accs, _ = data
        archs = np.array(archs)
        accs = np.array(accs)
        n = len(archs)

        archs_1, archs_2, better_lst = compare_data(archs, accs, accs, args)
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        loss = model.update_compare(archs_1, archs_2, better_lst)
        objs.update(loss, n_diff_pairs)
        
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))
    return objs.avg


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--search-space", type = str, required = True, 
        choices = ["nasbench-201", "nb301", "nds"], help = "Search Space / Benchmark")
    parser.add_argument("--train-ratio", type = float, required = True, 
            help = "Proportions of training samples")
    parser.add_argument("--pretrain-ratio", type = float, default = 1., 
            help = "Proportions of pre-training samples")
    parser.add_argument("--train-pkl", type = str, required = True, help = "Training Datasets pickle")
    parser.add_argument("--valid-pkl", type = str, required = True, help = "Evaluate Datasets pickle")
    parser.add_argument("--train-dir", default = None, help = "Save train log / results into TRAIN_DIR")
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--load", default = None, help = "Load comparator from disk.")
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--report-freq", default = 200, type = int)
    parser.add_argument("--test-every", default = 10, type = int)
    parser.add_argument("--test-only", default = False, action = "store_true")
    args = parser.parse_args()

    setproctitle.setproctitle("python {} config: {}; train_dir: {}; cwd: {}"\
                              .format(__file__, args.cfg_file, args.train_dir, os.getcwd()))

    backup_cfg_file, device = prepare(args)

    # Initialize the search space
    if args.search_space == "nasbench-201":
        search_space = get_search_space("nasbench-201", load_nasbench = False)
    elif args.search_space == "nb301":
        search_space = get_search_space("nb301")
    elif args.search_space == "nds":
        search_space = get_search_space("germ")
    
    # Load the data
    logging.info("Load pkl cache from {} and {}".format(args.train_pkl, args.valid_pkl))
    with open(args.train_pkl, "rb") as rf:
        train_data = pickle.load(rf)
    with open(args.valid_pkl, "rb") as rf:
        valid_data = pickle.load(rf)

    # Load and update the configuration
    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f, Loader = yaml.FullLoader)

    cfg["train_ratio"] = args.train_ratio
    cfg["pretrain_ratio"] = args.pretrain_ratio

    logging.info("Config: %s", cfg)

    # Initialize the predictor
    arch_network_type = cfg.get("arch_network_type", "dynamic_ensemble_pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    if args.load is not None:
        logging.info("Load %s from %s", arch_network_type, args.load)
        model.load(args.load)
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # Construct the datasets
    logging.info("Pretrain dataset ratio: %.3f; Train dataset ratio: %.3f", args.pretrain_ratio, args.train_ratio)
    _num = len(train_data)

    real_data = train_data[:int(_num * args.train_ratio)]

    if args.search_space == "nb301":
        real_data = MultiLFArchDataset(real_data, args.low_fidelity_type, args.low_fidelity_normalize)
        train_data = train_data[:int(_num * args.pretrain_ratio)]
        train_data = MultiLFArchDataset(train_data, args.low_fidelity_type, args.low_fidelity_normalize)
        valid_data = NasBench301Dataset(valid_data, False, args.low_fidelity_type, args.low_fidelity_normalize)
    else:
        real_data = MultiLFArchDataset(real_data, args.low_fidelity_type, args.low_fidelity_normalize)
        train_data = train_data[:int(_num * args.pretrain_ratio)]
        train_data = MultiLFArchDataset(train_data, args.low_fidelity_type, args.low_fidelity_normalize)
        valid_data = MultiLFArchDataset(valid_data, args.low_fidelity_type, args.low_fidelity_normalize)

    logging.info("Number of architectures: pre-train: %d; train: %d; valid: %d", \
            len(train_data), len(real_data), len(valid_data))

    # Construct the data-loaders
    val_loader = DataLoader(
        valid_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    real_loader = DataLoader(
        real_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, 
        shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
 
    # init test
    if args.test_only:
        if args.search_space == "nb301":
            real_corr, patk = nb301_valid(val_loader, model, args)
            logging.info("Valid: kendall tau {:.4f}; patk {}".format(real_corr, patk))
        else:
            low_fidelity_corr, real_corr, patk = mtl_valid(val_loader, model, args)
            logging.info("Valid: kendall tau {}; real {:.4f}; patk {}".\
                    format("; ".join(["{}: {:.4f}".format(_type, low_fidelity_corr[_type]) 
                        for _type in low_fidelity_corr.keys()]), real_corr, patk))
        return

    for i_epoch in range(1, args.pretrain_epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = pretrain(train_loader, model, i_epoch, args, arch_network_type)
        logging.info("Pre-Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        
        if i_epoch == args.pretrain_epochs or i_epoch % args.test_every == 0:
            if args.search_space == "nb301":
                real_corr, patk = nb301_valid(val_loader, model, args)
                logging.info("Pre-Valid: Epoch {:3d}: kendall tau {:.4f}; patk {}".format(
                    i_epoch, real_corr, patk))
            else:
                low_fidelity_corr, real_corr, patk = mtl_valid(val_loader, model, args)
                logging.info("Pre-Valid: Epoch {:3d}: kendall tau {}; real {:.4f}; patk {}".\
                        format(i_epoch, 
                            "; ".join(["{}: {:.4f}".format(_type, low_fidelity_corr[_type]) 
                                for _type in low_fidelity_corr.keys()]),
                            real_corr, patk))
    
    save_path = os.path.join(args.train_dir, "pre_final.ckpt")
    model.save(save_path)
    logging.info("Save pre-train checkpoint to {}".format(save_path))

    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = train(real_loader, model, i_epoch, args, arch_network_type)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        
        if i_epoch == args.epochs or i_epoch % args.test_every == 0:
            if args.search_space == "nb301":
                real_corr, patk = nb301_valid(val_loader, model, args, save_path = args.train_dir)
                logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}; patk {}".format(
                    i_epoch, real_corr, patk))
            else:
                low_fidelity_corr, real_corr, patk = mtl_valid(
                        val_loader, model, args, save_path = args.train_dir)
                logging.info("Valid: Epoch {:3d}: kendall tau {}; real {:.4f}; patk {}".\
                        format(i_epoch, 
                            "; ".join(["{}: {:.4f}".format(_type, low_fidelity_corr[_type]) 
                                for _type in low_fidelity_corr.keys()]),
                            real_corr, patk))
                
    save_path = os.path.join(args.train_dir, "final.ckpt")
    model.save(save_path)
    logging.info("Save checkpoint to {}".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
