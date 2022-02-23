# -*- coding: utf-8 -*-
# pylint: disable-all

"""
Train architecture-performance predictor on NAS-Bench-201 / NAS-Bench-301 / NDS-ResNet/ResNeXT-A.

First, pretrain the predictor on a single type of low-fidelity information.
Second, finetune the predictor on the actual performance data.

Copyright (c) 2022 Junbo Zhao, Xuefei Ning
"""

import os
import sys
import logging
import argparse
import pickle
import yaml

import numpy as np
import setproctitle
from torch.utils.data import DataLoader

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork

from utils import compare_data, prepare, valid, nb301_valid
from dataset import ArchDataset, NasBench301Dataset


def train(real_loader, model, epoch, args, arch_network_type, use_low_fidelity: bool = False):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, real_accs, low_fidelity) in enumerate(real_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if args.compare:
            archs_1, archs_2, better_lst = compare_data(archs, accs, accs, args)
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            n = len(archs)
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(real_loader), objs.avg,
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
    parser.add_argument("--no-pretrain", default = False, action = "store_true", 
            help = "Directly train the predictor without pretraining")
    parser.add_argument("--low-fidelity-type", type = str, default = "one_shot", 
            help = "Applied low-fidelity information type")
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
    cfg["low_fidelity_type"] = args.low_fidelity_type

    logging.info("Config: %s", cfg)

    # Initialize the predictor
    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
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
        real_data = NasBench301Dataset(real_data, True, args.low_fidelity_type, args.low_fidelity_normalize)
        train_data = train_data[:int(_num * args.pretrain_ratio)]
        train_data = NasBench301Dataset(train_data, True, args.low_fidelity_type, args.low_fidelity_normalize)
        valid_data = NasBench301Dataset(valid_data, False, args.low_fidelity_type, args.low_fidelity_normalize)
    else:
        real_data = ArchDataset(real_data, args.low_fidelity_type, args.low_fidelity_normalize)
        train_data = train_data[:int(_num * args.pretrain_ratio)]
        train_data = ArchDataset(train_data, args.low_fidelity_type, args.low_fidelity_normalize)
        valid_data = ArchDataset(valid_data, args.low_fidelity_type, args.low_fidelity_normalize)

    logging.info("Number of architectures: pre-train: %d; train: %d; valid: %d", \
            len(train_data), len(real_data), len(valid_data))

    # Initialize the data-loaders
    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    real_loader = DataLoader(
        real_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))

    # init test
    if args.test_only:
        if args.search_space == "nb301":
            real_corr, patk = nb301_valid(val_loader, model, args)
            logging.info("Valid: kendall tau {:.4f}; patk {}".format(real_corr, patk))
        else:
            low_fidelity_corr, real_corr, patk = valid(val_loader, model, args)
            logging.info("Valid: kendall tau {} {:.4f}; real {:.4f}; patk {}".format(
                args.low_fidelity_type, low_fidelity_corr, real_corr, patk))
        return

    # Step 1: pre-train the predictor with low_fidelity objective
    if not args.no_pretrain:
        for i_epoch in range(1, args.epochs + 1):
            model.on_epoch_start(i_epoch)
            avg_loss = train(train_loader, model, i_epoch, args, arch_network_type, True)
            logging.info("Pre-train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

            if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
                if args.search_space == "nb301":
                    real_corr, patk = nb301_valid(val_loader, model, args)
                    logging.info("Pre-valid: Epoch {:3d}: kendall tau {:.4f}; patk {}".\
                            format(i_epoch, real_corr, patk))
                else:
                    low_fidelity_corr, real_corr, patk = valid(val_loader, model, args)
                    logging.info("Pre-valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
                        format(i_epoch, args.low_fidelity_type, low_fidelity_corr, real_corr, patk))
    
        save_path = os.path.join(args.train_dir, "pre_train.ckpt")
        model.save(save_path)
        logging.info("Finish pre-training. Save checkpoint to {}".format(save_path))

    # Step 2: Finetune the predictor on real data
    model.reinit_optimizer()
    model.reinit_scheduler()

    for i_epoch in range(1, args.finetune_epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = train(real_loader, model, i_epoch, args, arch_network_type)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        
        if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
            if args.search_space == "nb301":
                real_corr, patk = nb301_valid(val_loader, model, args, save_path = args.train_dir)
                logging.info("Pre-valid: Epoch {:3d}: kendall tau {:.4f}; patk {}".\
                        format(i_epoch, real_corr, patk))
            else:
                low_fidelity_corr, real_corr, patk = valid(val_loader, model, args, save_path = args.train_dir)
                logging.info("Valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
                        format(i_epoch, args.low_fidelity_type, low_fidelity_corr, real_corr, patk))
                
    save_path = os.path.join(args.train_dir, "final.ckpt")
    model.save(save_path)
    logging.info("Save checkpoint to {}".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
