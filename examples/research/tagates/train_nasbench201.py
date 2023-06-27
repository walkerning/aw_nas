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

import numpy as np
import setproctitle
from scipy.stats import stats
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork

from common_utils import test_xk, NasBenchDataset


def train(train_loader: DataLoader, model: ArchNetwork, epoch: int, args: argparse.Namespace) -> float:
    """
    Train the predictor for an epoch.

    Args:
        train_loader (DataLoader): The dataloader.
        model (ArchNetwork): The predictor to be trained.
        epoch (int): Current epoch.
        args (argparse.Namespace): Arguments.

    Returns:
        float: The loss on the training dataset.
    """
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()

    model.train()

    for step, (archs, f_accs, h_accs, zs_as_l, zs_embs) in enumerate(train_loader):
        archs = np.array(archs)
        h_accs = np.array(h_accs)
        f_accs = np.array(f_accs)
        zs_as_l = np.array(zs_as_l) # arch-level
        zs_embs = np.array(zs_embs) # param-level

        n = len(archs)

        accs = f_accs

        # If `args.compare == True`, train the predictor with ranking loss.
        # Else, train the predictor with regression loss.
        if args.compare:
            n_max_pairs = int(args.max_compare_ratio * n)
            acc_diff = np.array(accs)[:, None] - np.array(accs)
            acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
            ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
            ex_thresh_num = len(ex_thresh_inds[0])
            if ex_thresh_num > n_max_pairs:
                if args.choose_pair_criterion == "diff":
                    keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
                elif args.choose_pair_criterion == "random":
                    keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace = False)
                ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
            archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]

            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))

            zs_embs_1, zs_embs_2 = zs_embs[ex_thresh_inds[1]], zs_embs[ex_thresh_inds[0]]
            zs_as_l_1, zs_as_l_2 = zs_as_l[ex_thresh_inds[1]], zs_as_l[ex_thresh_inds[0]]
            archs_1 = (archs_1, zs_as_l_1, zs_embs_1)
            archs_2 = (archs_2, zs_as_l_2, zs_embs_2)

            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)

        else:
            archs = (archs, zs_as_l, zs_embs)
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


def valid(val_loader: DataLoader, model: ArchNetwork, args: argparse.Namespace):
    """
    Test the predictor on the validation dataset.

    Args:
        val_loader (DataLoader): Validation dataset dataloader.
        model (ArchNetwork): The predictor to be tested.
        args (argparse.Namespace): Arguments.
    """
    model.eval()

    all_scores = []
    true_accs = []

    if args.save_emb:
        logging.info("embedding dimension: %d", model.arch_embedder.out_dim)
        all_embs = np.zeros((0, model.arch_embedder.out_dim), dtype = np.float32)

    for step, (archs, accs, _, zs_as_l, zs_embs) in enumerate(val_loader):
        archs = (archs, zs_as_l, zs_embs)

        if args.save_emb:
            all_embs = np.concatenate(
                (all_embs, model.arch_embedder(archs).detach().cpu().numpy()), axis = 0)

        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs)

    if args.save_predict is not None:
        with open(args.save_predict, "wb") as wf:
            pickle.dump((true_accs, all_scores), wf)

    if args.save_emb is not None:
        with open(args.save_emb, "wb") as wf:
            pickle.dump((archs, true_accs, all_embs, all_scores), wf)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    regression_loss = ((np.array(all_scores) - np.array(true_accs)) ** 2).mean()

    patk = test_xk(true_accs, all_scores)

    if args.early_stop_valid is not None:

        valid_corr = []
        test_corr = []

        for num_v in args.early_stop_valid:
            valid_corr.append(stats.kendalltau(true_accs[:num_v], all_scores[:num_v]).correlation)
            test_corr.append(stats.kendalltau(true_accs[num_v:], all_scores[num_v:]).correlation)

        return (valid_corr, test_corr, corr), regression_loss, patk

    return corr, regression_loss, patk


def main(argv):
    parser = argparse.ArgumentParser(prog = "train_nasbench201.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--report_freq", default = 200, type = int)
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--train-dir", default = None, help = "Save train log/results into TRAIN_DIR")
    parser.add_argument("--train-ratio", default = None, type = float)
    parser.add_argument("--save-every", default = 10, type = int)
    parser.add_argument("--test-only", default = False, action = "store_true")
    parser.add_argument("--test-every", type = int, default = 1)
    parser.add_argument("--load", default = None, help = "Load comparator from disk.")
    parser.add_argument("--save-predict", default = None, help = "Save the predict scores")
    parser.add_argument("--save-emb", default = None, help = "Save the embeddings")
    parser.add_argument("--train-pkl", type = str, required = True, help = "Training Datasets pickle, containing zeroshot embedding")
    parser.add_argument("--valid-pkl", type = str, required = True, help = "Evaluate Datasets pickle, containing zeroshot embedding")
    parser.add_argument("--no-init-test", default = False, action = "store_true", help = "Do not run initial KD test")
    parser.add_argument("--early-stop-valid", default = [20, 40], help = "A interger indicating how many architectures are used for early-stop")
    args = parser.parse_args(argv)

    setproctitle.setproctitle("python train_nasbench201.py config: {}; train_dir: {}; cwd: {}"\
                              .format(args.cfg_file, args.train_dir, os.getcwd()))

    # log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream = sys.stdout, level = logging.INFO,
                        format = log_format, datefmt = "%m/%d %I:%M:%S %p")

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

    search_space = get_search_space("nasbench-201", load_nasbench = False)

    logging.info("Load pkl cache from {} and {}".format(args.train_pkl, args.valid_pkl))
    with open(args.train_pkl, "rb") as rf:
        train_data = pickle.load(rf)
    with open(args.valid_pkl, "rb") as rf:
        valid_data = pickle.load(rf)

    with open(backup_cfg_file, "r") as cfg_rf:
        cfg = yaml.load(cfg_rf)

    cfg["train_ratio"] = args.train_ratio

    logging.info("Config: %s", cfg)

    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    if args.load is not None:
        logging.info("Load %s from %s", arch_network_type, args.load)
        model.load(args.load)
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # d[0]: arch representation
    # d[1]: param-level zeroshot (as p, symmetry breaking)
    # d[2]: final accuracies
    # d[4]: arch-level zeroshot (as l)
    # d[-1]: accuracies of different epochs
    train_data = [[d[0], d[2], d[-1], d[4], d[1]] for d in train_data]
    valid_data = [[d[0], d[2], d[-1], d[1], d[1]] for d in valid_data]

    # init nasbench data loaders
    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        _num = len(train_data)
        train_data = train_data[:int(_num * args.train_ratio)]
        logging.info("Train dataset ratio: %.3f", args.train_ratio)
    num_train_archs = len(train_data)
    logging.info("Number of architectures: train: %d; valid: %d", num_train_archs, len(valid_data))

    train_data = NasBenchDataset(train_data)
    valid_data = NasBenchDataset(valid_data)

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = args.num_workers,
        collate_fn = lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = args.num_workers,
        collate_fn = lambda items: list(zip(*items)))

    # init test
    if not args.no_init_test or args.test_only:

        if args.early_stop_valid:
            (valid_corr, test_corr, corr), valid_loss, patk = valid(val_loader, model, args)
            logging.info("INIT: valid loss {:.4g}; kendall tau {} {:.4f}; patk {}".format(
                valid_loss,
                " ".join(["({:d} - {:.4f} {:.4f})".format(num_v, v_corr, t_corr)
                    for num_v, v_corr, t_corr in zip(
                        args.early_stop_valid, valid_corr, test_corr)]), corr, patk
                )
            )
        else:
            corr, valid_loss, patk = valid(val_loader, model, args)
            logging.info("INIT: valid loss {:.4g}; kendall tau {:.4f}; patk {}".format(valid_loss, corr, patk))

    if args.test_only:
        return

    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = train(train_loader, model, i_epoch, args)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

        if i_epoch % args.test_every == 0:
            if args.early_stop_valid:
                (valid_corr, test_corr, corr), valid_loss, patk = valid(val_loader, model, args)
                logging.info("Valid (EARLY STOP): Epoch {:3d}: valid loss {:.4g}; kendall tau {} {:.4f}; patk {}".format(
                    i_epoch, valid_loss,
                    " ".join(["({:d} - {:.4f} {:.4f})".format(num_v, v_corr, t_corr)
                          for num_v, v_corr, t_corr in zip(
                                  args.early_stop_valid, valid_corr, test_corr)]),
                    corr, patk
                ))
            else:
                corr, valid_loss, patk = valid(val_loader, model, args)
                logging.info("Valid: Epoch {:3d}: valid loss {:.4g}; kendall tau {:.4f}; patk {}".format(i_epoch, valid_loss, corr, patk))

        if i_epoch % args.save_every == 0:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
