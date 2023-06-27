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

from common_utils import NasBenchDataset, test_xk


def train(train_loader: DataLoader, model: ArchNetwork, epoch: int, args: argparse.Namespace) -> float:
    """
    Train the anytime predictor for an epoch.

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
        # h_accs = h_accs[:, h_accs.shape[1] // 2]
        f_accs = np.array(f_accs)
        zs_as_l = np.array(zs_as_l) # arch-level
        zs_embs = np.array(zs_embs) # param-level

        n = len(archs)

        archs = (archs, zs_as_l, zs_embs)

        loss = model.update_predict(archs, [h_accs, f_accs])
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
    Test the anytime predictor on the validation dataset.

    Args:
        val_loader (DataLoader): Validation dataset dataloader.
        model (ArchNetwork): The predictor to be tested.
        args (argparse.Namespace): Arguments.
    """
    model.eval()

    all_scores = []
    true_accs = []
    all_h_scores = []
    true_h_accs = []

    if args.save_emb:
        logging.info("embedding dimension: %d", model.arch_embedder.out_dim)
        all_embs = np.zeros((0, model.arch_embedder.out_dim), dtype = np.float32)

    with torch.no_grad():
        for step, (archs, accs, h_accs, zs_as_l, zs_embs) in enumerate(val_loader):
            archs = (archs, zs_as_l, zs_embs)

            if args.save_emb:
                all_embs = np.concatenate(
                    (all_embs, model.arch_embedder(archs).detach().cpu().numpy()), axis = 0)

            h_scores, scores = model.predict(archs, anytime = True)

            scores = list(scores.cpu().data.numpy())
            h_scores = list(h_scores.cpu().data.numpy())

            all_scores += scores
            all_h_scores += h_scores
            true_accs += list(accs)
            true_h_accs += list(h_accs)

    if args.save_predict is not None:
        with open(args.save_predict, "wb") as wf:
            pickle.dump((true_accs, all_scores), wf)

    if args.save_emb is not None:
        with open(args.save_emb, "wb") as wf:
            pickle.dump((archs, true_accs, all_embs, all_scores), wf)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    regression_loss = ((np.array(all_scores) - np.array(true_accs)) ** 2).mean()
    patk = test_xk(true_accs, all_scores)

    h_corr = stats.kendalltau(true_h_accs, all_h_scores).correlation
    h_regression_loss = ((np.array(all_h_scores) - np.array(true_h_accs)) ** 2).mean()
    h_patk = test_xk(true_h_accs, all_h_scores)

    return [corr, h_corr], [regression_loss, h_regression_loss], [patk, h_patk]


def main(argv):
    parser = argparse.ArgumentParser(prog = "train_nasbench201_anytime.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--report_freq", default = 200, type = int)
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--train-dir", default = None, help = "Save train log/results into TRAIN_DIR")
    parser.add_argument("--train-ratio", default = None, type = float)
    parser.add_argument("--save-every", default = 10, type = int)
    parser.add_argument("--load", default = None, help = "Load comparator from disk.")
    parser.add_argument("--test-only", default = False, action = "store_true")
    parser.add_argument("--test-every", default = 50, type = int)
    parser.add_argument("--save-predict", default = None, help = "Save the predict scores")
    parser.add_argument("--save-emb", default = None, help = "Save the embeddings")
    parser.add_argument("--train-pkl", type = str, required = True, help = "Training Datasets pickle, containing zeroshot embedding")
    parser.add_argument("--valid-pkl", type = str, required = True, help = "Evaluate Datasets pickle, containing zeroshot embedding")
    parser.add_argument("--no-init-test", default = False, action = "store_true", help = "Do not run initial KD test")
    args = parser.parse_args(argv)

    setproctitle.setproctitle("python train_nasbench201_anytime.py config: {}; train_dir: {}; cwd: {}"\
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

    arch_network_type = cfg.get("arch_network_type", "any_time_pointwise_comparator")
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
        corr, valid_loss, patk = valid(val_loader, model, args)
        logging.info("Valid: final loss {:.5f}; final kendall tau {:.5f}; patk {}".format(valid_loss[0], corr[0], patk[0]))
        logging.info("Valid: half valid loss {:.5f}; half kendall tau {:.5f}; patk {}".format(valid_loss[1], corr[1], patk[1]))

    if args.test_only:
        return

    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = train(train_loader, model, i_epoch, args)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

        if i_epoch % args.test_every == 0:
            corr, valid_loss, patk = valid(val_loader, model, args)

            logging.info("Valid: Epoch {:3d}: final valid loss {:.5f}; final kendall tau {:.5f}; patk {}".format(i_epoch, valid_loss[0], corr[0], patk[0]))
            logging.info("Valid: Epoch {:3d}: half valid loss {:.5f}; half kendall tau {:.5f}; patk {}".format(i_epoch, valid_loss[1], corr[1], patk[1]))

        if i_epoch % args.save_every == 0 or i_epoch == args.epochs:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
