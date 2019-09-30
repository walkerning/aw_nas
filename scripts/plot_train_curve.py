# -*- coding: utf-8 -*-
"""
python plot_train_curves.py -s <file to save image> <log file1> <log file2> ...

The corresponding legend label will be set to the basename of the log file.
"""
#pylint: disable=invalid-name,wrong-import-position
from __future__ import print_function

import re
import os
import argparse
from textwrap import fill
from functools import reduce # pylint:disable=redefined-builtin

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import gridspec

def _convert_float(x):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return [_convert_float(item) for item in x]
    return float(x)

parser = argparse.ArgumentParser()
parser.add_argument("--save", "-s", required=True, help="save to path")
parser.add_argument("--type", "-t", choices=["rnn", "cnn", "pgd_robustness", "cnn_oneshot_search", "ftt"],
                    default="cnn", help="the type of logs")
parser.add_argument("--simplify", action="store_true", help="try to simplify the legend names using "
                    "the different part in the label")

args, fnames = parser.parse_known_args()

## --- patterns ---
log_y_names = ["perp"]
if args.type == "rnn": # rnn
    train_pattern = re.compile("train: perp ([0-9.]+); bpc [0-9.]+ ; "
                               "loss ([0-9.]+) ; loss with reg ([0-9.]+)")
    train_obj_names = ["perp", "loss", "loss with reg"]
    train_ylims = [(40, 200)] + [None] * 2

    valid_pattern = re.compile("valid: perp ([0-9.]+) ; bpc [0-9.]+ ; loss ([0-9.]+)")
    valid_obj_names = ["perp", "loss"]
    valid_ylims = [(40, 200), None]
elif args.type == "cnn": # cnn
    train_pattern = re.compile("train_acc ([0-9.]+) ; train_obj ([0-9.]+)")
    train_obj_names = ["acc", "loss"]
    train_ylims = [(80, 100), None]
    valid_pattern = re.compile("valid_acc ([0-9.]+) ; valid_obj ([0-9.]+)")
    valid_obj_names = ["acc", "loss"]
    valid_ylims = [(20, 80), None]
elif args.type == "cnn_oneshot_search":
    train_pattern = re.compile(r"\[evaluator update\][^\n]+loss: ([0-9.]+); acc: ([0-9.]+)")
    train_obj_names = ["loss", "acc"]
    train_ylims = [None, (0.1, 0.9)]
    valid_pattern = re.compile(r"TEST[^\n]+ loss: ([0-9.]+) \(mean ([0-9.]+)\); "
                               r"acc: ([0-9.]+) \(mean ([0-9.]+)\)")
    valid_obj_names = ["loss", "mean_loss", "acc", "mean_acc"]
    valid_ylims = [None, None, (0.1, 0.9), (0.1, 0.9)]
elif args.type == "pgd_robustness":
    train_pattern = re.compile("train_acc ([0-9.]+) ; train_obj ([0-9.]+)")
    train_obj_names = ["acc", "loss"]
    train_ylims = [(30, 100), None]
    valid_pattern = re.compile("valid_acc ([0-9.]+) ; valid_obj ([0-9.]+) ; "
                               "valid performances: acc_clean: ([0-9.]+); acc_adv: ([0-9.]+)")
    valid_obj_names = ["acc", "loss", "acc_clean", "acc_adv"]
    valid_ylims = [(30, 100), None, (0.3, 1.0), (0.3, 1.0)]
elif args.type == "ftt":
    train_pattern = re.compile("train_acc ([0-9.]+) ; train_obj ([0-9.]+)")
    train_obj_names = ["acc", "loss"]
    train_ylims = [(20, 100), None]
    valid_pattern = re.compile("valid_acc ([0-9.]+) ; valid_obj ([0-9.]+) ; "
                               "valid performances: acc_clean: ([0-9.]+); acc_fault: ([0-9.]+)")
    valid_obj_names = ["acc", "loss", "acc_clean", "acc_fault"]
    valid_ylims = [(20, 100), None, (0.2, 1.0), (0.2, 1.0)]

## --- parse logs ---
labels = []
file_train_objs_list = []
file_valid_objs_list = []
for fname in fnames:
    label = os.path.basename(fname)
    if label in {"train.log", "search.log"}:
        label = os.path.basename(os.path.dirname(fname))
    if "." in fname:
        label = label.rsplit(".", 1)[0]
    labels.append(label)
    content = open(fname, "r").read().strip()
    train_objs = list(zip(*_convert_float(train_pattern.findall(content))))
    valid_objs = list(zip(*_convert_float(valid_pattern.findall(content))))
    assert len(train_objs) == len(train_obj_names) and len(train_objs[0]) > 0, \
        "maybe `--type` is not correctly set?"
    if not valid_objs:
        valid_objs = [tuple() for _ in range(len(valid_obj_names))]
    # assert len(valid_objs) == len(valid_obj_names) and len(valid_objs[0]) > 0, \
    #     "maybe `--type` is not correctly set?"
    file_train_objs_list.append(train_objs)
    file_valid_objs_list.append(valid_objs)

if args.simplify:
    bans = [set(l.split("_")) for l in labels]
    _inds = {n: i for i, n in enumerate(labels[0].split("_"))}
    common_ban = reduce(lambda s1, s2: s1.intersection(s2), bans, bans[0])
    common_label = "_".join(sorted(common_ban, key=lambda b: _inds[b]))
    labels = [" ".join(sorted(list(s.difference(common_ban)))) for s in bans]
    labels = [fill(l, 20) if l else "--" for l in labels]
else:
    common_label = ""

## --- plot ---
num_train_objs = len(train_obj_names)
num_valid_objs = len(valid_obj_names)
num_cols = max(num_train_objs, num_valid_objs)
fig = plt.figure(figsize=(2*(num_cols+1), 5))
gs = gridspec.GridSpec(nrows=2, ncols=(num_cols+1), width_ratios=[3]*num_cols + [2])

# valid
for i in range(num_valid_objs):
    ax = fig.add_subplot(gs[0, i])
    for label, objs in zip(labels, file_valid_objs_list):
        ax.plot(objs[i], label=label)
    if valid_ylims[i] is not None:
        ax.set_ylim(valid_ylims[i])
    if valid_obj_names[i] in log_y_names:
        ax.set_yscale("log")
    ax.set_title("valid " + valid_obj_names[i])
# train
for i in range(num_train_objs):
    ax = fig.add_subplot(gs[1, i])
    handles = []
    for label, objs in zip(labels, file_train_objs_list):
        handles.append(ax.plot(objs[i], label=label)[0])
    if train_ylims[i] is not None:
        ax.set_ylim(train_ylims[i])
    if train_obj_names[i] in log_y_names:
        ax.set_yscale("log")
    ax.set_title("train " + train_obj_names[i])

# use a separate subplot to show the legends
ax = fig.add_subplot(gs[:, num_cols:], frameon=False) # no frame (remove the four spines)
plt.gca().axes.get_xaxis().set_visible(False) # no ticks
plt.gca().axes.get_yaxis().set_visible(False)
plt.legend(tuple(handles), labels, loc="center") # center the legends info in the subplot

plt.suptitle(common_label) # set the common label as the suptitle

# tight_layout do not take into account the suptitle, so specify the rect here
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(args.save)
