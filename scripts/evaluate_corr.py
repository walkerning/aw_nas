"""
For a trained NAS system, test the correlation of performance in `deriving` and
`train_surrogate` stages.
"""

from __future__ import print_function

import re
import argparse

import yaml
from scipy.stats.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--derive-file", required=True, type=str,
                    help="The yaml file contains a list of derived architectures.")
parser.add_argument("--train-dir", required=True,
                    help="Assume the training dir is `TRAIN_DIR/<index>` for the <index>-th "
                    "(0-based) arch in DERIVE_FILE.")
parser.add_argument("--derive-perf-in-file", action="store_true", type=bool,
                    help="")
parser.add_argument("--cfg", default=None, type=str,
                    help="The search config. Required if DERIVE_PERF_IN_FILE is false, "
                    "ignored otherwise.")
parser.add_argument("--load", default=None, type=str,
                    help="Load the components from this dir, and test the derived performance"
                    " of the archs. Required if DERIVE_PERF_IN_FILE is false, ignored otherwise.")
parser.add_argument("--gpu", default=None, type=str,
                    help="The device to be used for test the derived performances. "
                    "Required if DERIVE_PERF_IN_FILE is false, ignored otherwise.")

args = parser.parse_args()
derive_file = args.derive_file

if args.derive_perf_in_file:
    derive_perfs = [float(x) for x in re.findall("Reward ([0-9.]+)", open(derive_file, "r").read())]
else:
    # TODO: trainer要加一个get_perf, 其实就是evaluator的get_perf... 现在的dervie肯定要用这个包装...
    pass
