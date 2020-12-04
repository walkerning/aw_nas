#pylint: disable-all
"""
A predictor diagnoistic script, to diagnose whether the predictor-based search is working as expected.
First, are the predicted scores increasing? If not, the inner search method might be too weak.
Second, if the predicted scores are increasing, but the actual evaluation rewards of these topologies are not.
Check the correlation between the rewards and predicted scores!
"""

import aw_nas
import os
import yaml
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork
import torch
import pickle
import numpy as np
from scipy.stats import stats

def spearmanr(surrogate_accs, accs):
    return stats.spearmanr(surrogate_accs, accs).correlation

def kendalltau(surrogate_accs, accs):
    return stats.kendalltau(surrogate_accs, accs).correlation

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1]
    def print_with_fmt(container, fmt):
        if isinstance(container, (list, tuple)):
            return ", ".join([print_with_fmt(i, fmt) for i in container])
        return fmt.format(container)

    cfg_file = os.path.join(base_dir, "config.yaml")
    with open(cfg_file, "r") as rf:
        cfg = yaml.load(rf)
    ss = get_search_space(cfg["search_space_type"], **cfg["search_space_cfg"])
    pred = ArchNetwork.get_class_(cfg["controller_cfg"]["arch_network_type"])(ss, **cfg["controller_cfg"]["arch_network_cfg"])

    def calc_correlation(name_list, rollouts, corr_func=kendalltau):
        perfs = [[r.perf[name] for r in rollouts] for name in name_list]
        num_perf = len(perfs)
        corr_mat = np.zeros((num_perf, num_perf))
        for i in range(num_perf):
            for j in range(num_perf):
                corr_mat[i][j] = corr_func(perfs[i], perfs[j])
        return corr_mat

    def print_corr_mat(name_list, corr_mat, print_name):
        print(print_name)
        sz = len(name_list)
        name_list = [name[:12] for name in name_list]
        fmt_str = " ".join(["{:12}"] * (sz + 1))
        print(fmt_str.format("perf name", *name_list))
        fmt_str = "{:12} " + " ".join(["{:<12.4f}"] * (sz))
        for name, row in zip(name_list, corr_mat):
            print(fmt_str.format(name, *row))
        print("-----")


    def get_statistics(name, stage_rollouts):
        perfs = [r.perf[name] for r in stage_rollouts]
        return {"median": np.median(perfs), "mean": np.mean(perfs), "std": np.std(perfs), "max": np.max(perfs), "min": np.min(perfs)}
    def print_stat(name, from_stage=0):
        print(name)
        print("\n".join(["{i_stage}: {mean:.4f}+-{std:.4f} [{min:.4f}, ({median:.4f}), {max:.4f}]".format(i_stage=i_stage+from_stage+1, **get_statistics(name, stage_rollouts)) for i_stage, stage_rollouts in enumerate(rollouts[from_stage:])]))

    def try_cvt_int(name):
        try:
            return int(name)
        except:
            return None

    ckpt_dirs = sorted([try_cvt_int(name) for name in os.listdir(base_dir) if try_cvt_int(name) is not None])
    final_ckpt_dir = str(ckpt_dirs[-1])
    rollouts = pickle.load(open(os.path.join(base_dir, final_ckpt_dir, "controller_rollouts.pkl"), "rb"))

    use_pred = sys.argv[2] if len(sys.argv) >= 3 else final_ckpt_dir
    pred.load(os.path.join(base_dir, use_pred, "controller_predictor"))


    print_stat("reward")
    print_stat("acc_clean")
    print_stat("acc_adv")

    # print_stat("predicted_score", from_stage=4)
    # let's call predictor on the rollouts
    _pad_archs = ss.pad_archs if hasattr(ss, "pad_archs") else lambda arch: arch

    def _predict_rollouts(model, rollouts):
        num_r = len(rollouts)
        cur_ind = 0
        while cur_ind < num_r:
            end_ind = min(num_r, cur_ind + 64)
            padded_archs = _pad_archs([r.arch for r in rollouts[cur_ind:end_ind]])
            scores = model.predict(padded_archs).cpu().data.numpy()
            for r, score in zip(rollouts[cur_ind:end_ind], scores):
                r.set_perf(score, name="predicted_score")
            cur_ind = end_ind
        return rollouts


    [_predict_rollouts(pred, stage_rollouts) for stage_rollouts in rollouts]
    print_stat("predicted_score")
    # name_list = ["predicted_score", "reward", "acc_clean", "acc_adv"]
    # for i_stage, stage_rollouts in enumerate(rollouts):
    #     corr_mat = calc_correlation(name_list, stage_rollouts)
    #     print_corr_mat(name_list, corr_mat, print_name="Stage {}".format(i_stage))
    def reward_score_corrs(rollouts):
        per_stage = [calc_correlation(["reward", "predicted_score"], stage_rollouts)[0][1] for stage_rollouts in rollouts]
        cum_stage = [calc_correlation(["reward", "predicted_score"], sum(rollouts[:i+1], []))[0][1] for i in range(len(rollouts))]
        return per_stage, cum_stage

    for pred_epoch in ckpt_dirs:
        path = os.path.join(base_dir, str(pred_epoch), "controller_predictor")
        if not os.path.exists(path):
            continue

        pred.load(path)
        [_predict_rollouts(pred, stage_rollouts) for stage_rollouts in rollouts]
        print("Predictor epoch {}, reward-score correlation:".format(pred_epoch))
        per_stage, cum_stage = reward_score_corrs(rollouts)
        print("per stage:", print_with_fmt(per_stage, "{:.4f}"))
        print("cum stage:", print_with_fmt(cum_stage, "{:.4f}"))
