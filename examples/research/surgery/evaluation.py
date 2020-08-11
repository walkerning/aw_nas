import os
import argparse

import yaml
import numpy as np
from scipy.stats import stats, spearmanr

from aw_nas.btcs.nasbench_201 import NasBench201SearchSpace
from aw_nas import utils

parser = argparse.ArgumentParser()
parser.add_argument('-t','--type', default='deiso', choices=['deiso', 'iso', 'iso2deiso'])
parser.add_argument('derive_result', type=str)
args = parser.parse_args()

# Calculate the BR@K
def minn_at_k(true_scores, predict_scores, ks=[0.01, 0.05, 0.10, 0.20]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = [
        (k, int(np.min(true_ranks[predict_best_inds[:int(k * len(true_scores))]])) + 1)
        for k in ks]
    return minn_at_ks

# Calculate the P@K and Kendall-Tau
def test_xk(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        k = int(num_archs * ratio)
        if k < 1:
            continue
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        # [#samples, #samples/#total_samples, models in top-K, P@K%, Kendall-Tau]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks

# Parse the derive results
def parse_derive(filename):
    f = open(filename)
    lines = f.readlines()
    arch_dict = {}
    reward = None
    arch = None
    for i in range(len(lines)):
        if i % 3 == 0:
            reward = float(lines[i][lines[i].find("Reward")+7:lines[i].find(")")])
        elif i % 3 == 1:
            arch = lines[i].strip()[3:-1]
        else:
            arch_dict[arch] = reward
    f.close()
    return arch_dict

# Read the isomorphic ground truth
def get_iso_dict():
    with open(os.path.join(utils.get_awnas_dir("AWNAS_DATA", "nasbench-201"), "iso_dict.yaml")) as fnb:
        query_dict = yaml.load(fnb.read())
    return query_dict

# Read the De-isomorphic ground truth
def get_deiso_dict():
    ss = NasBench201SearchSpace(17, 4)
    query_dict = {}
    iso_group = []
    with open(os.path.join(utils.get_awnas_dir("AWNAS_DATA", "nasbench-201"), "deiso_dict.txt")) as iso_f:
        lines = iso_f.readlines()[1:]
        for line in lines:
            line_split = line.strip().split(" ")
            acc_list = []
            name_list = []
            for i in range(len(line_split) // 7):
                split_ = line_split[i*7:(i+1)*7]
                arch_mat = np.zeros([ss.num_vertices,ss.num_vertices]).astype(np.int32)
                ele = [float(i) for i in split_]
                arch_mat[np.tril_indices(ss.num_vertices, k=-1)] = ele[:int(ss.num_vertices * (ss.num_vertices - 1) / 2)]
                acc_list.append(ele[-1])
                name_list.append(ss.matrix2str(arch_mat))
            for name_ in name_list:
                query_dict[name_] = np.mean(acc_list)
            iso_group.append(name_list)
    return query_dict, iso_group

# Print the evaluation results 
def print_info(final_arch_dict, query_dict):
    search_list = []
    query_list = []
    for arch in final_arch_dict.keys():
        accs = query_dict[arch]
        query_list.append(accs)
        search_list.append(final_arch_dict[arch])
    print("linear corr: {}".format(np.corrcoef(np.array(search_list), np.array(query_list))))
    print("spearman corr: {}".format(spearmanr(np.array(search_list), np.array(query_list))))
    print("BR@K: {}".format(minn_at_k(np.array(search_list), np.array(query_list))))
    print("P@K/Kendall-Tau: {}".format(test_xk(np.array(search_list), np.array(query_list))))

# Post De-isomorphism
def iso_to_deiso(iso_dict, iso_group):
    with open(os.path.join(utils.get_awnas_dir("AWNAS", "nasbench-201"), "non-isom.txt")) as fiso:
        lines = fiso.readlines()[:6466]
        lines = [line.strip() for line in lines]
    deiso_dict = {}
    for group_ in iso_group:
        acc_list = []
        name_ = group_[0]
        for ele_ in group_:
            acc_list.append(iso_dict[ele_])
            if ele_ in lines:
                name_ = ele_
        deiso_dict[name_] = np.mean(acc_list)
    return deiso_dict

def main():
    final_arch_dict = parse_derive(args.derive_result)
    if args.type == "iso2deiso":
        query_dict, iso_group = get_deiso_dict()
        print_info(iso_to_deiso(final_arch_dict, iso_group), query_dict)
    elif args.type == "deiso":
        query_dict, iso_group = get_deiso_dict()
        print_info(final_arch_dict, query_dict)
    else:
        iso_dict = get_iso_dict()
        print_info(final_arch_dict, iso_dict)

if __name__ == "__main__":
    main()

