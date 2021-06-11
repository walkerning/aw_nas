#pylint: disable-all
"""
E.g., nb201: python evaluation.py -t deiso /home/eva_share_users/foxfi/surgery/nb201/results_derive/deiso_plateaulr_seed202020/*.yaml --save /home/eva_share_users/foxfi/surgery/nb201/results_derive/deiso_plateaulr_seed202020_stat.pkl
"""
import os
import argparse

import yaml
import numpy as np
from scipy.stats import stats, spearmanr
import pickle

from collections import namedtuple
from aw_nas.btcs.nasbench_101 import NasBench101SearchSpace

try:
    from aw_nas.btcs.nasbench_201 import NasBench201SearchSpace
except Exception as e:
    print(e)
    print("Should install nb201 requirements!")

from aw_nas import utils
from aw_nas.common import genotype_from_str
from aw_nas.utils.common_utils import _parse_derive_file
try:
    from aw_nas.btcs.nasbench_301 import NB301SearchSpace
    import nasbench301 as nb
except Exception as e:
    print(e)
    print("Should install nb301 requirements")

nb201_dir = os.path.join(utils.get_awnas_dir("AWNAS_DATA", "data"), "nasbench-201")

parser = argparse.ArgumentParser()
parser.add_argument("-t","--type", default="deiso", choices=["deiso", "iso", "iso2deiso",
"iso2deiso_withoutpost", "nb301", "nb101"])
parser.add_argument("--model", default=None)
parser.add_argument("--oneshot", action="store_true")
parser.add_argument("--save", help="Pickle all evaluation result to this file.", default=None)
args, derive_files = parser.parse_known_args()



# Calculate the BR@K, WR@K
def minmax_n_at_k(predict_scores, true_scores, ks=[0.001, 0.005, 0.01, 0.05, 0.10, 0.20]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:int(k * len(true_scores))]]
        if len(ranks) < 1:
            continue
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append((k, minn, float(minn) / num_archs, maxn, float(maxn) / num_archs))
    return minn_at_ks


# Calculate the P@topK, P@bottomK, and Kendall-Tau in predicted topK/bottomK
def p_at_tb_k(predict_scores, true_scores, ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]):
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs-k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(predict_scores[top_inds], true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(predict_scores[bottom_inds], true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append((ratio, k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks

# Parse the derive results
def parse_derive(filename):
    if args.type == "nb301":
        with open(filename, "rb") as fr:
            rollouts = pickle.load(fr)
        arch_dict = {str(r.genotype): r.perf for r in rollouts}
        return arch_dict
    if args.type == "nb101":
        with open(filename, "rb") as fr:
            rollouts = pickle.load(fr)
        return {r.genotype: r.perf for r in rollouts}
    # nb201
    with open(filename, "r") as r_f:
        arch_dict = _parse_derive_file(r_f)
    return arch_dict


class Model(object):
    def __init__(self, path, tabular_path):
        self.path = path
        self.model = nb.load_ensemble(path)
        self.search_space = NB301SearchSpace()
        self.genotype_type = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")
        self.tabular = {}

        self.tabular_path = tabular_path or os.path.join(path, "tabular.pkl")
        if os.path.exists(self.tabular_path):
            self.load(self.tabular_path)


    def load(self, path=None):
        path = path or self.tabular_path
        with open(path, "rb") as fr:
            self.tabular = pickle.load(fr)
        return self

    def save(self, path=None):
        path = path or self.tabular_path
        with open(path, "wb") as fw:
            pickle.dump(self.tabular, fw)

    def __getitem__(self, key):
        if key in self.tabular:
            return self.tabular[key]
        geno = genotype_from_str(key, self.search_space)
        res = self.model.predict(config=self.genotype_type(
            normal=[g[:2] for g in geno.normal_0], normal_concat=[2,3,4,5],
            reduce=[g[:2] for g in geno.reduce_1], reduce_concat=[2,3,4,5]),
                representation="genotype", with_noise=False)
        self.tabular[key] = res
        return res


def get_nb301_iso_dict(model_dir, tabular_path=None):
    model = Model(model_dir, tabular_path)
    return model


class NB101Model(object):
    def __init__(self, model):
        self.model = model

    def __getitem__(self, key):
        key.matrix = key.matrix.astype(int)
        return self.model.query(key)["test_accuracy"]

def get_nb101_dict(model):
    return NB101Model(model)


# Read the isomorphic ground truth
def get_iso_dict():
    with open(os.path.join(nb201_dir, "iso_dict.yaml")) as fnb:
        query_dict = yaml.load(fnb.read())
    return query_dict

# Read the De-isomorphic ground truth
def get_deiso_dict(return_deiso_query_dict=False):
    ss = NasBench201SearchSpace(17, 4, load_nasbench=False)
    query_dict = {}
    iso_group = []
    with open(os.path.join(nb201_dir, "deiso_dict.txt")) as iso_f:
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
    if return_deiso_query_dict:
        with open(os.path.join(nb201_dir, "non-isom.txt"), "r") as r_f:
            deiso_archs = r_f.read().strip().split("\n")
        query_dict = {arch: query_dict[arch] for arch in deiso_archs}
    return query_dict, iso_group

criteria = {
    "oneshot average": lambda x, y: np.mean(x),
    "linear": lambda x, y: np.corrcoef(x, y)[0][1],
    "kd": lambda x, y: stats.kendalltau(x, y).correlation,
    "spearmanr": lambda x, y: spearmanr(x, y).correlation,
    "BWR@K": lambda x, y: minmax_n_at_k(x, y),
    "P@tbK": lambda x, y : p_at_tb_k(x, y),
}

# Print the evaluation results 
def get_info(final_arch_dict, query_dict, gt_threshold=0., info_names=["oneshot average", "linear", "kd", "spearmanr", "BWR@K", "P@tbK"], acc_name="acc", cal_loss=True):
    if cal_loss:
        query_list, oneshot_acc_list, oneshot_loss_list = zip(*[(query_dict[arch], final_arch_dict[arch][acc_name], final_arch_dict[arch]["loss"]) for arch in final_arch_dict])
        oneshot_loss_list = - np.array(oneshot_loss_list)
    else:
        query_list, oneshot_acc_list = zip(*[(query_dict[arch], final_arch_dict[arch][acc_name] if acc_name else final_arch_dict[arch]) for arch in final_arch_dict if arch in query_dict])
    print("len of list: ", len(query_list))
    query_list = np.array(query_list) / 100
    oneshot_acc_list = np.array(oneshot_acc_list)

    if gt_threshold > 0:
        query_list = np.round(query_list / gt_threshold)


    res = {"oneshot_acc": {}}
    for info_name in info_names:
        res["oneshot_acc"][info_name] = criteria[info_name](oneshot_acc_list, query_list)
    if cal_loss:
        res["oneshot_loss"] = {}
        for info_name in info_names:
            res["oneshot_loss"][info_name] = criteria[info_name](oneshot_loss_list, query_list)
            if info_name == "oneshot average":
                res["oneshot_loss"][info_name] = - res["oneshot_loss"][info_name]
    return res

def get_zeroshot_info(final_arch_dict, query_dict, gt_threshold=0.,
        info_names=["oneshot average", "linear", "kd", "spearmanr", "BWR@K",
            "P@tbK"], bs=1):
    keys = list(list(final_arch_dict.values())[0].keys())
    keys = [k for k in keys if k != "model"]
    negative_keys = ["loss"]
    query_list, *oneshot_perfs = zip(*[(query_dict[arch],
        *[final_arch_dict[arch][k] for k in keys]) for arch in final_arch_dict])
    query_list = np.array(query_list) / 100
    oneshot_perfs = {k: v[:, :bs].mean(-1) for k, v in zip(keys, np.array(oneshot_perfs))}
    for k in negative_keys:
        oneshot_perfs[k] = -oneshot_perfs[k]

    if gt_threshold > 0:
        query_list = np.round(query_list / gt_threshold)

    res = {f"oneshot_{k}": {} for k in keys}
    for k in keys:
        for info_name in info_names:
            res[f"oneshot_{k}"][info_name] = \
                criteria[info_name](oneshot_perfs[k], query_list)
            if info_name == "oneshot average" and k in negative_keys:
                res[f"oneshot_{k}"][info_name] = - res[f"oneshot_{k}"][info_name]
    return res




# Post De-isomorphism
def iso_to_deiso(iso_dict, iso_group):
    with open(os.path.join(nb201_dir, "non-isom.txt")) as fiso:
        lines = fiso.readlines()[:6466]
        lines = [line.strip() for line in lines]
    deiso_dict = {}
    for group_ in iso_group:
        acc_list = []
        name_ = group_[0]
        for ele_ in group_:
            acc_list.append(iso_dict[ele_]["reward"])
            if ele_ in lines:
                name_ = ele_
        deiso_dict[name_] = np.mean(acc_list)
    return deiso_dict

def main():
    all_res = {}
    if args.type == "nb101":
        ss = NasBench101SearchSpace()
        ss._init_nasbench()
    for derive_file in derive_files:
        final_arch_dict = parse_derive(derive_file)
        print("Arch num:", len(final_arch_dict))
        log_path = str(derive_file).rsplit(".", 1)[0] + "_statistic.pkl"
        if args.type == "nb301":
            query_dict = get_nb301_iso_dict(args.model)
            non_sparse = get_info(final_arch_dict, query_dict)
            query_dict.save()
            if args.oneshot:
                sparse = get_info(final_arch_dict, query_dict, gt_threshold=0.01)
                info = {"non_sparse": non_sparse, "sparse": sparse}
            else:
                info = {}
                for bs in [1, 3, 5]:
                    non_sparse = get_zeroshot_info(final_arch_dict, query_dict, bs=bs)
                    info.update({bs: non_sparse})
        elif args.type == "nb101":
            model = ss.nasbench
            query_dict = get_nb101_dict(model)
            if args.oneshot:
                info = get_info(final_arch_dict, query_dict)
            else:
                info = {bs: get_zeroshot_info(final_arch_dict, query_dict, bs=bs)
                        for bs in [1, 3, 5]}
        else:
            # nb201
            if args.type == "iso2deiso":
                query_dict, iso_group = get_deiso_dict()
                info = get_info(iso_to_deiso(final_arch_dict, iso_group), query_dict,
                                acc_name=None, cal_loss=False)
            elif args.type == "deiso":
                query_dict, iso_group = get_deiso_dict()
                info = get_info(final_arch_dict, query_dict)
            elif args.type == "iso": # iso
                iso_dict = get_iso_dict()
                info = get_info(final_arch_dict, iso_dict,
                                acc_name="reward", cal_loss=False)
            elif args.type == "iso2deiso_withoutpost":
                query_dict, iso_group = get_deiso_dict(return_deiso_query_dict=True)
                print("len query dict: ", len(query_dict))
                info = get_info(final_arch_dict, query_dict, acc_name="reward", cal_loss=False)
        print("Save to {}".format(log_path), info)
        with open(log_path, "wb") as fw:
            pickle.dump(info, fw)
        all_res[derive_file] = info
    if args.save:
        with open(args.save, "wb") as fw:
            pickle.dump(all_res, fw)

if __name__ == "__main__":
    main()

