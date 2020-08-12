# pylint: disable-all
import pickle
import numpy as np
# from aw_nas.common import get_search_space
from aw_nas.btcs.nasbench_101 import NasBench101SearchSpace
from nasbench import api

search_space = NasBench101SearchSpace(load_nasbench=True)
fixed_statistics = list(search_space.nasbench.fixed_statistics.items())
print("Number of arch data: {}".format(len(fixed_statistics)))

valid_ratio = 0.1
num_valid = int(len(fixed_statistics) * valid_ratio)
train_data = []
for key, f_metric in fixed_statistics[:-num_valid]:
    num_v = f_metric["module_adjacency"].shape[0]
    if num_v < 7:
        padded_adj = np.concatenate((f_metric["module_adjacency"][:-1],
                                     np.zeros((7 - num_v, num_v), dtype=np.int8),
                                     f_metric["module_adjacency"][-1:]))
        padded_adj = np.concatenate((padded_adj[:, :-1], np.zeros((7, 7 - num_v)), padded_adj[:, -1:]), axis=1)
        padded_ops = f_metric["module_operations"][:-1] + ["none"] * (7 - num_v) + f_metric["module_operations"][-1:]
    else:
        padded_adj = f_metric["module_adjacency"]
        padded_ops = f_metric["module_operations"]
    arch = (padded_adj, search_space.op_to_idx(padded_ops))
    metrics = search_space.nasbench.computed_statistics[key]
    valid_acc = np.mean([metrics[108][i]["final_validation_accuracy"] for i in range(3)])
    half_valid_acc = np.mean([metrics[108][i]["halfway_validation_accuracy"]
                              for i in range(3)])
    train_data.append((arch, valid_acc, half_valid_acc))

valid_data = []
for key, f_metric in fixed_statistics[-num_valid:]:
    num_v = f_metric["module_adjacency"].shape[0]
    if num_v < 7:
        padded_adj = np.concatenate((f_metric["module_adjacency"][:-1],
                                     np.zeros((7 - num_v, num_v), dtype=np.int8),
                                     f_metric["module_adjacency"][-1:]))
        padded_adj = np.concatenate((padded_adj[:, :-1], np.zeros((7, 7 - num_v)), padded_adj[:, -1:]), axis=1)
        padded_ops = f_metric["module_operations"][:-1] + ["none"] * (7 - num_v) + f_metric["module_operations"][-1:]
    else:
        padded_adj = f_metric["module_adjacency"]
        padded_ops = f_metric["module_operations"]
    arch = (padded_adj, search_space.op_to_idx(padded_ops))
    metrics = search_space.nasbench.computed_statistics[key]
    valid_acc = np.mean([metrics[108][i]["final_validation_accuracy"] for i in range(3)])
    half_valid_acc = np.mean([metrics[108][i]["halfway_validation_accuracy"]
                              for i in range(3)])
    valid_data.append((arch, valid_acc, half_valid_acc))

with open("nasbench_allv.pkl", "wb") as wf:
    pickle.dump(train_data, wf)
with open("nasbench_allv_valid.pkl", "wb") as wf:
    pickle.dump(valid_data, wf)
