#pylint: disable-all
import os
import pickle
from collections import OrderedDict, defaultdict

import json
import yaml
import numpy as np

NDS_DATA_HOME = "/home/eva_share_users/foxfi/nds_data/"

def dump_resnet():
    with open(os.path.join(NDS_DATA_HOME, "ResNet.json")) as r_f:
        results = json.load(r_f)

    resnet_genotypes = []
    resnet_params = []
    resnet_flops = []
    resnet_testaccs = []

    for result in results:
        resnet_genotype = str(OrderedDict(
            [("depths.{}".format(i), result["net"]["ds"][i + 1]) for i in range(3)] + \
            [("widths.{}".format(i), result["net"]["ws"][i + 1]) for i in range(3)]
        ))
        resnet_genotypes.append(resnet_genotype)
        resnet_params.append(result["params"])
        resnet_flops.append(result["flops"])
        resnet_testaccs.append(100 - result["min_test_top1"])

    print("Total {} archs".format(len(resnet_genotypes)))
    with open("resnet_archs_25k.yaml", "w") as w_f:
        yaml.dump(resnet_genotypes, w_f)
    with open("resnet_gt_25k.pkl", "wb") as w_f:
        pickle.dump([resnet_genotypes, resnet_params, resnet_flops, resnet_testaccs], w_f)


def dump_resnet_3seed():
    resnet_testaccs = defaultdict(list)
    resnet_genotypes = []
    resnet_params = []
    resnet_flops = []

    for seed in [1, 2, 3]:
        with open(os.path.join(NDS_DATA_HOME, "ResNet_rng{}.json".format(seed))) as r_f:
            results = json.load(r_f)

        for result in results:
            resnet_genotype = str(OrderedDict(
                [("depths.{}".format(i), result["net"]["ds"][i + 1]) for i in range(3)] + \
                [("widths.{}".format(i), result["net"]["ws"][i + 1]) for i in range(3)]
            ))
            resnet_testaccs[resnet_genotype].append(100 - result["min_test_top1"])
            if seed == 1:
                resnet_genotypes.append(resnet_genotype)
                resnet_params.append(result["params"])
                resnet_flops.append(result["flops"])
    resnet_testaccs_list = []
    for genotype in resnet_genotypes:
        assert len(resnet_testaccs[genotype]) == 3
        resnet_testaccs_list.append(float(np.mean(resnet_testaccs[genotype])))
    print("Total {} archs".format(len(resnet_genotypes)))

    with open("resnet_archs_5k_3seed.yaml", "w") as w_f:
        yaml.dump(resnet_genotypes, w_f)
    with open("resnet_gt_5k_3seed.pkl", "wb") as w_f:
        pickle.dump([resnet_genotypes, resnet_params, resnet_flops, resnet_testaccs_list], w_f)


def dump_resnexta():
    with open(os.path.join(NDS_DATA_HOME, "ResNeXt-A.json")) as r_f:
        results = json.load(r_f)

    resnext_genotypes = []
    resnext_params = []
    resnext_flops = []
    resnext_testaccs = []

    for result in results:
        resnext_genotype = str(OrderedDict(
            [("depths.{}".format(i), result["net"]["ds"][i + 1]) for i in range(3)] + \
            [("widths.{}".format(i), result["net"]["ws"][i + 1]) for i in range(3)] + \
            [("bot_muls.{}".format(i), result["net"]["bot_muls"][i + 1]) for i in range(3)] + \
            [("num_groups.{}".format(i), result["net"]["num_gs"][i + 1]) for i in range(3)]
            
        ))
        resnext_genotypes.append(resnext_genotype)
        resnext_params.append(result["params"])
        resnext_flops.append(result["flops"])
        resnext_testaccs.append(100 - result["min_test_top1"])

    print("Total {} archs".format(len(resnext_genotypes)))
    with open("resnexta_archs_25k.yaml", "w") as w_f:
        yaml.dump(resnext_genotypes, w_f)
    with open("resnexta_gt_25k.pkl", "wb") as w_f:
        pickle.dump([resnext_genotypes, resnext_params, resnext_flops, resnext_testaccs], w_f)

dump_resnet()
dump_resnet_3seed()
dump_resnexta()
