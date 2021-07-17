import os
import sys
import yaml
import pickle

from aw_nas import germ
from aw_nas.weights_manager.base import BaseWeightsManager
from aw_nas.common import rollout_from_genotype_str

ss = germ.GermSearchSpace()
wm = BaseWeightsManager.get_class_("germ")(
    ss, "cuda", rollout_type="germ",
    germ_supernet_type="nds_resnexta",
    germ_supernet_cfg={
        "num_classes": 10,
        "stem_type": "res_stem_cifar",
        "group_search": True
    }
)

arch_file = sys.argv[1]
gt_file = sys.argv[2]

# ---- parse arch file ----
with open(arch_file, "r") as r_f:
    archs = yaml.load(r_f)

nogroup_archs = []
for arch in archs:
    rollout = rollout_from_genotype_str(arch, ss)
    if all(rollout['num_groups.{}'.format(i)] == 1 for i in range(3)):
        # all `num_groups` == 1
        [rollout.arch.pop("num_groups.{}".format(i)) for i in range(3)]
        nogroup_archs.append(rollout.genotype)

out_arch_fname = os.path.join(os.path.dirname(arch_file), "nogroup_{}".format(os.path.basename(arch_file)))
print("Dumped {} archs to {}".format(len(nogroup_archs), out_arch_fname))
with open(out_arch_fname, "w") as w_f:
    yaml.dump(nogroup_archs, w_f)

# ---- parse gt pickle file ----
with open(gt_file, "rb") as r_f:
    gt = pickle.load(r_f)

nogroup_gts = []
for arch, param, flops, acc in zip(*gt):
    rollout = rollout_from_genotype_str(arch, ss)
    if all(rollout['num_groups.{}'.format(i)] == 1 for i in range(3)):
        # all `num_groups` == 1
        [rollout.arch.pop("num_groups.{}".format(i)) for i in range(3)]
        nogroup_gts.append([rollout.genotype, param, flops, acc])

nogroup_gts = list(zip(*nogroup_gts))
out_gt_fname = os.path.join(os.path.dirname(gt_file), "nogroup_{}".format(os.path.basename(gt_file)))
with open(out_gt_fname, "wb") as w_f:
    pickle.dump(nogroup_gts, w_f)
print("Dumped {} gt entries to {}".format(len(nogroup_gts[0]), out_gt_fname))
