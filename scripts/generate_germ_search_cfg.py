# -*- coding: utf-8 -*-
#pylint: disable-all

import os
import sys
import inspect
from pprint import pprint

import yaml

from aw_nas.common import SearchSpace
from aw_nas.germ import GermSuperNet

code_snippet_path = sys.argv[1]
search_cfg = sys.argv[2]

ss_cfg_path = code_snippet_path +".sscfg.yaml"
with open(code_snippet_path, "rb") as source_file:
    germ_def_module = {}
    code = compile(source_file.read(), code_snippet_path, "exec")
    exec(code, germ_def_module)

# find GermSuperNet class in the code snippet
if "GERM_SUPERNET" in germ_def_module:
    cls = germ_def_module["GERM_SUPERNET"]
    cls = germ_def_module[cls] if isinstance(cls, str) else cls
else:
    germ_supernets = []
    for def_ in germ_def_module.values():
        if inspect.isclass(def_) and issubclass(def_, GermSuperNet) and def_ is not GermSuperNet:
            germ_supernets.append(def_)
    if not germ_supernets:
        print("There are no GermSuperNet class definition in {}.")
        sys.exit(1)
    if len(germ_supernets) == 1:
        cls = germ_supernets[0]
    else:
        print("There are multiple GermSuperNet classes in {}. Please assign the class/class name to GERM_SUPERNET in the code snippet".format(code_snippet_path))
        sys.exit(1)

print("Use GermSuperNet definition:" , cls.__name__, cls)
super_net = cls()

# generate search space cfg
ss_cfg = super_net.generate_search_space_cfg()
print("Number of searchable blocks: {}; Number of decisions: {}".format(len(ss_cfg["blocks"]), sum([len(v) for v in ss_cfg["blocks"]])))
print("Number of independent decisions: {}".format(len(ss_cfg["decisions"])))
pprint(ss_cfg["decisions"])

with open(ss_cfg_path, "w") as w_f:
    yaml.dump(ss_cfg, w_f)

search_space = SearchSpace.get_class_("germ")(search_space_cfg_file=ss_cfg_path)
print("Search space size: ", search_space.get_size())

# generate search_space/weights_manager in search cfg
cfg = {
    "rollout_type": "germ",
    "search_space_type": "germ",
    "search_space_cfg": {
        "search_space_cfg_file": ss_cfg_path
    },

    "weights_manager_type": "germ",
    "weights_manager_cfg": {
        "rollout_type": "germ",
        "germ_supernet_type": cls.NAME,
        "germ_def_file": os.path.abspath(code_snippet_path)
    },
}

with open(search_cfg, "r") as fr:
    new_cfg = yaml.load(fr)

new_cfg.update(cfg)

with open(search_cfg, "w") as w_f:
    yaml.dump(new_cfg, w_f)
print("Dump the search_space/weights_manager cfg to {}. Other settings should be added manually, you can use `awnas gen-sample-config -r germ` to help!".format(search_cfg))
