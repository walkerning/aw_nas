#pylint: disable-all
import pickle

import pytest

from aw_nas.utils.torch_utils import random_cnn_data

@pytest.mark.parametrize("case", [
    {
        # original
        "avail_depths": [1, 2, 3, 4, 6, 8, 12, 16, 24],
        "avail_widths": [16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]
    },
    {
        # width only
        "avail_depths": [4],
        "avail_widths": [16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]
    }
])
def test_germ_resnet(case):
    from aw_nas.germ import GermSearchSpace
    from aw_nas.weights_manager.base import BaseWeightsManager

    avail_depths = case["avail_depths"]
    avail_widths = case["avail_widths"]

    ss = GermSearchSpace()
    wm = BaseWeightsManager.get_class_("germ")(
        ss, "cuda", rollout_type="germ",
        germ_supernet_type="nds_resnet",
        germ_supernet_cfg={
            "num_classes": 10,
            "stem_type": "res_stem_cifar",
            "depths": [avail_depths] * 3,
            "widths": [avail_widths] * 3
        }
    )
    assert ss.get_size() == len(avail_depths)**3 * len(avail_widths)**3
    print("ResNet search space size: ", ss.get_size())

    # ---- random sample and forward ----
    data = random_cnn_data(device="cuda", batch_size=2, input_c=3, output_c=10)
    for _ in range(5):
        rollout = ss.random_sample()
        cand_net = wm.assemble_candidate(rollout)
        outputs = cand_net(data[0])
        assert outputs.shape == (2, 10)
        print(rollout)
        print(outputs)

def test_germ_resnexta():
    from aw_nas.germ import GermSearchSpace
    from aw_nas.weights_manager.base import BaseWeightsManager

    ss = GermSearchSpace()
    wm = BaseWeightsManager.get_class_("germ")(
        ss, "cuda", rollout_type="germ",
        germ_supernet_type="nds_resnexta",
        germ_supernet_cfg={
            "num_classes": 10,
            "stem_type": "res_stem_cifar",
            "group_search": True
        }
    )
    assert ss.get_size() == 11390625

    # ---- random sample and forward ----
    data = random_cnn_data(device="cuda", batch_size=2, input_c=3, output_c=10)
    for _ in range(5):
        rollout = ss.random_sample()
        cand_net = wm.assemble_candidate(rollout)
        outputs = cand_net(data[0])
        assert outputs.shape == (2, 10)
        print(rollout)
        print(outputs)


    # sub search space without `num_groups` search
    ss_nogroup = GermSearchSpace()
    wm = BaseWeightsManager.get_class_("germ")(
        ss_nogroup, "cuda", rollout_type="germ",
        germ_supernet_type="nds_resnexta",
        germ_supernet_cfg={
            "num_classes": 10,
            "stem_type": "res_stem_cifar",
            "group_search": False
        }
    )
    assert ss_nogroup.get_size() == 421875 # no group

def test_germ_resnexta_pickle():
    from aw_nas.germ import GermSearchSpace
    from aw_nas.weights_manager.base import BaseWeightsManager

    ss = GermSearchSpace()
    wm = BaseWeightsManager.get_class_("germ")(
        ss, "cuda", rollout_type="germ",
        germ_supernet_type="nds_resnexta",
        germ_supernet_cfg={
            "num_classes": 10,
            "stem_type": "res_stem_cifar",
            "group_search": True
        }
    )
    dump_res = pickle.dumps(wm.super_net)
    reloaded_supernet = pickle.loads(dump_res)
    for name, decision in reloaded_supernet.named_decisions():
        print(name, decision.to_string())
