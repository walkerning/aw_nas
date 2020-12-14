import numpy as np
import torch
import pytest


# ---- Test fault_injection ----
def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float,
                       device=device),
            torch.tensor(np.random.randint(0, high=10,
                                           size=batch_size)).long().to(device))


def _supernet_sample_cand(net):
    ss = net.search_space

    rollout = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    cand_net = net.assemble_candidate(rollout)
    return cand_net


@pytest.mark.parametrize("super_net", [{
    "search_space_cfg": {
        "num_steps": 1,
        "num_layers": 1,
        "num_cell_groups": 1,
        "cell_layout": [0],
        "reduce_cell_groups": []
    }
}],
                         indirect=["super_net"])
def test_inject(super_net):
    cand_net = _supernet_sample_cand(super_net)

    data = _cnn_data()

    from aw_nas.objective.fault_injection import FaultInjector
    injector = FaultInjector(gaussian_std=None, mode="fixed")
    injector.set_random_inject(0.001)
    # forward stem
    cand_net.eval()

    def inject(state, context):
        if context.is_last_concat_op:
            return
        context.last_state = injector.inject(state)

    cand_net.forward_one_step_callback(data[0], callback=inject)


# # ---- Test robustness ----
# def test_generate_adv(super_net):
#     cand_net = _supernet_sample_cand(super_net)
#     data = _cnn_data()
#     from aw_nas.objective.robustness import PgdAdvGenerator
#     generator = PgdAdvGenerator(epsilon=0.03, n_step=5, step_size=0.0078, rand_init=False)
#     inputs_adv = generator.generate_adv(data[0], None, data[1], cand_net)

#     assert (inputs_adv - data[0] != 0).any()
#     assert ((inputs_adv - data[0]).abs().max() <= 0.03 + 1e-5).all()


@pytest.mark.parametrize("case", [{
    "search_space_cfg": {
        "width_choice": [3, 4, 6],
        "depth_choice": [2, 3, 4],
        "kernel_choice": [3, 5, 7],
        "image_size_choice": [224],
        "num_cell_groups": [1, 4, 4, 4, 4, 4],
        "expansions": [1, 6, 6, 6, 6, 6],
    },
    "prof_prims_cfg": {
        "spatial_size": 224,
        "primitive_type": "mobilenet_v3_block",
        "performances": ["latency"],
        "mult_ratio": 1.0,
        "base_channels": [16, 16, 24, 32, 64, 96, 160, 320, 1280],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [False, False, True, False, True, True],
        "acts": ["relu", "relu", "relu", "relu", "relu", "relu"]
    },
    "hardware_perfmodel_type": "regression",
    "hardware_perfmodel_cfg": {
        "preprocessors":
        ("block_sum", "remove_anomaly", "flatten", "extract_sum_features"),
        "prof_prims_cfg": {
            "primitive_type": 'mobilenet_v3_block',
            'spatial_size': 300,
            'strides': [ 1, 2, 2, 2, 1, 2 ],
            'base_channels': [ 16, 16, 24, 32, 64, 96, 160, 960, 1280 ],
            'mult_ratio': 1.0,
            'use_ses': [ False, False, True, False, True, True ],
            'acts': [ "relu", "relu", "relu", "h_swish", "h_swish", "h_swish" ]
        }
    },
    "perf_names": ["latency"],
    "prof_nets": [[{
        "overall_latency":
        26.2,
        "primitives": [{
            "prim_type": "mobilenet_v3_block",
            "C": 16,
            "C_out": 16,
            "spatial_size": 112,
            "expansion": 1,
            "use_se": False,
            "stride": 1,
            "affine": True,
            "kernel_size": 3,
            "activation": "relu",
            "performances": {"latency": 12.5},
        }, {
            "prim_type": "mobilenet_v3_block",
            "C": 24,
            "C_out": 24,
            "spatial_size": 56,
            "expansion": 4,
            "use_se": False,
            "stride": 1,
            "affine": True,
            "kernel_size": 5,
            "activation": "relu",
            "performances": {"latency": 14.5},
        }]
    }]],
    "genotypes":
    "image_size=224, cell_0=1, cell_1=2, cell_2=2, cell_3=1, cell_4=1, cell_5=1, cell_0_block_0=(1, 3), cell_1_block_0=(1, 3), cell_1_block_1=(4, 5), cell_1_block_2=(6, 3), cell_1_block_3=(2, 3), cell_2_block_0=(3, 3), cell_2_block_1=(3, 5), cell_2_block_2=(6, 3), cell_2_block_3=(5, 3), cell_3_block_0=(3, 5), cell_3_block_1=(3, 5), cell_3_block_2=(6, 3), cell_3_block_3=(3, 3), cell_4_block_0=(3, 3), cell_4_block_1=(2, 3), cell_4_block_2=(2, 3), cell_4_block_3=(6, 3), cell_5_block_0=(6, 3), cell_5_block_1=(6, 5), cell_5_block_2=(4, 5), cell_5_block_3=(4, 5)"
}])
def test_hardware(case):
    from collections import namedtuple

    from aw_nas.objective.hardware import HardwareObjective
    from aw_nas.common import get_search_space

    latency = [p["performances"]["latency"] for p in case["prof_nets"][0][0]["primitives"]]
    ss = get_search_space("ofa_mixin", **case["search_space_cfg"])
    if case["hardware_perfmodel_type"] == "regression":
        try:
            from sklearn import linear_model
        except ImportError as e:
            pytest.xfail("Do not install scikit-learn, this should fail")
    obj = HardwareObjective(search_space=ss, hardware_perfmodel_type=case['hardware_perfmodel_type'], hardware_perfmodel_cfg=case["hardware_perfmodel_cfg"], perf_names=case['perf_names'])
    obj.hardware_perfmodels[0].train(case["prof_nets"])
    rollout = ss.rollout_from_genotype(case["genotypes"])
    C = namedtuple("cand_net", ["rollout"])
    cand_net = C(rollout)
    perfs = obj.get_perfs(None, None, None, cand_net)
    assert 0 < perfs[0] < sum(latency)
