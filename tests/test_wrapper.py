import os
import numpy as np
import torch
import pytest

def _cnn_data(device="cuda", batch_size=2, shape=28):
    return (
        torch.rand(batch_size, 3, shape, shape, dtype=torch.float, device=device),
        torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device),
    )

def test_wrapper_ss(tmp_path):
    from aw_nas.common import (
        get_search_space,
        genotype_from_str,
        rollout_from_genotype_str,
    )

    ss = get_search_space(
        "wrapper",
        backbone_search_space_type="cnn",
        backbone_search_space_cfg={},
        backbone_rollout_type="discrete",
        neck_search_space_type=None,
        neck_search_space_cfg={},
        neck_rollout_type=None,
    )
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)
    assert rollout_rec == rollout

    # genotype from str
    genotype = genotype_from_str(str(rollout.genotype), ss)
    rollout_rec2 = rollout_from_genotype_str(str(rollout.genotype), ss)
    assert rollout_rec2 == rollout

    # plot
    path = os.path.join(str(tmp_path), "wrapper")
    rollout.plot_arch(path, label="wrapper rollout example")
    print("Plot save to path: ", path)


@pytest.mark.parametrize("case", ["discrete", "differentiable"])
def test_wrapper_supernet_classification(case):
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager.wrapper import WrapperWeightsManager
    from aw_nas.objective import ClassificationObjective

    ss = get_search_space(
        "wrapper",
        backbone_search_space_type="cnn",
        backbone_search_space_cfg={
            "num_cell_groups": 2,
            "num_init_nodes": 2,
            "num_layers": 5,
            "cell_layout": [0, 1, 0, 1, 0],
            "num_steps": 2,
            "num_node_inputs": 2,
        },
        backbone_rollout_type=case,
        neck_search_space_type=None,
        neck_search_space_cfg={},
        neck_rollout_type=None,
    )
    device = "cuda"
    if case == "discrete":
        rollout = ss.random_sample()
        backbone_type = "supernet"
    elif case == "differentiable":
        from aw_nas.controller import DiffController

        controller = DiffController(ss.backbone, device)
        rollout = controller.sample(1)[0]
        backbone_type = "diff_supernet"

    wm = WrapperWeightsManager(
        ss,
        device,
        rollout_type="wrapper",
        backbone_type=backbone_type,
        backbone_cfg={},
        neck_type=None,
        neck_cfg={},
        head_type="classification",
        head_cfg={"num_classes": 10},
        feature_levels=[-1],
    )

    data = _cnn_data(device="cuda", batch_size=2)
    cand_net = wm.assemble_candidate(rollout)
    outputs = cand_net(data[0])
    assert outputs.shape == (2, 10)
    obj = ClassificationObjective(ss)
    loss = obj.get_loss(data[0], outputs, data[1], cand_net)
    print(loss)

    wm_backbone = WrapperWeightsManager(
        ss.backbone,
        device,
        rollout_type=case,
        backbone_type=backbone_type,
        backbone_cfg={},
        neck_type=None,
        neck_cfg={},
        head_type="classification",
        head_cfg={"num_classes": 10},
        feature_levels=[-1],
    )
    data = _cnn_data(device="cuda", batch_size=2)
    cand_net = wm_backbone.assemble_candidate(rollout)
    outputs = cand_net(data[0])
    assert outputs.shape == (2, 10)
    obj = ClassificationObjective(ss)
    loss = obj.get_loss(data[0], outputs, data[1], cand_net)
    print(loss)


@pytest.mark.parametrize("case", ["ssd", "fpn"])
def test_wrapper_supernet_detection(case):
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager.wrapper import WrapperWeightsManager
    from aw_nas.objective import ClassificationObjective

    ss = get_search_space(
        "wrapper",
        backbone_search_space_type="ofa",
        backbone_search_space_cfg={
            "num_cell_groups": [1, 4, 4, 4, 4, 4, 1],
            "expansions": [1, 6, 6, 6, 6, 6, 6],
            "kernel_choice": [7, 5, 3],
            "width_choice": [6, 5, 4],
            "depth_choice": [4, 3, 2],
            "image_size_choice": [32],
        },
        backbone_rollout_type="ofa",
        neck_search_space_type=None,
        neck_search_space_cfg={},
        neck_rollout_type=None,
    )
    device = "cuda"
    backbone_type = "ofa_supernet"
    rollout = ss.random_sample()

    if case == "ssd":
        neck_cfg = {}
        head_type = "anchor_based"
        head_cfg = {"aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]}
        feature_levels = [4, 5]
    elif case == "fpn":
        neck_cfg = {"out_channels": 64, "pyramid_layers": 5}
        head_type = "atss"
        head_cfg = {"pyramid_layers": 5}
        feature_levels = [3, 4, 5]

    wm = WrapperWeightsManager(
        ss,
        device,
        rollout_type="wrapper",
        backbone_type=backbone_type,
        backbone_cfg={},
        neck_type=case,
        neck_cfg=neck_cfg,
        head_type=head_type,
        head_cfg={"num_classes": 20, **head_cfg},
        feature_levels=feature_levels,
    )

    data, _ = _cnn_data(device="cuda", batch_size=2, shape=300)
    cand_net = wm.assemble_candidate(rollout)

    if case == "ssd":
        conf, reg = cand_net(data)
        num_feats = 6
        shape = [19, 10, 5, 3, 2, 1]
    elif case == "fpn":
        conf, reg, centerness = cand_net(data)
        num_feats = 5
        shape = [38, 19, 10, 5, 3]

    assert len(conf) == num_feats and len(reg) == num_feats
    for i, s in enumerate(shape):
        conf[i].shape[-1] == s
        reg[i].shape[-1] == s
