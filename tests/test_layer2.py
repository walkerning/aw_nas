#pylint: disable-all
import os
import numpy as np
import torch
import pytest

def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))


@pytest.mark.parametrize("genotype_str", [
    None,
])
def test_micro_dense_ss(tmp_path, genotype_str):
    from aw_nas.common import get_search_space, rollout_from_genotype_str
    ss = get_search_space(
        "micro-dense", num_cell_groups=2, num_steps=4
    )
    if genotype_str is None:
        rollout = ss.random_sample()
        genotype_str = str(rollout.genotype)
    print(genotype_str)
    rollout_rec = rollout_from_genotype_str(genotype_str, ss)
    if genotype_str is None:
        assert [(rollout_rec.arch[i] == rollout.arch[i]).all() for i in range(len(rollout.arch))]
    else:
        assert str(rollout_rec.genotype) == genotype_str

    path = os.path.join(str(tmp_path), "cell")
    rollout.plot_arch(path, label="cell example")
    print("Plot save to path: ", path)

@pytest.mark.parametrize("search_space_cfg", [
    {
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "reduce_cell_groups": [1]
    },
    {
        "num_cell_groups": 3,
        "cell_layout": [0, 1, 2, 0, 1, 0, 2, 0, 0, 1, 0],
        "reduce_cell_groups": [2]
    },
    {
        "num_cell_groups": 2,
        "cell_layout": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        "reduce_cell_groups": [1]
    },
    {
        "num_cell_groups": 2,
        "cell_layout": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        "reduce_cell_groups": [1]
    }
])
def test_macro_ss(search_space_cfg, tmp_path):
    from aw_nas.common import get_search_space, rollout_from_genotype_str
    ss = get_search_space(
        "macro-stagewise", **search_space_cfg)
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)
    assert [(rollout_rec.arch[i] == rollout.arch[i]).all() for i in range(len(rollout.arch))]

    # genotype = ("StagewiseMacroGenotype(stage_0='num_node~3+||+||', stage_1='num_node~3+||+||', stage_2='num_node~4+||+|1|+|1|2|',"
    # " stage_3='num_node~3+|0|+||')")
    path = os.path.join(str(tmp_path), "macro")
    rollout.plot_arch(path, label="macro connection example")
    print("Plot save to path: ", path)


def test_layer2_ss(tmp_path):
    from aw_nas.common import get_search_space, genotype_from_str, rollout_from_genotype_str
    ss = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4
    })

    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)
    assert rollout_rec == rollout

    # genotype from str
    genotype = genotype_from_str(str(rollout.genotype), ss)
    rollout_rec2 = rollout_from_genotype_str(str(rollout.genotype), ss)
    assert rollout_rec2 == rollout

    # plot
    path = os.path.join(str(tmp_path), "layer2")
    rollout.plot_arch(path, label="layer2 rollout example")
    print("Plot save to path: ", path)

@pytest.mark.parametrize("genotype_str", [
#    None,
    "(StagewiseMacroGenotype(stage_0='num_node~4+|0|+|0|1|+|2|', stage_1='num_node~4+|0|+|0|+|1|2|', stage_2='num_node~4+|0|+|1|+|2|'), DenseMicroGenotype(cell_0='init_node~1+|sep_conv_3x3~0|+||+|sep_conv_3x3~0|skip_connect~1|sep_conv_3x3~2|+|skip_connect~0|sep_conv_3x3~0|skip_connect~1|skip_connect~2|skip_connect~3|sep_conv_3x3~3|', cell_1='init_node~1+|skip_connect~0|+||+|sep_conv_3x3~0|sep_conv_3x3~1|+|skip_connect~0|skip_connect~2|skip_connect~3|'))"
])
def test_layer2_final_model(genotype_str, tmp_path):
    from aw_nas.common import get_search_space, rollout_from_genotype_str
    from aw_nas.btcs.layer2.final_model import MacroStagewiseFinalModel

    ss = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 1, 0, 0],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4
    })
    if genotype_str is None:
        rollout = ss.random_sample()
    else:
        rollout = rollout_from_genotype_str(genotype_str, ss)

    final_model = MacroStagewiseFinalModel(
        ss, "cuda", str(rollout.genotype),
        micro_model_type="micro-dense-model",
        micro_model_cfg={
            "output_process_op": "nor_conv_1x1"
        },
        init_channels=12, use_stem="conv_bn_3x3")
    data = _cnn_data(device="cuda", batch_size=2)
    logits = final_model(data[0])
    assert logits.shape[-1] == 10

def test_layer2_controller():
    from aw_nas.common import get_search_space
    from aw_nas.btcs.layer2.controller import Layer2Controller, MacroStagewiseDiffController, MicroDenseDiffController
    ss = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4,
    })
    device="cuda"
    macro_controller_cfg = {
    }
    micro_controller_cfg = {
        # "use_edge_normalization": True,
    }
    controller=Layer2Controller(ss,"layer2",
                                mode="eval",
                                macro_controller_type="macro-stagewise-diff",
                                micro_controller_type="micro-dense-diff",
                                macro_controller_cfg=macro_controller_cfg,
                                micro_controller_cfg=micro_controller_cfg,
                                )
    rollouts = controller.sample(3)
    # print(controller.micro_controller.cg_alphas[0]) # ck the cg_alphas
    # print(controller.macro_controller.cg_alphas)
    from aw_nas import utils
    controller_opt_cfg={
        "type": "Layer2",
        "macro": {
            "type": "SGD",
            "lr": 0.01,
        },
        "micro": {
            "type":"Adam",
            "lr": 0.0005,
        }
    }
    controller_opt = utils.init_optimizer(controller.parameters(), controller_opt_cfg)

    # test backward
    rand_x = torch.rand(rollouts[0].macro.arch[0].shape).cuda()
    # (rollouts[0].macro.arch[0]*rand_x).sum().backward()
    (rollouts[0].macro.arch[0]).sum().backward()
    (rollouts[0].micro.arch[0]).sum().backward()
    print("--- the macro controller's grad ---")
    print(controller.macro_controller.cg_alphas[0].grad)
    print("--- the micro controller's grad ---")
    print(controller.micro_controller.cg_alphas[0].grad)

