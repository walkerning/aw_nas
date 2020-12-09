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
        "width_choice": [0.25, 0.5, 0.75, 1.0],
        "reduce_cell_groups": [1]
    },
    {
        "num_cell_groups": 3,
        "cell_layout": [0, 1, 2, 0, 1, 0, 2, 0, 0, 1, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.0],
        "reduce_cell_groups": [2]
    },
    {
        "num_cell_groups": 2,
        "cell_layout": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.0],
        "reduce_cell_groups": [1]
    },
    {
        "num_cell_groups": 2,
        "cell_layout": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.0],
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
        "width_choice": [0.25, 0.5, 0.75, 1.0],
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
    # "(StagewiseMacroGenotype(width='0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5', stage_0='num_node~4+|0|+|1|+|2|', stage_1='num_node~4+|0|+||+|1|', stage_2='num_node~4+|0|+|1|+|2|'), DenseMicroGenotype(cell_0='init_node~1+|sep_conv_3x3~0|+||+|sep_conv_3x3~0|skip_connect~1|sep_conv_3x3~2|+|skip_connect~0|sep_conv_3x3~0|skip_connect~1|skip_connect~2|skip_connect~3|sep_conv_3x3~3|', cell_1='init_node~1+|skip_connect~0|+||+|sep_conv_3x3~0|sep_conv_3x3~1|+|skip_connect~0|skip_connect~2|skip_connect~3|'))"
    "(StagewiseMacroGenotype(width='0.5,0.5,0.5,0.25,0.25,0.25,1.0,1.0', stage_0='num_node~4+|0|+|1|+|2|', stage_1='num_node~4+|0|+||+|1|', stage_2='num_node~4+|0|+|1|+|2|'), DenseMicroGenotype(cell_0='init_node~1+|sep_conv_3x3~0|+||+|sep_conv_3x3~0|skip_connect~1|sep_conv_3x3~2|+|skip_connect~0|sep_conv_3x3~0|skip_connect~1|skip_connect~2|skip_connect~3|sep_conv_3x3~3|', cell_1='init_node~1+|skip_connect~0|+||+|sep_conv_3x3~0|sep_conv_3x3~1|+|skip_connect~0|skip_connect~2|skip_connect~3|'))"
])
def test_layer2_final_model(genotype_str, tmp_path):
    from aw_nas.common import get_search_space, rollout_from_genotype_str
    from aw_nas.btcs.layer2.final_model import MacroSinkConnectFinalModel

    ss = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 1, 0, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.0],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4
    })
    if genotype_str is None:
        rollout = ss.random_sample()
    else:
        rollout = rollout_from_genotype_str(genotype_str, ss)

    final_model = MacroSinkConnectFinalModel(
        ss, "cuda", str(rollout.genotype),
        micro_model_type="micro-dense-model",
        micro_model_cfg={
            "process_op_type": "nor_conv_1x1"
        },
        init_channels=12, use_stem="conv_bn_3x3")
    data = _cnn_data(device="cuda", batch_size=2)
    logits = final_model(data[0])
    assert logits.shape[-1] == 10

def test_layer2_controller():
    from aw_nas.common import get_search_space
    from aw_nas.btcs.layer2.controller import Layer2DiffController, Layer2Optimizer, MacroStagewiseDiffController, MicroDenseDiffController
    ss = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.],
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
    controller=Layer2DiffController(ss,"layer2",
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
    controller_opt = Layer2Optimizer(controller.parameters(), **controller_opt_cfg)
    # controller_opt = utils.init_optimizer(controller.parameters(), controller_opt_cfg)

    # test backward
    rand_x = torch.rand(rollouts[0].macro.arch[0].shape).cuda()
    # (rollouts[0].macro.arch[0]*rand_x).sum().backward()
    (rollouts[0].macro.arch[0]).sum().backward()
    (rollouts[0].micro.arch[0]).sum().backward()
    print("--- the macro controller's grad ---")
    print(controller.macro_controller.cg_alphas[0].grad)
    print("--- the micro controller's grad ---")
    print(controller.micro_controller.cg_alphas[0].grad)


def test_layer2_weights_manager():
    from aw_nas.common import get_search_space
    from aw_nas.btcs.layer2.weights_manager import Layer2MacroSupernet

    search_space = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4
    })

    macro_arch = [
        np.tril(np.ones(n), -1)
        for n in search_space.macro_search_space.stage_node_nums
    ]

    micro_arch = [
        np.repeat(np.tril(np.ones(5), -1)[:, :, np.newaxis], 2, axis=-1),
        np.repeat(np.tril(np.ones(5), -1)[:, :, np.newaxis], 2, axis=-1),
    ]

    # fully dense macro and micro rollout
    rollout = search_space.random_sample()
    rollout.macro.arch = macro_arch
    rollout.micro.arch = micro_arch

    device = torch.device("cuda")

    supernet = Layer2MacroSupernet(
        search_space, device
    )

    out = supernet.forward(torch.randn(10, 3, 32, 32).to(device), rollout)
    assert out.shape == torch.Size([10, 10])

def test_layer2_diff_weights_manager():
    from aw_nas.common import get_search_space
    from aw_nas.btcs.layer2.diff_weights_manager import  Layer2MacroDiffSupernet
    from aw_nas.btcs.layer2.controller import Layer2DiffController, MacroStagewiseDiffController, MicroDenseDiffController

    search_space = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4
    })

    macro_controller_cfg = {
    }
    micro_controller_cfg = {
        # "use_edge_normalization": True,
    }
    device = torch.device("cuda")
    controller=Layer2DiffController(search_space,"layer2",
                                mode="eval",
                                device=device,
                                # macro_controller_type="macro-stagewise-diff",
                                macro_controller_type="macro-sink-connect-diff",
                                micro_controller_type="micro-dense-diff",
                                macro_controller_cfg=macro_controller_cfg,
                                micro_controller_cfg=micro_controller_cfg,
                                )
    rollouts = controller.sample()
    rollout = rollouts[0]

    supernet = Layer2MacroDiffSupernet(
        search_space, device
    )

    out = supernet.forward(torch.randn(10, 3, 32, 32).to(device), rollout)
    assert out.shape == torch.Size([10, 10])
    out.sum().backward()
    print(controller.macro_controller.cg_alphas[0].grad)

def test_tfnas_macro_controller():
    from aw_nas.common import get_search_space
    from aw_nas.btcs.layer2.controller import Layer2DiffController, MacroStagewiseDiffController, MicroDenseDiffController
    ss = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "width_choice": [0.25, 0.5, 0.75, 1.],
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
    controller=Layer2DiffController(ss,"layer2",
                                mode="eval",
                                macro_controller_type="macro-sink-connect-diff",
                                micro_controller_type="micro-dense-diff",
                                macro_controller_cfg=macro_controller_cfg,
                                micro_controller_cfg=micro_controller_cfg,
                                )
    rollouts = controller.sample(3)
    (rollouts[0].macro.arch[0]).sum().backward()

@pytest.mark.parametrize("genotype_str", [
    None])

def test_check_connectivity(genotype_str):
    from aw_nas.common import get_search_space, rollout_from_genotype_str
    search_space = get_search_space("layer2", macro_search_space_cfg={
        "num_cell_groups": 2,
        "cell_layout": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "reduce_cell_groups": [1]
    }, micro_search_space_cfg={
        "num_cell_groups": 2,
        "num_steps": 4
    })
    for i in range(10):
        rollout = search_space.random_sample()
        print(rollout.macro.ck_connect())

