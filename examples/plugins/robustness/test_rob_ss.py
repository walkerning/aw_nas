# pylint: disable-all
import os

import numpy as np
import torch
import pytest


def _cnn_data(device="cuda", batch_size=2):
    return (
        torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
        torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device),
    )


def test_ss():
    from aw_nas.common import get_search_space, genotype_from_str

    ss = get_search_space(
        "dense_rob", cell_layout=[0, 0, 1, 0, 0, 1, 0, 0], reduce_cell_groups=[1]
    )
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)

    genotype_str = str(rollout.genotype)
    genotype_rec = genotype_from_str(genotype_str, ss)
    assert genotype_rec == rollout.genotype

    # test msrobnet-1560M and its search space
    ss = get_search_space(
        "dense_rob",
        num_cell_groups=8,
        num_init_nodes=2,
        cell_layout=[0, 1, 2, 3, 4, 5, 6, 7],
        reduce_cell_groups=[2, 5],
        num_steps=4,
        primitives=["none", "skip_connect", "sep_conv_3x3", "ResSepConv"],
    )

    genotype_str = "DenseRobGenotype(normal_0='init_node~2+|skip_connect~0|sep_conv_3x3~1|+|none~0|skip_connect~1|skip_connect~2|+|none~0|sep_conv_3x3~1|ResSepConv~2|skip_connect~3|+|skip_connect~0|sep_conv_3x3~1|sep_conv_3x3~2|sep_conv_3x3~3|sep_conv_3x3~4|', normal_1='init_node~2+|ResSepConv~0|sep_conv_3x3~1|+|none~0|sep_conv_3x3~1|skip_connect~2|+|ResSepConv~0|ResSepConv~1|ResSepConv~2|none~3|+|none~0|skip_connect~1|sep_conv_3x3~2|sep_conv_3x3~3|skip_connect~4|', reduce_2='init_node~2+|ResSepConv~0|skip_connect~1|+|sep_conv_3x3~0|none~1|none~2|+|skip_connect~0|ResSepConv~1|sep_conv_3x3~2|none~3|+|ResSepConv~0|skip_connect~1|ResSepConv~2|none~3|skip_connect~4|', normal_3='init_node~2+|skip_connect~0|skip_connect~1|+|ResSepConv~0|none~1|ResSepConv~2|+|ResSepConv~0|none~1|ResSepConv~2|skip_connect~3|+|none~0|sep_conv_3x3~1|none~2|skip_connect~3|sep_conv_3x3~4|', normal_4='init_node~2+|ResSepConv~0|sep_conv_3x3~1|+|skip_connect~0|skip_connect~1|none~2|+|sep_conv_3x3~0|skip_connect~1|sep_conv_3x3~2|sep_conv_3x3~3|+|ResSepConv~0|ResSepConv~1|none~2|skip_connect~3|sep_conv_3x3~4|', reduce_5='init_node~2+|sep_conv_3x3~0|sep_conv_3x3~1|+|ResSepConv~0|sep_conv_3x3~1|ResSepConv~2|+|none~0|sep_conv_3x3~1|ResSepConv~2|sep_conv_3x3~3|+|ResSepConv~0|skip_connect~1|skip_connect~2|skip_connect~3|sep_conv_3x3~4|', normal_6='init_node~2+|none~0|ResSepConv~1|+|none~0|sep_conv_3x3~1|ResSepConv~2|+|skip_connect~0|ResSepConv~1|ResSepConv~2|none~3|+|ResSepConv~0|skip_connect~1|ResSepConv~2|none~3|sep_conv_3x3~4|', normal_7='init_node~2+|none~0|none~1|+|none~0|ResSepConv~1|none~2|+|sep_conv_3x3~0|skip_connect~1|ResSepConv~2|none~3|+|ResSepConv~0|skip_connect~1|skip_connect~2|none~3|ResSepConv~4|')"
    rec_genotype = genotype_from_str(genotype_str, ss)

    # test a genotype does not fit for this search space
    with pytest.raises(TypeError):
        genotype_str = "DenseRobGenotype(normal_0='init_node~1+|sep_conv_3x3~0|+|skip_connect~0|none~1|+|sep_conv_3x3~0|sep_conv_3x3~1|skip_connect~2|+|skip_connect~0|skip_connect~1|sep_conv_3x3~2|none~3|', reduce_1='init_node~1+|none~0|+|none~0|sep_conv_3x3~1|+|none~0|none~1|sep_conv_3x3~2|+|sep_conv_3x3~0|none~1|none~2|none~3|', normal_2='init_node~1+|sep_conv_3x3~0|+|sep_conv_3x3~0|skip_connect~1|+|skip_connect~0|none~1|skip_connect~2|+|skip_connect~0|sep_conv_3x3~1|sep_conv_3x3~2|skip_connect~3|', reduce_3='init_node~1+|skip_connect~0|+|skip_connect~0|skip_connect~1|+|none~0|sep_conv_3x3~1|sep_conv_3x3~2|+|skip_connect~0|skip_connect~1|sep_conv_3x3~2|skip_connect~3|', normal_4='init_node~1+|sep_conv_3x3~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|sep_conv_3x3~1|sep_conv_3x3~2|+|sep_conv_3x3~0|none~1|skip_connect~2|skip_connect~3|')"
        rec_genotype = genotype_from_str(genotype_str, ss)


def test_ss_plot(tmp_path):
    from aw_nas.common import get_search_space, plot_genotype

    ss_cfgs = {
        "cell_layout": [0, 1, 2, 3, 4, 5],
        "num_init_nodes": 2,
        "num_cell_groups": 6,
        "reduce_cell_groups": [1, 3],
    }
    ss = get_search_space("dense_rob", **ss_cfgs)

    rollout = ss.random_sample()
    path = os.path.join(str(tmp_path), "cell")
    rollout.plot_arch(path, label="robnas cell example")
    print(rollout.genotype)
    print("Plot save to path: ", path)

    rollout_2 = ss.random_sample()
    path_2 = os.path.join(str(tmp_path), "cell_2")
    plot_genotype(
        str(rollout_2.genotype),
        dest=path_2,
        cls="dense_rob",
        label="robnas cell example",
        **ss_cfgs
    )
    print(rollout_2.genotype)
    print("Plot save to path: ", path_2)


def test_rob_final_model():
    from aw_nas.common import get_search_space
    from aw_nas.final.base import FinalModel

    ss = get_search_space(
        "dense_rob",
        cell_layout=[0, 1, 2, 3, 4, 5],
        num_cell_groups=6,
        reduce_cell_groups=[1, 3],
    )
    geno_str = str(ss.random_sample().genotype)

    model = FinalModel.get_class_("dense_rob_final_model")(ss, "cuda", geno_str)

    data = _cnn_data()
    logits = model(data[0])
    assert logits.shape[-1] == 10


def test_rob_weights_manager():
    import re
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager.base import BaseWeightsManager
    from aw_nas.final.base import FinalModel

    ss = get_search_space(
        "dense_rob",
        cell_layout=[0, 1, 2, 3, 4, 5],
        num_cell_groups=6,
        reduce_cell_groups=[1, 3],
    )
    wm = BaseWeightsManager.get_class_("dense_rob_wm")(ss, "cuda")
    rollout = ss.random_sample()
    cand_net = wm.assemble_candidate(rollout)
    print(
        "len parameters, all supernet params: ", len(list(cand_net.named_parameters()))
    )
    state_dict = cand_net.state_dict()
    print("partial statedict:", len(state_dict))

    geno_str = str(rollout.genotype)
    model = FinalModel.get_class_("dense_rob_final_model")(ss, "cuda", geno_str)
    # remove `p_ops.<num>.`
    final_state_dict = {
        re.sub("p_ops\.\d+\.", "", key): value for key, value in state_dict.items()
    }
    model.load_state_dict(final_state_dict)
