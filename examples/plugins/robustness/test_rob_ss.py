#pylint: disable-all
import os
import numpy as np
import torch

def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))


def test_ss():
    from aw_nas.common import get_search_space
    ss = get_search_space(
        "dense_rob", cell_layout=[0, 0, 1, 0, 0, 1, 0, 0],
        reduce_cell_groups=[1]
    )
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)


def test_ss_plot(tmp_path):
    from aw_nas.common import get_search_space, plot_genotype
    ss_cfgs = {
        "cell_layout": [0, 1, 2, 3, 4, 5],
        "num_init_nodes": 2,
        "num_cell_groups": 6,
        "reduce_cell_groups": [1, 3]
    }
    ss = get_search_space("dense_rob",  **ss_cfgs)

    rollout = ss.random_sample()
    path = os.path.join(str(tmp_path), "cell")
    rollout.plot_arch(path, label="robnas cell example")
    print(rollout.genotype)
    print("Plot save to path: ", path)

    rollout_2 = ss.random_sample()
    path_2 = os.path.join(str(tmp_path), "cell_2")
    plot_genotype(str(rollout_2.genotype), dest=path_2, cls="dense_rob",
                  label="robnas cell example", **ss_cfgs)
    print(rollout_2.genotype)
    print("Plot save to path: ", path_2)

def test_rob_final_model():
    from aw_nas.common import get_search_space
    from aw_nas.final.base import FinalModel

    ss = get_search_space(
        "dense_rob", cell_layout=[0, 1, 2, 3, 4, 5],
        num_cell_groups=6,
        reduce_cell_groups=[1, 3]
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
        "dense_rob", cell_layout=[0, 1, 2, 3, 4, 5],
        num_cell_groups=6,
        reduce_cell_groups=[1, 3]
    )
    wm = BaseWeightsManager.get_class_("dense_rob_wm")(ss, "cuda")
    rollout = ss.random_sample()
    cand_net = wm.assemble_candidate(rollout)
    print("len parameters, all supernet params: ", len(list(cand_net.named_parameters())))
    state_dict = cand_net.state_dict()
    print("partial statedict:", len(state_dict))

    geno_str = str(rollout.genotype)
    model = FinalModel.get_class_("dense_rob_final_model")(ss, "cuda", geno_str)
    # remove `p_ops.<num>.`
    final_state_dict = {re.sub("p_ops\.\d+\.", "", key): value for key, value in state_dict.items()}
    model.load_state_dict(final_state_dict)
