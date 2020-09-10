#pylint: disable-all
import os
import numpy as np
import torch
import pytest

def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))


def test_micro_dense_ss(tmp_path):
    from aw_nas.common import get_search_space
    ss = get_search_space(
        "micro-dense", num_cell_groups=2, num_steps=4
    )
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)
    assert [(rollout_rec.arch[i] == rollout.arch[i]).all() for i in range(len(rollout.arch))]

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
])
def test_macro_ss(search_space_cfg, tmp_path):
    from aw_nas.common import get_search_space
    ss = get_search_space(
        "macro-stagewise", **search_space_cfg)
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)
    assert [(rollout_rec.arch[i] == rollout.arch[i]).all() for i in range(len(rollout.arch))]

    path = os.path.join(str(tmp_path), "macro")
    rollout.plot_arch(path, label="macro connection example")
    print("Plot save to path: ", path)


def test_layer2_ss(tmp_path):
    from aw_nas.common import get_search_space
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

    path = os.path.join(str(tmp_path), "layer2")
    rollout.plot_arch(path, label="layer2 rollout example")
    print("Plot save to path: ", path)
