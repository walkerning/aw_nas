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
