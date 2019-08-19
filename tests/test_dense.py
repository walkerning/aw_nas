import pytest
import torch
import numpy as np

def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))

@pytest.mark.parametrize("case", [
    {
        "arch": [
            [4] * 6,
            [4] * 12,
            [4] * 24,
            [4] * 6
        ],
        "stem": 8,
        "trans": [16, 32, 64]
    },
    {
        "arch": [
            [4] * 6,
            [6] * 6,
            [4, 5, 6, 7, 8, 9, 10]
        ],
        "stem": 11,
        "trans": [17, 26]
    }
])
def test_dense_rollout(case):
    from aw_nas.common import get_search_space, genotype_from_str
    ss = get_search_space("cnn_dense", num_dense_blocks=len(case["arch"]))
    genotype = ss.genotype(case["arch"])
    assert genotype.stem == case["stem"]
    for i, trans_c in enumerate(case["trans"]):
        assert getattr(genotype, "transition_{}".format(i)) == trans_c
    assert genotype_from_str(str(genotype), ss) == genotype
    print(genotype)
    print("relative conv flops: ", ss.relative_conv_flops(case["arch"]))


def test_construct_final_densenet():
    from aw_nas.common import get_search_space
    from aw_nas.final.dense import DenseGenotypeModel
    arch = [
        [4] * 6,
        [4] * 12,
        [4] * 24,
        [4] * 6
    ]
    ss = get_search_space("cnn_dense", num_dense_blocks=len(arch))
    genotype_str = str(ss.genotype(arch))
    model = DenseGenotypeModel(ss, torch.device("cuda"), genotypes=genotype_str)
    data = _cnn_data()
    logits = model(data[0])
    assert logits.shape[-1] == 10
