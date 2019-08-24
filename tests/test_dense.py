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

def test_dense_dynamic_rollout():
    from aw_nas.common import get_search_space, genotype_from_str
    ss = get_search_space("cnn_dense", dynamic_transition=True, reduction=None,
                          transition_channels=None, num_dense_blocks=4)
    arch = ([
        [4] * 6,
        [4] * 12,
        [4] * 24,
        [4] * 6
    ], [18, 20, 30])
    genotype = ss.genotype(arch)
    for i, trans_c in enumerate(arch[1]):
        assert getattr(genotype, "transition_{}".format(i)) == trans_c
    assert genotype_from_str(str(genotype), ss) == genotype
    print(genotype)
    print("relative conv flops: ", ss.relative_conv_flops(arch))

class _StubCfgTemplate(object):
    def create_cfg(self, genotype):
        return None

class _StubPopulation(object):
    def __init__(self, model_records):
        self.model_records = model_records
        self.cfg_template = _StubCfgTemplate()

    def get_model(self, ind):
        return self.model_records[ind]

@pytest.mark.parametrize("case", [
    {
        "search_space_cfg": {
            "num_dense_blocks": 4
        },
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
        "search_space_cfg": {
            "num_dense_blocks": 4,
            "first_ratio" : None,
            "stem_channel": 12,
            "reduction": None,
            "transition_channels": None,
            "dynamic_transition": True
        },
        "arch": [
            [
                [4] * 6,
                [4] * 12,
                [4] * 24,
                [4] * 6
            ], [
                12,
                24,
                48
            ]
        ],
        "stem": 12,
        "trans": [12, 24, 48]
    }])
def test_dense_mutation_rollout(case):
    from aw_nas.common import get_search_space
    from aw_nas.rollout.mutation import ModelRecord
    from aw_nas.rollout.dense import DenseMutation, DenseMutationRollout
    ss_cfg = case.pop("search_space_cfg", {})
    search_space = get_search_space("cnn_dense", **ss_cfg)
    par_genotype = search_space.genotype(case["arch"])

    assert par_genotype.stem == case["stem"]
    for i, trans_c in enumerate(case["trans"]):
        assert getattr(par_genotype, "transition_{}".format(i)) == trans_c

    # let's do some mutation
    mutations = [DenseMutation(search_space, DenseMutation.WIDER, block_idx=1, miniblock_idx=2,
                               modified=8)]
    if ss_cfg.get("dynamic_transition", False):
        mutations.append(DenseMutation(search_space, DenseMutation.TRANSITION, block_idx=1,
                                       miniblock_idx=2,
                                       modified=20))
    population = _StubPopulation({0: ModelRecord(par_genotype, None, search_space)})
    rollout = DenseMutationRollout(population, 0, mutations, search_space)
    print("parent genotype: ", par_genotype)
    print("mutated genotype: ", rollout.genotype)

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
