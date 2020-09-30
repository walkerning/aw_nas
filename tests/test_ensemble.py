import os
import pytest

@pytest.mark.parametrize("case", [
    {"cls": "cnn", "ensemble_size": 5},
    {"cls": "cnn", "loose_end": True}
])
def test_ensemble_search_space(case, tmp_path):
    from aw_nas.rollout.ensemble import EnsembleSearchSpace
    from aw_nas.common import rollout_from_genotype_str

    inner_search_space_type = case.pop("cls")
    ensemble_size = case.pop("ensemble_size", 3)
    ss = EnsembleSearchSpace(inner_search_space_type=inner_search_space_type,
                             inner_search_space_cfg=case,
                             ensemble_size=ensemble_size)

    ensemble_rollout = ss.random_sample()
    print(ensemble_rollout.genotype)
    rec_rollout = ss.rollout_from_genotype(ensemble_rollout.genotype)
    print(rec_rollout)
    assert ensemble_rollout == rec_rollout

    # test genotype from str
    rec_rollout_2 = rollout_from_genotype_str(str(ensemble_rollout.genotype), search_space=ss)
    assert ensemble_rollout == rec_rollout_2

    # test plot_arch
    paths = ensemble_rollout.plot_arch(os.path.join(str(tmp_path), "ensemble_arch"))
    print("Saved plots to {}".format(paths))


def test_ensemble_weights_manager():
    # test forward, assemble candidate, save, load
    # test candidate.forward
    # @TODO
    pass
