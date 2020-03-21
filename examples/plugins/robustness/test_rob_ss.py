def test_ss():
    from aw_nas.common import get_search_space
    ss = get_search_space(
        "dense_rob", cell_layout=[0, 0, 1, 0, 0, 1, 0, 0],
        reduce_cell_groups=[1]
    )
    rollout = ss.random_sample()
    print(rollout.genotype)
    rollout_rec = ss.rollout_from_genotype(rollout.genotype)
