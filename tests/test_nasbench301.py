import os
import pytest

AWNAS_TEST_NASBENCH = os.environ.get("AWNAS_TEST_NASBENCH", None)

@pytest.mark.skipif(not AWNAS_TEST_NASBENCH, reason="do not test the nasbench301 BTC by default.")
@pytest.mark.parametrize("case", ["xgb_v0.9", "gnn_gin_v0.9"])
def test_nb301_query(case):
    from aw_nas.btcs.nasbench_301 import NB301SearchSpace
    from aw_nas.btcs.nasbench_301 import NB301Evaluator

    ss = NB301SearchSpace()
    model_dir_ = os.path.expanduser("~/awnas/data/nasbench-301/nb_models")
    rollouts = [ss.random_sample() for _ in range(2)]
    # test mutate
    rollouts.append(ss.mutate(rollouts[0]))
    rollouts.append(ss.mutate(rollouts[1]))
    evaluator = NB301Evaluator(
        None, None, None,
        path=os.path.join(model_dir_, case))
    rollouts = evaluator.evaluate_rollouts(rollouts, False)
    print(rollouts)

@pytest.mark.skipif(
        not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-301 BTC bu default")
def test_canonical_rejector_and_sampler():
    from aw_nas.controller.rejection import CanonicalTableRejector, RejectionSampleController
    from aw_nas.common import get_search_space
    from aw_nas.btcs.nasbench_301 import NB301Rollout

    search_space = get_search_space(cls="nb301")
    rejector = CanonicalTableRejector(search_space)
    nodes1 = [0,1,0,1,2,3,2,3]
    ops1 = [1,1,2,2,3,3,3,3]
    nodes2 = [0,1,0,2,1,3,2,4]
    ops2 = [0,3,2,4,1,3,5,6]
    archs1 = []
    archs1.append((nodes1, ops1))
    archs1.append((nodes2, ops2))
    rollout1 = NB301Rollout(archs1, info={}, search_space=search_space)

    ops1 = [2,2,1,1,3,3,3,3]
    archs2 = []
    archs2.append((nodes1, ops1))
    archs2.append((nodes2, ops2))
    rollout2 = NB301Rollout(archs2, info={}, search_space=search_space)
    #assert rejector.accept(rollout1)
    #assert rejector.accept(rollout2)

    # controller = RejectionSampleController(
    #     search_space, "cuda", rollout_type="nb301", maximum_sample_threshold=1,
    #     base_sampler_type="random_sample",
    #     base_sampler_cfg={})

    # sampled_rollouts = controller.sample(n=1000000, batch_size=1)
    # print(len(sampled_rollouts))
