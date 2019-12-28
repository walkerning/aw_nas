import os
import pytest

AWNAS_TEST_NASBENCH = os.environ.get("AWNAS_TEST_NASBENCH", None)


@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
def test_nasbench():
    from aw_nas.common import get_search_space
    from aw_nas.btcs import nasbench

    # might take several minutes
    ss = get_search_space("nasbench-101")
    # construct controller
    controller = nasbench.NasBench101Controller(ss)
    compare_controller = nasbench.NasBench101CompareController(ss, rollout_type="compare")
    # construct evaluator
    evaluator = nasbench.NasBench101Evaluator(None, None, None)

    # test random sample
    _ = ss.random_sample()

    # test controller.sample
    rollouts = controller.sample(n=4)
    # test genotype
    print(rollouts[0].genotype)

    # test evaluator.evaluate_rollout
    rollouts = evaluator.evaluate_rollouts(rollouts, False)
    print(rollouts)

    evaluator.rollout_type = "compare"
    rollouts = compare_controller.sample(n=4)
    print(rollouts[0].genotype)

    # test evaluator.evaluate_rollout for compare rollouts
    rollouts = evaluator.evaluate_rollouts(rollouts, False)
    print(rollouts)
