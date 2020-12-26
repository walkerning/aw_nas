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

