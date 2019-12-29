import os
import pytest

AWNAS_TEST_NASBENCH = os.environ.get("AWNAS_TEST_NASBENCH", None)


@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
def test_nasbench(nasbench_search_space):
    from scipy.stats import stats
    from aw_nas.btcs import nasbench
    from aw_nas.evaluator.arch_network import PointwiseComparator
    from aw_nas.rollout.compare import CompareRollout

    ss = nasbench_search_space
    # construct controller
    controller = nasbench.NasBench101Controller(ss)
    compare_controller = nasbench.NasBench101CompareController(ss, rollout_type="compare")
    # construct evaluator
    evaluator = nasbench.NasBench101Evaluator(None, None, None)

    # test random sample
    _ = ss.random_sample()

    # test controller.sample
    rollouts = controller.sample(n=20)
    # test genotype
    print(rollouts[0].genotype)

    # test evaluator.evaluate_rollout
    rollouts = evaluator.evaluate_rollouts(rollouts, False)
    print(rollouts)

    evaluator.rollout_type = "compare"
    c_rollouts = compare_controller.sample(n=4)
    print(c_rollouts[0].genotype)

    # test evaluator.evaluate_rollout for compare rollouts
    c_rollouts = evaluator.evaluate_rollouts(c_rollouts, False)
    print(c_rollouts)

    # test nb101-gcn embedder
    comparator = PointwiseComparator(ss, arch_embedder_type="nb101-gcn",
                                     arch_embedder_cfg={"hid_dim": 96})
    comparator_2 = PointwiseComparator(ss, arch_embedder_type="nb101-gcn",
                                       arch_embedder_cfg={"hid_dim": 96})
    pred_scores = comparator.predict([r.arch for r in rollouts])
    pred_scores_2 = comparator_2.predict([r.arch for r in rollouts])
    label_scores = [r.perf["reward"] for r in rollouts]
    corr_init_1 = stats.kendalltau(label_scores, pred_scores.cpu().data.numpy()).correlation
    corr_init_2 = stats.kendalltau(label_scores, pred_scores_2.cpu().data.numpy()).correlation
    # compare_scores = comparator.compare([r.rollout_1.arch for r in c_rollouts],
    #                                     [r.rollout_2.arch for r in c_rollouts])

    # try training for several epochs using update_predict
    for i_step in range(5):
        loss = comparator.update_predict([(r.arch, r.perf["reward"]) for r in rollouts])
        print("update predict {}: {:.4f}".format(i_step, loss))

    # try training for several epochs using update_compare
    # construct compare rollouts between every pair in rollouts
    c_rollouts_2 = [CompareRollout(rollout_1=rollouts[i], rollout_2=rollouts[j])
                    for i in range(len(rollouts)) for j in range(i)]
    better_lst = [label_scores[j] > label_scores[i] for i in range(len(rollouts)) for j in range(i)]
    for i_step in range(5):
        loss = comparator_2.update_compare_rollouts(c_rollouts_2, better_lst)
        print("update compare {}: {:.4f}".format(i_step, loss))

    # test after training
    pred_scores_after = comparator.predict([r.arch for r in rollouts])
    pred_scores_2_after = comparator_2.predict([r.arch for r in rollouts])
    corr_after_1 = stats.kendalltau(label_scores, pred_scores_after.cpu().data.numpy()).correlation
    corr_after_2 = stats.kendalltau(
        label_scores, pred_scores_2_after.cpu().data.numpy()).correlation
    print("True accs: ", label_scores)
    print("PREDICT: before training: {} (corr {:.3f}); after training: {} (corr {:.3f})".format(
        pred_scores, corr_init_1, pred_scores_after, corr_after_1
    ))
    print("COMPARE: before training: {} (corr {:.3f}); after training: {} (corr {:.3f})".format(
        pred_scores_2, corr_init_2, pred_scores_2_after, corr_after_2
    ))
