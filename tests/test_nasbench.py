import os
import pytest
import numpy as np

AWNAS_TEST_NASBENCH = os.environ.get("AWNAS_TEST_NASBENCH", None)

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
@pytest.mark.parametrize("case", [
    {"cls": "nasbench-101", "load_nasbench": False, "validate_spec": False}])
def test_rollout_from_genotype_str(case):
    from aw_nas.common import get_search_space, rollout_from_genotype_str

    genotype_str = case.pop("genotype_str", None)
    ss = get_search_space(**case)
    if genotype_str:
        rec_rollout = rollout_from_genotype_str(genotype_str, ss)
    else:
        rollout = ss.random_sample()
        rec_rollout = rollout_from_genotype_str(str(rollout.genotype), ss)
        assert all(np.all(rec_rollout.arch[i] == rollout.arch[i])
                   for i in range(len(rec_rollout.arch)))

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
def test_plot_arch(tmp_path):
    from aw_nas.common import get_search_space
    from aw_nas.btcs.nasbench_101 import NasBench101Rollout

    nasbench_ss = get_search_space("nasbench-101", load_nasbench=False)
    prefix = os.path.join(str(tmp_path), "nb101-cell")
    arch_1 = (np.array([[0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8), [1, 2, 1, 1, 0])
    rollout = NasBench101Rollout(*arch_1, search_space=nasbench_ss)
    print("genotype: ", rollout.genotype, "save to: ", prefix)
    rollout.plot_arch(prefix, label="test plot")

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
@pytest.mark.parametrize(
    "case", [
        {"embedder_type": "nb101-lstm"},
        {"embedder_type": "nb101-lstm",
         "embedder_cfg": {
             "use_hid": True,
             "num_layers": 3
         }},
        {"embedder_type": "nb101-seq"},
        {"embedder_type": "nb101-flow"},
        {"embedder_type": "nb101-flow",
         "embedder_cfg": {
             "use_final_only": True,
             "use_global_node": True
         }},
        {"embedder_type": "nb101-gcn"},
        {"embedder_type": "nb101-gcn",
         "embedder_cfg": {
             "use_final_only": True,
             "use_global_node": True
         }},
    ])
def test_embedder(case):
    from aw_nas.evaluator.arch_network import ArchEmbedder
    from aw_nas.common import get_search_space

    nasbench_search_space = get_search_space("nasbench-101", load_nasbench=False)
    device = "cuda"
    embedder = ArchEmbedder.get_class_(case["embedder_type"])(
        nasbench_search_space,
        **case.get("embedder_cfg", {}))
    embedder.to(device)
    arch_1 = (np.array([[0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8), [1, 2, 1, 1, 0])
    arch_2 = (np.array([[0, 1, 0, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8),
            nasbench_search_space.op_to_idx(
                ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3',
                 'conv3x3-bn-relu', 'none', 'output']))
    print(arch_1)
    print(arch_2)
    print(embedder.forward([arch_1, arch_2]))
    # embedder.embed_and_transform_arch(arch_2)

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
def test_nasbench(nasbench_search_space):
    import numpy as np
    from scipy.stats import stats
    from aw_nas.btcs import nasbench_101
    from aw_nas.evaluator.arch_network import PointwiseComparator
    from aw_nas.rollout.compare import CompareRollout

    ss = nasbench_search_space
    # construct controller
    controller = nasbench_101.NasBench101Controller(ss, device="cuda")
    compare_controller = nasbench_101.NasBench101CompareController(ss, device="cuda", rollout_type="compare")
    # construct evaluator
    evaluator = nasbench_101.NasBench101Evaluator(None, None, None)

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
    true_scores = np.random.rand(len(rollouts))
    for i_step in range(5):
        loss = comparator.update_predict([r.arch for r in rollouts],
                                         true_scores)
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

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench BTC by default.")
def test_equal(nasbench_search_space):
    import numpy as np
    from nasbench import api

    ss = nasbench_search_space
    arch_1 = (np.array([[0, 0, 1, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8),
              [1, 2, 1, 1, 0])
    arch_2 = (np.array([[0, 0, 1, 0, 1, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0]], dtype=np.int8),
              [1, 2, 1, 1, 0])
    spec_1 = api.ModelSpec(arch_1[0], ["input"] + [ss.ops_choices[ind] for
                                                   ind in arch_1[1]] + ["output"])
    spec_2 = api.ModelSpec(arch_2[0], ["input"] + [ss.ops_choices[ind] for
                                                   ind in arch_2[1]] + ["output"])

    assert not spec_1 == spec_2
    r_1 = ss.rollout_from_genotype(spec_1)
    r_2 = ss.rollout_from_genotype(spec_2)
    assert r_1.genotype.hash_spec(ss.ops_choices) == r_2.genotype.hash_spec(ss.ops_choices)
    assert r_1 == r_2
    ss.compare_reduced = False
    assert r_1 != r_2
