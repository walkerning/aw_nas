import os
import pytest

AWNAS_TEST_NASBENCH = os.environ.get("AWNAS_TEST_NASBENCH", None)

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
def test_plot_arch(tmp_path):
    import numpy as np
    from aw_nas.common import get_search_space
    from aw_nas.btcs.nasbench_201 import NasBench201Rollout

    nasbench_ss = get_search_space("nasbench-201", load_nasbench=False)
    prefix = os.path.join(str(tmp_path), "nb201-cell")
    arch_1 = np.array([[0., 0., 0., 0.],
                       [4., 0., 0., 0.],
                       [2., 4., 0., 0.],
                       [0., 0., 2., 0.]])
    rollout = NasBench201Rollout(arch_1, search_space=nasbench_ss)
    print("genotype: ", rollout.genotype, "save to: ", prefix)
    rollout.plot_arch(prefix, label="test plot")


@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
@pytest.mark.parametrize(
    "case", [
        {"embedder_type": "nb201-flow"},
        {"embedder_type": "nb201-seq"},
        {"embedder_type": "nb201-lstm"},
        {"embedder_type": "nb201-lstm",
         "embedder_cfg": {"use_hid": True}},
    ])
def test_embedder(case):
    import numpy as np
    from aw_nas.evaluator.arch_network import ArchEmbedder
    from aw_nas.common import get_search_space

    nasbench_search_space = get_search_space("nasbench-201", load_nasbench=False)
    device = "cuda"
    embedder = ArchEmbedder.get_class_(case["embedder_type"])(
        nasbench_search_space,
        **case.get("embedder_cfg", {}))
    embedder.to(device)
    arch_1 = np.array([[0., 0., 0., 0.],
                       [4., 0., 0., 0.],
                       [2., 4., 0., 0.],
                       [0., 0., 2., 0.]])
    arch_2 = np.array([[0., 0., 0., 0.],
                       [3., 0., 0., 0.],
                       [3., 1., 0., 0.],
                       [4., 3., 1., 0.]])
    print(arch_1)
    print(arch_2)
    print(embedder.forward([arch_1, arch_2]))
