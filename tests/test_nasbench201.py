import os
import pytest
import numpy as np
import torch
from torch import optim

AWNAS_TEST_NASBENCH = os.environ.get("AWNAS_TEST_NASBENCH", None)

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
@pytest.mark.parametrize("case", [
    {"cls": "nasbench-201", "load_nasbench": False}])
def test_rollout_from_genotype_str(case):
    from aw_nas.common import get_search_space, rollout_from_genotype_str

    genotype_str = case.pop("genotype_str", None)
    ss = get_search_space(**case)
    if genotype_str:
        rec_rollout = rollout_from_genotype_str(genotype_str, ss)
    else:
        rollout = ss.random_sample()
        rec_rollout = rollout_from_genotype_str(str(rollout.genotype), ss)
        assert np.all(np.array(rec_rollout.arch) == np.array(rollout.arch))

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
def test_plot_arch(tmp_path):
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


def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))

def _nasbench_201_sample_cand(net):
    ss = net.search_space

    rollout = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    cand_net = net.assemble_candidate(rollout)
    return cand_net


@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
@pytest.mark.parametrize("nasbench_201", [
    {"candidate_member_mask": False},
], indirect=["nasbench_201"])
def test_nasbench_201_assemble_nomask(nasbench_201):
    net = nasbench_201
    cand_net = _nasbench_201_sample_cand(net)
    assert set(dict(cand_net.named_parameters()).keys()) == set(dict(net.named_parameters()).keys())
    assert set(dict(cand_net.named_buffers()).keys()) == set(dict(net.named_buffers()).keys())

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
def test_nasbench_201_assemble(nasbench_201):
    net = nasbench_201
    cand_net = _nasbench_201_sample_cand(net)

    # test named_parameters/named_buffers
    # print("Supernet parameter num: {} ; buffer num: {}"\
    #       .format(len(list(net.parameters())), len(list(net.buffers()))))
    # print("candidatenet parameter num: {} ; buffer num: {}"\
    #       .format(len(list(cand_net.named_parameters())), len(list(cand_net.buffers()))))

    s_names = set(dict(net.named_parameters()).keys())
    c_names = set(dict(cand_net.named_parameters()).keys())
    assert len(s_names) > len(c_names)
    assert len(s_names.intersection(c_names)) == len(c_names)

    s_b_names = set(dict(net.named_buffers()).keys())
    c_b_names = set(dict(cand_net.named_buffers()).keys())
    assert len(s_b_names) > len(c_b_names)
    assert len(s_b_names.intersection(c_b_names)) == len(c_b_names)
    # names = sorted(set(c_names).difference([g[0] for g in grads]))

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
@pytest.mark.parametrize("nasbench_201", [
    {},
], indirect=["nasbench_201"])
def test_nasbench_201_forward(nasbench_201):
    # test forward
    cand_net = _nasbench_201_sample_cand(nasbench_201)

    data = _cnn_data()
    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape[-1] == 10

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
@pytest.mark.parametrize("nasbench_201", [{
    "gpus": [0, 1]
}], indirect=["nasbench_201"])
def test_nasbench_201_data_parallel_forward(nasbench_201):
    # test forward
    cand_net = _nasbench_201_sample_cand(nasbench_201)
    batch_size = 10
    data = _cnn_data(batch_size=batch_size)
    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape == (batch_size, 10)

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
@pytest.mark.parametrize("nasbench_201", [{
    "gpus": [0, 1]
}], indirect=["nasbench_201"])
def test_nasbench_201_data_parallel_gradient(nasbench_201):
    cand_net = _nasbench_201_sample_cand(nasbench_201)
    batch_size = 10
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape == (batch_size, 10)
    _ = cand_net.gradient(data, mode="train")

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
def test_nasbench_201_candidate_forward_with_params(nasbench_201):
    cand_net = _nasbench_201_sample_cand(nasbench_201)
    batch_size = 2
    data = _cnn_data(batch_size=batch_size)

    old_params = {n: p.clone() for n, p in
                  cand_net.active_named_members("parameters", check_visited=True)}
    print("num active params: ", len(old_params))
    # random init
    new_params = {n: torch.autograd.Variable(p.new(p.size()).normal_(), requires_grad=True)
                  for n, p in cand_net.active_named_members("parameters", check_visited=True)}
    res = torch.sum(cand_net.forward_with_params(data[0], new_params, mode="train"))

    # assert set back to original params
    for name, new_value in cand_net.named_parameters():
        if name in old_params:
            assert (new_value == old_params[name]).all()

    try:
        torch.autograd.grad(
            res,
            list(dict(cand_net.active_named_members("parameters", check_visited=True)).values()),
            retain_graph=True)
    except Exception:
        print("should raise!")
    else:
        assert False, "Should raise, as new params are used"
    torch.autograd.grad(res, list(new_params.values()), allow_unused=True)

@pytest.mark.skipif(
    not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default.")
def test_mutate_and_evo():
    from aw_nas.common import get_search_space
    search_space = get_search_space(cls="nasbench-201", load_nasbench=False)
    rollout = search_space.random_sample()
    mutated_rollout = search_space.mutate(rollout)
    print("before mutate: ", rollout)
    print("after mutate: ", mutated_rollout)

    from aw_nas.controller.evo import EvoController
    controller = EvoController(search_space, device="cpu", rollout_type="nasbench-201",
                               mode="train", population_size=4, parent_pool_size=2)
    # random sample 4
    rollouts = controller.sample(4)
    for rollout in rollouts:
        rollout.perf["reward"] = np.random.rand()
    controller.step(rollouts)
    new_rollouts = controller.sample(2)
    print(new_rollouts)
    for rollout in new_rollouts:
        rollout.perf["reward"] = np.random.rand()
    controller.step(new_rollouts)
    with controller.begin_mode("eval"):
        eval_rollouts = controller.sample(2)
        rewards = [r.perf["reward"] for r in rollouts + new_rollouts]
        print("all rollout rewards ever seen: ", rewards)
        print("eval sample (population): ", [r.perf["reward"] for r in eval_rollouts])
        controller.eval_sample_strategy = "all"
        eval_rollouts = controller.sample(2)
        print("eval sample (all): ", [r.perf["reward"] for r in eval_rollouts])

@pytest.mark.skipif(
        not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC by default")
def test_gcn_controller_gcn_mlp():
    from aw_nas.btcs.nasbench_201 import GCN
    from aw_nas.btcs.nasbench_201 import MLP
    gcn = GCN(4, 3, 10).cuda()
    mlp = MLP(4, 3, [20, 15, 10, 5]).cuda()
    x = torch.randn(4, 10).cuda()
    out = mlp(gcn(x))
    print(out)


@pytest.mark.skipif(
        not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC bu default")
def test_gcn_controller(tmp_path):
    from aw_nas.btcs.nasbench_201 import NasBench201GcnController, NasBench201DiffRollout, NB201DiffSharedNet, NB201CandidateNet
    from aw_nas.common import get_search_space
    search_space = get_search_space(cls="nasbench-201", load_nasbench=False)
    controller = NasBench201GcnController(search_space)
    rollouts = controller.sample(n=1)
    rollouts[0].plot_arch(os.path.join(str(tmp_path), "nb201_diff_rollout"))

    supernet = NB201DiffSharedNet(search_space, device="cuda")
    cand_net = supernet.assemble_candidate(rollouts[0])
    inputs = torch.randn((2, 3, 3, 3), dtype=torch.float32).to(torch.device("cuda"))
    outputs = cand_net(inputs)
    loss = outputs.sum()
    loss.backward()
    optimizer = optim.SGD(controller.parameters(), lr=0.001, momentum=0.9)
    controller.step(rollouts, optimizer)

@pytest.mark.skipif(
        not AWNAS_TEST_NASBENCH, reason="do not test the nasbench-201 BTC bu default")
def test_canonical_rejector_and_sampler():
    from aw_nas.controller.rejection import CanonicalTableRejector, RejectionSampleController
    from aw_nas.common import get_search_space
    from aw_nas.btcs.nasbench_201 import NasBench201Rollout

    search_space = get_search_space(cls="nasbench-201", load_nasbench=True)
    rejector = CanonicalTableRejector(search_space)
    indices = np.tril_indices(4, k=-1)
    def _pop_rollout(ops):
        arch = np.zeros((4, 4))
        arch[indices] = ops
        return NasBench201Rollout(arch, search_space=search_space)
    rollout_1 = _pop_rollout([1, 4, 4, 4, 1, 4])
    rollout_1_iso = _pop_rollout([1, 4, 4, 1, 4, 4])
    rollout_2 = _pop_rollout([1, 0, 0, 4, 1, 2])
    assert rejector.accept(rollout_1)
    assert rejector.accept(rollout_2)
    assert not rejector.accept(rollout_1_iso)

    # controller = RejectionSampleController(
    #         search_space, "cuda", rollout_type="nasbench-201", maximum_sample_threshold=1, base_sampler_cfg={})
    # # num should be definitely < 7000, since there are 6466 isomorphic groups in total
    # sampled_rollouts = controller.sample(n=7000, batch_size=1)
    # assert len(sampled_rollouts) < 7000
    # print(len(sampled_rollouts))

    # TODO: use sequential inner sampler, check table size equal 6466
    controller = RejectionSampleController(
        search_space, "cuda", rollout_type="nasbench-201", maximum_sample_threshold=1,
        base_sampler_type="nasbench-201-rs",
        base_sampler_cfg={"avoid_repeat": True, "deiso": False})
    # num should be 6466
    sampled_rollouts = controller.sample(n=16000, batch_size=1)
    print(len(sampled_rollouts))
    assert len(sampled_rollouts) == 6466
