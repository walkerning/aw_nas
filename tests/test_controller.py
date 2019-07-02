from __future__ import print_function
from aw_nas.common import get_search_space
import pytest

# ---- test controller rl ----
def test_rl_controller():
    import torch
    import numpy as np

    from aw_nas.controller import RLController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = RLController(search_space, device)
    controller_i = RLController(search_space, device,
                                independent_cell_group=True,
                                rl_agent_cfg={"batch_update": False})
    assert len(list(controller.parameters())) == 10
    assert len(list(controller_i.parameters())) == 20
    rollouts = controller.sample(3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]
    optimizer = torch.optim.SGD(controller.parameters(), lr=0.01)
    loss = controller.step(rollouts, optimizer)

    rollouts = controller_i.sample(3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]
    optimizer = torch.optim.SGD(controller_i.parameters(), lr=0.01)
    loss = controller_i.step(rollouts, optimizer)

@pytest.mark.parametrize("case", [
    {"type": "anchor_lstm"},
    {"type": "embed_lstm"}
])
def test_controller_network(case):
    from aw_nas.controller.rl_networks import BaseRLControllerNet
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    cls = BaseRLControllerNet.get_class_(case["type"])
    net = cls(search_space, device, None)

    batch_size = 3
    arch, log_probs, entropies, (hx, cx) = net.sample(batch_size)
    assert len(hx) == net.num_lstm_layers
    assert len(cx) == net.num_lstm_layers
    assert hx[0].shape == (batch_size, net.controller_hid)
    assert len(arch) == batch_size
    num_actions = len(arch[0][0]) + len(arch[0][1])
    assert log_probs.shape == (batch_size, num_actions)
    assert entropies.shape == (batch_size, num_actions)

@pytest.mark.parametrize("case", [
    {"type": "anchor_lstm"},
    {"type": "embed_lstm"}
])
def test_controller_network_cellwise_primitives(case):
    import numpy as np
    from aw_nas.controller.rl_networks import BaseRLControllerNet
    search_space = get_search_space(cls="cnn", num_cell_groups=2,
                                    cell_shared_primitives=[
                                        ("none",
                                         "avg_pool_3x3",
                                         "max_pool_3x3",
                                         "skip_connect"),
                                        ("skip_connect",
                                         "avg_pool_3x3", "dil_conv_3x3")
                                    ])
    device = "cuda"
    cls = BaseRLControllerNet.get_class_(case["type"])
    net0 = cls(search_space, device, cell_index=0)
    net1 = cls(search_space, device, cell_index=1)

    assert net0.w_op_soft.weight.shape[0] == 4
    assert net1.w_op_soft.weight.shape[0] == 3

    batch_size = 3
    arch, log_probs, entropies, (hx, cx) = net0.sample(batch_size)
    assert len(hx) == net0.num_lstm_layers
    assert len(cx) == net0.num_lstm_layers
    assert hx[0].shape == (batch_size, net0.controller_hid)
    assert len(arch) == batch_size
    num_actions = len(arch[0][0]) + len(arch[0][1])
    assert log_probs.shape == (batch_size, num_actions)
    assert entropies.shape == (batch_size, num_actions)

    net_shared = cls(search_space, device, cell_index=None)
    assert net_shared.w_op_soft.weight.shape[0] == 5

    batch_size = 3
    try:
        arch, log_probs, entropies, (hx, cx) = net_shared.sample(batch_size)
    except:
        pass
    else:
        assert 1, "must raise, when no `cell_index` is provided for `cnet.sample` "\
            "handled all cell groups with cellwise primitives"
    arch, log_probs, entropies, (hx, cx) = net_shared.sample(batch_size, cell_index=0)
    assert all(np.max(single_arch[1]) <= 3 for single_arch in arch)
    arch, log_probs, entropies, (hx, cx) = net_shared.sample(batch_size, cell_index=1)
    assert all(np.max(single_arch[1]) <= 2 for single_arch in arch)

def test_rl_agent_ppo():
    import numpy as np
    import torch
    from aw_nas.controller import RLController
    from aw_nas.controller.rl_agents import PPOAgent

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = RLController(search_space, device)
    agent = PPOAgent(controller)

    # set pseudo performance of rollouts
    rollouts = controller.sample(n=3)
    [r.set_perf(np.random.rand(), name="reward") for r in rollouts]

    ori_params = {n: v.clone() for n, v in controller.named_parameters()}
    lr = 0.01
    optimizer = torch.optim.SGD(controller.parameters(), lr=lr)
    agent.step(rollouts, optimizer)
    for n, v in controller.named_parameters():
        assert (ori_params[n] - v).abs().mean() > 1e-10

# --- test controller differentiable ----
def test_diff_controller():
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)

    assert controller.cg_alphas[0].shape == (14, len(search_space.shared_primitives))
    rollouts = controller.sample(3)
    assert isinstance(rollouts[0].genotype, search_space.genotype_type)

def test_diff_controller_use_prob():
    from aw_nas import utils
    import numpy as np
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, use_prob=True)

    assert controller.cg_alphas[0].shape == (14, len(search_space.shared_primitives))
    rollouts = controller.sample(3)
    assert np.abs((utils.get_numpy(rollouts[0].sampled[0]) - utils.softmax(rollouts[0].logits[0])))\
             .mean() < 1e-6
    assert isinstance(rollouts[0].genotype, search_space.genotype_type)

def test_diff_controller_force_uniform():
    import numpy as np
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, force_uniform=True, use_prob=True)

    rollouts = controller.sample(1)
    assert np.equal(rollouts[0].sampled[0].data, 1./len(search_space.shared_primitives) * \
                    np.ones((14, len(search_space.shared_primitives)))).all()

def test_diff_controller_cellwise_primitives():
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn", num_cell_groups=2,
                                    cell_shared_primitives=[
                                        ("none",
                                         "avg_pool_3x3",
                                         "max_pool_3x3",
                                         "skip_connect"),
                                        ("skip_connect",
                                         "avg_pool_3x3", "dil_conv_3x3")
                                    ])
    device = "cuda"
    controller = DiffController(search_space, device)
    assert controller.cg_alphas[0].shape == (14, 4)
    assert controller.cg_alphas[1].shape == (14, 3)
    rollout = controller.sample(1)[0]
    assert isinstance(rollout.genotype, search_space.genotype_type)
    assert set([conn[0] for conn in rollout.genotype.normal_0]).issubset(["none",
                                                                          "avg_pool_3x3",
                                                                          "max_pool_3x3",
                                                                          "skip_connect"])
    assert set([conn[0] for conn in rollout.genotype.reduce_1]).issubset(["avg_pool_3x3",
                                                                          "dil_conv_3x3",
                                                                          "skip_connect"])
    print(rollout.genotype)

def test_diff_controller_rollout_batch_size():
    import numpy as np
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)

    rollout = controller.sample(1, batch_size=4)[0]
    assert rollout.sampled[0].shape == (14, 4, len(search_space.shared_primitives))
    assert rollout.logits[0].shape == (14, len(search_space.shared_primitives))
    print(rollout.genotype)
