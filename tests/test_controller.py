
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
    {"type": "anchor_lstm", "cfg": {}},
    {"type": "embed_lstm", "cfg": {}},
])
def test_controller_network(case):
    from aw_nas.controller.rl_networks import BaseRLControllerNet
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    cls = BaseRLControllerNet.get_class_(case["type"])
    net = cls(search_space, device)

    batch_size = 3
    arch, log_probs, entropies, (hx, cx) = net.sample(batch_size)
    assert len(hx) == net.num_lstm_layers
    assert len(cx) == net.num_lstm_layers
    assert hx[0].shape == (batch_size, net.controller_hid)
    assert len(arch) == batch_size
    num_actions = len(arch[0][0]) + len(arch[0][1])
    assert log_probs.shape == (batch_size, num_actions)
    assert entropies.shape == (batch_size, num_actions)

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
