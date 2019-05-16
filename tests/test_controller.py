from aw_nas.common import get_search_space

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

def test_controller_network():
    from aw_nas.controller.rl_networks import AnchorControlNet

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    net = AnchorControlNet(search_space, device)
    batch_size = 3
    arch, log_probs, entropies, (hx, cx) = net.sample(batch_size)
    assert len(hx) == net.num_lstm_layers
    assert hx[0].shape == (batch_size, net.controller_hid)
    assert len(arch) == batch_size
    num_actions = len(arch[0][0]) + len(arch[0][1])
    assert log_probs.shape == (batch_size, num_actions)
    assert entropies.shape == (batch_size, num_actions)
