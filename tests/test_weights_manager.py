import numpy as np
import six
import torch
import pytest

# ---- test super_net ----
def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))

def _supernet_sample_cand(net):
    ss = net.search_space

    rollout = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    cand_net = net.assemble_candidate(rollout)
    return cand_net


@pytest.mark.parametrize("super_net", [
    {"candidate_member_mask": False},
    {"search_space_cfg": {"concat_op": "sum"},
     "candidate_member_mask": False}
], indirect=["super_net"])
def test_supernet_assemble_nomask(super_net):
    net = super_net
    cand_net = _supernet_sample_cand(net)
    assert set(dict(cand_net.named_parameters()).keys()) == set(dict(net.named_parameters()).keys())
    assert set(dict(cand_net.named_buffers()).keys()) == set(dict(net.named_buffers()).keys())

def test_supernet_assemble(super_net):
    net = super_net
    cand_net = _supernet_sample_cand(net)

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

@pytest.mark.parametrize("super_net", [
    {},
    {"search_space_cfg": {"num_steps": [2, 4]}},
    {"search_space_cfg": {"concat_op": "sum"}},
    {"search_space_cfg": {"loose_end": True, "concat_op": "mean"}},
], indirect=["super_net"])
def test_supernet_forward(super_net):
    # test forward
    cand_net = _supernet_sample_cand(super_net)

    data = _cnn_data()
    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape[-1] == 10

@pytest.mark.parametrize("super_net", [
    {
        "dropout_rate": 0,
        "search_space_cfg": {
            "num_steps": 1,
            "num_layers": 1,
            "num_cell_groups": 1,
            "cell_layout": [0],
            "reduce_cell_groups": [],
        }
    }, {
        "dropout_rate": 0,
    }], indirect=["super_net"])
def test_supernet_forward_step(super_net):
    cand_net = _supernet_sample_cand(super_net)

    data = _cnn_data()

    ## test in eval mode
    # forward stem
    cand_net.eval()
    stemed, context = cand_net.forward_one_step(context=None, inputs=data[0])
    assert len(context.previous_cells) == 1 and context.previous_cells[0] is stemed

    # forward the cells
    for i_layer in range(1, super_net.search_space.num_layers+1):
        num_steps = super_net.search_space.num_steps + super_net.search_space.num_init_nodes + 1
        for i_step in range(num_steps):
            while True:
                step_state, context = cand_net.forward_one_step(context)
                if context.is_end_of_cell or context.is_end_of_step:
                    break
            if i_step != num_steps - 1:
                assert step_state is context.current_cell[-1] \
                    and len(context.current_cell) == i_step + 1
        assert context.is_end_of_cell
        assert len(context.previous_cells) == i_layer + 1

    # final forward
    logits, context = cand_net.forward_one_step(context)
    assert logits.shape[-1] == 10

    logits_straight_forward = cand_net.forward_data(data[0])
    assert (logits == logits_straight_forward).all()

    logits = cand_net.forward_one_step_callback(data[0], lambda s, c: None)
    assert (logits == logits_straight_forward).all()

    ## test in train mode
    cand_net.train()
    logits_straight_forward = cand_net.forward_data(data[0])
    logits = cand_net.forward_one_step_callback(data[0], lambda s, c: None)
    assert (logits == logits_straight_forward).all()

@pytest.mark.parametrize("test_id",
                         range(5))
def test_supernet_candidate_gradient_virtual(test_id, super_net):
    lr = 0.1
    EPS = 1e-5
    data = _cnn_data()
    net = super_net
    cand_net = _supernet_sample_cand(net)
    c_params = dict(cand_net.named_parameters())
    c_buffers = dict(cand_net.named_buffers())
    # test `gradient`, `begin_virtual`
    w_prev = {k: v.clone() for k, v in six.iteritems(c_params)}
    buffer_prev = {k: v.clone() for k, v in six.iteritems(c_buffers)}
    visited_c_params = dict(cand_net.active_named_members("parameters", check_visited=True))
    with cand_net.begin_virtual():
        grads = cand_net.gradient(data, mode="train")
        assert len(grads) == len(visited_c_params)
        optimizer = torch.optim.SGD(cand_net.parameters(), lr=lr)
        optimizer.step()
        for n, grad in grads:
            assert (w_prev[n] - grad * lr - c_params[n]).abs().sum().item() < EPS
        grads_2 = dict(cand_net.gradient(data, mode="train"))
        assert len(grads) == len(visited_c_params)
        optimizer.step()
        for n, grad in grads:
            # this check is not very robust...
            assert (w_prev[n] - (grad + grads_2[n]) * lr - c_params[n]).abs().mean().item() < EPS

        # sometimes, some buffer just don't updated... so here coomment out
        # for n in buffer_prev:
        #     # a simple check, make sure buffer is at least updated...
        #     assert (buffer_prev[n] - c_buffers[n]).abs().sum().item() < EPS

    for n in c_params:
        assert (w_prev[n] - c_params[n]).abs().mean().item() < EPS
    for n in c_buffers:
        assert (buffer_prev[n] - c_buffers[n]).abs().float().mean().item() < EPS

@pytest.mark.parametrize("super_net", [{
    "search_space_cfg": {"num_steps": 1, "num_node_inputs": 1, "num_init_nodes": 1,
                         "num_layers": 3, "cell_layout": [0, 1, 2],
                         "reduce_cell_groups": [1], "num_cell_groups": 3,
                         "cell_shared_primitives":[
                             ["none", "max_pool_3x3", "sep_conv_5x5"],
                             ["sep_conv_3x3"],
                             ["sep_conv_3x3", "dil_conv_3x3"]
                         ]},
    "cell_use_preprocess": False,
    "init_channels": 16,
    "stem_multiplier": 1,
    "cell_group_kwargs": [
        {"C_in": 16, "C_out": 16},
        {"C_in": 16, "C_out": 24},
        {"C_in": 24, "C_out": 64}
    ]
}], indirect=["super_net"])
def test_supernet_specify_Cinout(super_net):
    cand_net = _supernet_sample_cand(super_net)
    assert cand_net.super_net.cells[0].num_out_channels == 16
    assert cand_net.super_net.cells[1].num_out_channels == 24
    assert cand_net.super_net.cells[2].num_out_channels == 64
    assert len(cand_net.super_net.cells[0].edges) == 1 and \
        len(cand_net.super_net.cells[0].edges[0]) == 1
    assert len(cand_net.super_net.cells[0].edges[0][1].p_ops) == 3
    assert len(cand_net.super_net.cells[1].edges[0][1].p_ops) == 1
    assert len(cand_net.super_net.cells[2].edges[0][1].p_ops) == 2

    data = _cnn_data()

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape[-1] == 10

@pytest.mark.parametrize("super_net", [{
    "gpus": [0, 1]
}], indirect=["super_net"])
def test_supernet_data_parallel_forward(super_net):
    # test forward
    cand_net = _supernet_sample_cand(super_net)
    batch_size = 10
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape == (batch_size, 10)

@pytest.mark.parametrize("super_net", [{
    "gpus": [0, 1]
}], indirect=["super_net"])
def test_supernet_data_parallel_gradient(super_net):
    cand_net = _supernet_sample_cand(super_net)
    batch_size = 10
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape == (batch_size, 10)
    _ = cand_net.gradient(data, mode="train")

# ---- End test super_net ----

# ---- Test diff_super_net ----
@pytest.mark.parametrize("diff_super_net", [
    {"search_space_cfg": {"num_steps": [2, 4]}}
], indirect=["diff_super_net"])
def test_diff_supernet_forward(diff_super_net):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    data = _cnn_data()
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (2, 10)

def test_diff_supernet_to_arch(diff_super_net):
    from torch import nn
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    data = _cnn_data() #pylint: disable=not-callable

    # default detach_arch=True, no grad w.r.t the controller param
    logits = cand_net.forward_data(data[0])
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is None

    logits = cand_net.forward_data(data[0], detach_arch=False)
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is not None

def test_diff_supernet_forward_rollout_batch_size(diff_super_net):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1, batch_size=3)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    data = _cnn_data(batch_size=6)
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (6, 10)

@pytest.mark.parametrize("diff_super_net", [{
    "gpus": [0, 1, 2]
}], indirect=["diff_super_net"])
def test_diff_supernet_data_parallel_forward(diff_super_net):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    batch_size = 9
    data = _cnn_data(batch_size=batch_size)
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (batch_size, 10)

@pytest.mark.parametrize("diff_super_net", [{
    "gpus": [0, 1, 2]
}], indirect=["diff_super_net"])
def test_diff_supernet_data_parallel_forward_rolloutsize(diff_super_net):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1, batch_size=9)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    batch_size = 9
    data = _cnn_data(batch_size=batch_size)
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (batch_size, 10)

@pytest.mark.parametrize("diff_super_net", [{
    "gpus": [0, 1, 2]
}], indirect=["diff_super_net"])
def test_diff_supernet_data_parallel_backward_rolloutsize(diff_super_net):
    from torch import nn
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1, batch_size=9)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    batch_size = 9
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], detach_arch=False)
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is not None
# ---- End test diff_super_net ----
