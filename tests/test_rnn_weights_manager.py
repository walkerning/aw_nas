import numpy as np
import six
import torch
from torch import nn

import pytest
from packaging import version


# ---- Test rnn_super_net ----
_criterion = nn.CrossEntropyLoss()
def _rnn_criterion(_, outputs, targets):
    logits, _, _, _ = outputs
    return _criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

def _rnn_data(time_steps, batch_size, num_tokens):
    data = torch.LongTensor(np.random.randint(0, high=num_tokens, #pylint: disable=no-member
                                              size=(time_steps+1, batch_size)))
    data = (data[:-1, :].cuda(), data[1:, :].cuda()) # inputs, targets
    return data

def _rnn_super_sample_cand(net):
    from aw_nas.common import Rollout
    ss = net.search_space

    rollout = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    cand_net = net.assemble_candidate(rollout)
    return cand_net

@pytest.mark.parametrize("rnn_super_net", [
    {"candidate_member_mask": False}], indirect=["rnn_super_net"])
def test_rnn_supernet_assemble_nomask(rnn_super_net):
    net = rnn_super_net
    cand_net = _rnn_super_sample_cand(net)
    assert set(dict(cand_net.named_parameters()).keys()) == set(dict(net.named_parameters()).keys())
    assert set(dict(cand_net.named_buffers()).keys()) == set(dict(net.named_buffers()).keys())

@pytest.mark.parametrize("rnn_super_net", [
    {
        "batchnorm_edge": True
    },
    {
        "batchnorm_edge": False
    },
    {
        "num_tokens": 10, "batchnorm_edge": False, "batchnorm_out": True,
        "tie_weight": True, "share_from_weights": True,
        "share_primitive_weights": True
    }], indirect=["rnn_super_net"])
def test_rnn_supernet_assemble(rnn_super_net):
    net = rnn_super_net
    cand_net = _rnn_super_sample_cand(net)

    # test named_parameters/named_buffers
    # print("Supernet parameter num: {} ; buffer num: {}"\
    #       .format(len(list(net.parameters())), len(list(net.buffers()))))
    # print("candidatenet parameter num: {} ; buffer num: {}"\
    #       .format(len(list(cand_net.named_parameters())), len(list(cand_net.buffers()))))

    s_names = set(dict(net.named_parameters()).keys())
    c_names = set(dict(cand_net.named_parameters()).keys())
    assert len(s_names) >= len(c_names)
    assert len(s_names.intersection(c_names)) == len(c_names)


    s_b_names = set(dict(net.named_buffers()).keys())
    c_b_names = set(dict(cand_net.named_buffers()).keys())
    assert len(s_b_names) >= len(c_b_names) or len(s_b_names) == 0
    assert len(s_b_names.intersection(c_b_names)) == len(c_b_names)
    # names = sorted(set(c_names).difference([g[0] for g in grads]))

@pytest.mark.parametrize("rnn_super_net", [{"num_tokens": 10}],
                         indirect=["rnn_super_net"])
def test_rnn_supernet_forward(rnn_super_net):
    # test forward
    cand_net = _rnn_super_sample_cand(rnn_super_net)

    time_steps = 5
    batch_size = 2
    _num_tokens = rnn_super_net.num_tokens
    _num_hid = rnn_super_net.num_hid
    _num_layers = rnn_super_net._num_layers

    hiddens = rnn_super_net.init_hidden(batch_size)

    assert tuple(hiddens.shape) == (_num_layers, batch_size, _num_hid)
    data = _rnn_data(time_steps, batch_size, _num_tokens)

    logits, _, outs, next_hiddens = cand_net.forward_data(data[0], mode="eval", hiddens=hiddens)
    assert tuple(logits.shape) == (time_steps, batch_size, _num_tokens)
    assert tuple(outs.shape) == (time_steps, batch_size, _num_hid)
    assert (hiddens == next_hiddens).all()

@pytest.mark.parametrize("rnn_super_net",
                         [
                             {"num_tokens": 10, "tie_weight": True},
                             {"num_tokens": 10, "tie_weight": False},
                             {"num_tokens": 10, "batchnorm_edge": False, "batchnorm_out": True,
                              "batchnorm_step": True,
                              "tie_weight": True, "share_from_weights": True,
                              "share_primitive_weights": True}
                         ],
                         indirect=["rnn_super_net"])
def test_rnn_supernet_candidate_gradient_virtual(rnn_super_net):
    if version.parse(torch.__version__).minor >= 7:
        pytest.xfail("FIXME: We currently do not fix this bug yet. When using torch>=1.7.0, "
                     "we encountered: Warning: Error detected in SplitBackward. "
                     "RuntimeError: one of the variables needed for gradient computation "
                     "has been modified by an inplace operation")

    lr = 0.1
    EPS = 1e-4
    time_steps = 5

    batch_size = 2
    _num_tokens = rnn_super_net.num_tokens
    _num_hid = rnn_super_net.num_hid

    hiddens = rnn_super_net.init_hidden(batch_size)
    # check hiddens not zero after these steps
    assert all(hid.abs().mean().item() < EPS for hid in hiddens)

    net = rnn_super_net
    cand_net = _rnn_super_sample_cand(net)
    c_params = dict(cand_net.named_parameters())
    c_buffers = dict(cand_net.named_buffers())
    # test `gradient`, `begin_virtual`
    w_prev = {k: v.clone() for k, v in six.iteritems(c_params)}
    buffer_prev = {k: v.clone() for k, v in six.iteritems(c_buffers)
                   if "num_batches_tracked" not in k} # mean/var

    data = _rnn_data(time_steps, batch_size, _num_tokens)

    with cand_net.begin_virtual():
        torch.autograd.set_detect_anomaly(True)
        grads = cand_net.gradient(data, mode="train", hiddens=hiddens, criterion=_rnn_criterion)
        assert len(grads) == len(c_params)
        optimizer = torch.optim.SGD(cand_net.parameters(), lr=lr, momentum=0.)
        # it seems momentum=0. is required for this test to pass torch>1.4.0
        optimizer.step()
        for n, grad in grads:
            assert (w_prev[n] - grad * lr - c_params[n]).abs().mean().item() < EPS
        for n in buffer_prev:
            # a simple check, make sure buffer is at least updated.
            # some buffers such as bn.num_batches_tracked has type torch.int,
            # do not support mean(), convert to float to check
            assert (buffer_prev[n] - c_buffers[n]).abs().float().mean().item() > 0

    # check hiddens not zero after these steps
    assert all(hid.abs().mean().item() > 1e-3 for hid in hiddens)

    # check parameters/buffers is back
    for n in c_params:
        assert (w_prev[n] - c_params[n]).abs().mean().item() < EPS
    for n in buffer_prev:
        assert (buffer_prev[n] - c_buffers[n]).abs().float().mean().item() < EPS

# ---- End test rnn_super_net ----

# ---- Test rnn_diff_super_net ----
def test_rnn_diff_supernet_forward(rnn_diff_super_net):
    if version.parse(torch.__version__).minor >= 7:
        pytest.xfail("FIXME: We currently do not fix this bug yet. When using torch>=1.7.0, "
                     "we encountered: Warning: Error detected in SplitBackward. "
                     "RuntimeError: one of the variables needed for gradient computation "
                     "has been modified by an inplace operation")
    from aw_nas.controller import DiffController

    time_steps = 5
    batch_size = 2
    _num_tokens = rnn_diff_super_net.num_tokens
    _num_hid = rnn_diff_super_net.num_hid
    _num_layers = rnn_diff_super_net._num_layers
    search_space = rnn_diff_super_net.search_space
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1)[0]
    cand_net = rnn_diff_super_net.assemble_candidate(rollout)

    # init hiddens
    hiddens = rnn_diff_super_net.init_hidden(batch_size)

    data = _rnn_data(time_steps, batch_size, _num_tokens)

    logits, _, outs, next_hiddens = cand_net.forward_data(data[0], mode="eval", hiddens=hiddens)
    assert tuple(logits.shape) == (time_steps, batch_size, _num_tokens)
    assert tuple(outs.shape) == (time_steps, batch_size, _num_hid)
    assert len(next_hiddens) == _num_layers
    # the value is equal to the calculated results, the hidden is modified in-place
    assert (hiddens == next_hiddens).all()

def test_rnn_diff_supernet_to_arch(rnn_diff_super_net):
    from aw_nas.controller import DiffController

    search_space = rnn_diff_super_net.search_space
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1)[0]
    cand_net = rnn_diff_super_net.assemble_candidate(rollout)

    time_steps = 5
    batch_size = 2
    _num_tokens = rnn_diff_super_net.num_tokens
    data = _rnn_data(time_steps, batch_size, _num_tokens)

    hiddens = rnn_diff_super_net.init_hidden(batch_size)

    # default detach_arch=True, no grad w.r.t the controller param
    results = cand_net.forward_data(data[0], hiddens=hiddens)
    loss = _rnn_criterion(data[0], results, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is None

    results = cand_net.forward_data(data[0], hiddens=hiddens, detach_arch=False)
    loss = _rnn_criterion(data[0], results, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is not None
# ---- End test rnn_diff_super_net ----
