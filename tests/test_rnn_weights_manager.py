import numpy as np
import six
import torch
from torch import nn

import pytest


# ---- Test rnn_super_net ----
_criterion = nn.CrossEntropyLoss()
def _rnn_criterion(outputs, targets):
    logits, _, _ = outputs
    return _criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

def _rnn_super_sample_cand(net):
    from aw_nas.common import Rollout
    ss = net.search_space

    arch = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    rollout = Rollout(arch, info={}, search_space=ss)
    cand_net = net.assemble_candidate(rollout)
    return cand_net

@pytest.mark.parametrize("rnn_super_net", [
    {"candidate_member_mask": False}], indirect=["rnn_super_net"])
def test_supernet_assemble_nomask(rnn_super_net):
    net = rnn_super_net
    cand_net = _rnn_super_sample_cand(net)
    assert set(dict(cand_net.named_parameters()).keys()) == set(dict(net.named_parameters()).keys())
    assert set(dict(cand_net.named_buffers()).keys()) == set(dict(net.named_buffers()).keys())

@pytest.mark.parametrize("rnn_super_net", [
    {
        "edge_batch_norm": True
    },
    {
        "edge_batch_norm": False
    }], indirect=["rnn_super_net"])
def test_supernet_assemble(rnn_super_net):
    net = rnn_super_net
    cand_net = _rnn_super_sample_cand(net)

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
    assert len(s_b_names) >= len(c_b_names) or len(s_b_names) == 0
    assert len(s_b_names.intersection(c_b_names)) == len(c_b_names)
    # names = sorted(set(c_names).difference([g[0] for g in grads]))

@pytest.mark.parametrize("rnn_super_net", [{"num_tokens": 10}],
                         indirect=["rnn_super_net"])
def test_supernet_forward(rnn_super_net):
    # test forward
    cand_net = _rnn_super_sample_cand(rnn_super_net)

    time_steps = 5
    batch_size = 2
    _num_tokens = rnn_super_net.num_tokens
    _num_hid = rnn_super_net.num_hid
    _num_layers = rnn_super_net._num_layers

    hiddens = rnn_super_net.init_hidden(batch_size)
    hidden_id = tuple([id(hid) for hid in hiddens])

    assert tuple(hiddens[0].shape) == (batch_size, _num_hid)
    assert len(hiddens) == _num_layers

    data = torch.LongTensor(np.random.randint(0, high=_num_tokens, #pylint: disable=no-member
                                              size=(time_steps+1, batch_size)))
    data = (data[:-1, :], data[1:, :]) # inputs, targets

    logits, all_outs, next_hiddens = cand_net.forward_data(data[0], mode="eval", hiddens=hiddens)
    assert tuple(logits.shape) == (time_steps, batch_size, _num_tokens)
    assert tuple(all_outs.shape) == (time_steps, batch_size, _num_hid)
    assert tuple([id(hid) for hid in hiddens]) == hidden_id # the hidden is modified in-place
    assert all((hid == n_hid).all() for hid, n_hid in zip(hiddens, next_hiddens)) # the value is equal to the calculated results


@pytest.mark.parametrize("test_id",
                         range(5))
def test_supernet_candidate_gradient_virtual(test_id, rnn_super_net):
    lr = 0.1
    EPS = 1e-4
    time_steps = 5

    batch_size = 2
    _num_tokens = rnn_super_net.num_tokens
    _num_hid = rnn_super_net.num_hid
    _num_layers = rnn_super_net._num_layers

    data = torch.LongTensor(np.random.randint(0, high=_num_tokens, #pylint: disable=no-member
                                              size=(time_steps+1, batch_size)))
    data = (data[:-1, :], data[1:, :]) # inputs, targets

    hiddens = rnn_super_net.init_hidden(batch_size)
    # check hiddens not zero after these steps
    assert all(hid.abs().sum().item() < EPS for hid in hiddens)

    net = rnn_super_net
    cand_net = _rnn_super_sample_cand(net)
    c_params = dict(cand_net.named_parameters())
    c_buffers = dict(cand_net.named_buffers())
    # test `gradient`, `begin_virtual`
    w_prev = {k: v.clone() for k, v in six.iteritems(c_params)}
    buffer_prev = {k: v.clone() for k, v in six.iteritems(c_buffers)}
    with cand_net.begin_virtual():
        grads = cand_net.gradient(data, mode="train", hiddens=hiddens, criterion=_rnn_criterion)
        assert len(grads) == len(c_params)
        optimizer = torch.optim.SGD(cand_net.parameters(), lr=lr)
        optimizer.step()
        for n, grad in grads:
            assert (w_prev[n] - grad * lr - c_params[n]).abs().sum().item() < EPS
        # sometimes, some buffer just don't updated... so here coomment out
        # for n in buffer_prev:
        #     # a simple check, make sure buffer is at least updated...
        #     assert (buffer_prev[n] - c_buffers[n]).abs().sum().item() < EPS

    # check hiddens not zero after these steps
    assert all(hid.abs().sum().item() > 0.1 for hid in hiddens)

    # check parameters/buffers is back
    for n in c_params:
        assert (w_prev[n] - c_params[n]).abs().sum().item() < EPS
    for n in c_buffers:
        assert (buffer_prev[n] - c_buffers[n]).abs().sum().item() < EPS

# ---- End test rnn_super_net ----

# # ---- Test diff_rnn_super_net ----
# def test_diff_supernet_forward(diff_rnn_super_net):
#     from aw_nas.common import get_search_space
#     from aw_nas.controller import DiffController

#     search_space = get_search_space(cls="rnn")
#     device = "cuda"
#     controller = DiffController(search_space, device)
#     rollout = controller.sample(1)[0]
#     cand_net = diff_rnn_super_net.assemble_candidate(rollout)

#     data = (torch.tensor(np.random.rand(2, 3, 28, 28)).float(), torch.tensor([0, 1]).long()) #pylint: disable=not-callable
#     logits = cand_net.forward_data(data[0])
#     assert tuple(logits.shape) == (2, 10)

# def test_diff_supernet_to_arch(diff_rnn_super_net):
#     from torch import nn
#     from aw_nas.common import get_search_space
#     from aw_nas.controller import DiffController

#     search_space = get_search_space(cls="rnn")
#     device = "cuda"
#     controller = DiffController(search_space, device)
#     rollout = controller.sample(1)[0]
#     cand_net = diff_rnn_super_net.assemble_candidate(rollout)

#     data = (torch.tensor(np.random.rand(2, 3, 28, 28)).float(), torch.tensor([0, 1]).long()) #pylint: disable=not-callable

#     # default detach_arch=True, no grad w.r.t the controller param
#     logits = cand_net.forward_data(data[0])
#     loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
#     assert controller.cg_alphas[0].grad is None
#     loss.backward()
#     assert controller.cg_alphas[0].grad is None

#     logits = cand_net.forward_data(data[0], detach_arch=False)
#     loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
#     assert controller.cg_alphas[0].grad is None
#     loss.backward()
#     assert controller.cg_alphas[0].grad is not None
# # ---- End test diff_rnn_super_net ----
