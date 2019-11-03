import pytest

import numpy as np
import torch

# ---- test mepa ----
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
    {
        "dropout_rate": 0,
        "search_space_cfg": {
            "num_steps": 1,
            "num_layers": 1,
            "num_cell_groups": 1,
            "cell_layout": [0],
            "reduce_cell_groups": [],
        }
    }], indirect=["super_net"])
def test_outplace_sgd(super_net):
    import copy
    from torch import nn
    from aw_nas.evaluator.mepa import LearnableLrOutPlaceSGD

    cand_net = _supernet_sample_cand(super_net)
    batch_size = 2
    data = _cnn_data(batch_size=batch_size)
    sur_opt_1 = torch.optim.SGD(super_net.parameters(), lr=0.1)
    named_params = dict(super_net.named_parameters())
    named_params_backup = {n: p.clone() for n, p in named_params.items()}
    sur_opt_2 = LearnableLrOutPlaceSGD(named_params, 0.1, "cuda", 1)

    # set train mode
    cand_net.train()
    # step using optimizer 1
    logits = cand_net(data[0].cuda())
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    loss.backward()
    sur_opt_1.step()
    new_params_opt_1 = {n: p.clone() for n, p in super_net.named_parameters()}

    # set back to original weights
    for n, p in cand_net.named_parameters():
        p.data.copy_(named_params_backup[n])
    # step using optimizer 2
    logits_2 = cand_net(data[0].cuda())
    loss_2 = nn.CrossEntropyLoss()(logits_2, data[1].cuda())
    assert (logits_2 == logits).all() and loss_2 == loss
    new_params_opt_2 = sur_opt_2.step(loss_2, named_params, 0, high_order=False, allow_unused=True)

    # outplace SGD will not change parameters in-place
    for n, p in cand_net.named_parameters():
        assert (p - named_params_backup[n]).abs().mean() < 1e-8

    # in-place and out-place SGD optimizer should have roughly the same effect
    for n, p in new_params_opt_2.items():
        assert (new_params_opt_1[n] - p).abs().mean() < 1e-8

# ---- end test mepa ----
