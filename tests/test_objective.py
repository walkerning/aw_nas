import numpy as np
import torch
import pytest

# ---- Test fault_injection ----
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
        "search_space_cfg": {
            "num_steps": 1,
            "num_layers": 1,
            "num_cell_groups": 1,
            "cell_layout": [0],
            "reduce_cell_groups": []
        }}], indirect=["super_net"])
def test_inject(super_net):
    cand_net = _supernet_sample_cand(super_net)

    data = _cnn_data()

    from aw_nas.objective.fault_injection import FaultInjector
    injector = FaultInjector(gaussian_std=None, mode="fixed")
    injector.set_random_inject(0.001)
    # forward stem
    cand_net.eval()
    def inject(state, context):
        if context.is_last_concat_op:
            return
        context.last_state = injector.inject(state)
    cand_net.forward_one_step_callback(data[0], callback=inject)

# ---- Test robustness ----
def test_generate_adv(super_net):
    cand_net = _supernet_sample_cand(super_net)
    data = _cnn_data()
    from aw_nas.objective.robustness import PgdAdvGenerator
    generator = PgdAdvGenerator(epsilon=0.03, n_step=5, step_size=0.0078, rand_init=False)
    inputs_adv = generator.generate_adv(data[0], None, data[1], cand_net)

    assert (inputs_adv - data[0] != 0).any()
    assert ((inputs_adv - data[0]).abs().max() <= 0.03 + 1e-5).all()
