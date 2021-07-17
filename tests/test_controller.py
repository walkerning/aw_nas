from __future__ import print_function

import os
import pytest
import numpy as np
from aw_nas.common import get_search_space


@pytest.mark.parametrize("case", [
    {"search_space_type": "cnn",
     "search_space_cfg": {},
     "controller_type": "evo",
     "controller_cfg": {"rollout_type": "discrete"}},
    {"search_space_type": "cnn",
     "search_space_cfg": {},
     "controller_type": "pareto-evo",
     "controller_cfg": {"rollout_type": "discrete"}},
    {"search_space_type": "cnn",
     "search_space_cfg": {},
     "controller_type": "rl"},
    # {"search_space_type": "cnn",
    #  "search_space_cfg": {},
    #  "controller_type": "differentiable"},
    {"search_space_type": "cnn",
     "search_space_cfg": {},
     "controller_type": "predictor-based",
     "controller_cfg": {
         "rollout_type": "discrete",
         "inner_controller_type": "evo",
         "inner_controller_cfg": {"rollout_type": "discrete"},
         "arch_network_type": "pointwise_comparator",
         "arch_network_cfg": {},
         "begin_train_num": 100 # do not train
     }}
])
def test_controllers_save(case, tmp_path):
    import io
    import torch
    from torch import nn

    from aw_nas.controller.base import BaseController
    ss = get_search_space(case["search_space_type"], **case["search_space_cfg"])
    controller = BaseController.get_class_(case["controller_type"])(
        ss, "cuda", **case.get("controller_cfg", {}))
    controller.set_mode("train")
    controller.on_epoch_start(1)
    rollouts = controller.sample(n=3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]

    if isinstance(controller, nn.Module):
        optimizer = torch.optim.SGD(controller.parameters(), lr=0.1)
    else:
        optimizer = None
    controller.step(rollouts, optimizer=optimizer, perf_name="reward")

    # try calling save/load (used when calling `trainer.final_save()`)
    buffer_ = io.BytesIO()
    controller.save(os.path.join(tmp_path, "cont"))
    buffer_.seek(0)
    controller.load(os.path.join(tmp_path, "cont"))

    # try calling torch.save (used when calling `trainer.final_save()`)
    buffer_ = io.BytesIO()
    torch.save(controller, buffer_)
    buffer_.seek(0)
    torch.load(buffer_)

# ---- test controller rl ----
def test_rl_controller():
    import torch
    import numpy as np

    from aw_nas.controller import RLController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = RLController(search_space, device)
    controller_i = RLController(search_space,
                                device,
                                independent_cell_group=True)
                                # rl_agent_cfg={"batch_update": False})
    assert len(list(controller.parameters())) == 10
    assert len(list(controller_i.parameters())) == 20
    rollouts = controller.sample(3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]
    optimizer = torch.optim.SGD(controller.parameters(), lr=0.01)
    loss = controller.step(rollouts, optimizer, "reward")

    rollouts = controller_i.sample(3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]
    optimizer = torch.optim.SGD(controller_i.parameters(), lr=0.01)
    loss = controller_i.step(rollouts, optimizer, "reward")


@pytest.mark.parametrize("case", [{
    "type": "anchor_lstm"
}, {
    "type": "embed_lstm"
}])
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


@pytest.mark.parametrize("case", [{
    "type": "anchor_lstm"
}, {
    "type": "embed_lstm"
}])
def test_controller_network_cellwise_primitives(case):
    import numpy as np
    from aw_nas.controller.rl_networks import BaseRLControllerNet
    search_space = get_search_space(cls="cnn",
                                    num_cell_groups=2,
                                    cell_shared_primitives=[
                                        ("none", "avg_pool_3x3",
                                         "max_pool_3x3", "skip_connect"),
                                        ("skip_connect", "avg_pool_3x3",
                                         "dil_conv_3x3")
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
    arch, log_probs, entropies, (hx, cx) = net_shared.sample(batch_size,
                                                             cell_index=0)
    assert all(np.max(single_arch[1]) <= 3 for single_arch in arch)
    arch, log_probs, entropies, (hx, cx) = net_shared.sample(batch_size,
                                                             cell_index=1)
    assert all(np.max(single_arch[1]) <= 2 for single_arch in arch)


@pytest.mark.parametrize("case", [{
    "type": "anchor_lstm"
}, {
    "type": "embed_lstm"
}])
def test_controller_network_cellwise_num_steps(case):
    import numpy as np
    from aw_nas.controller.rl_networks import BaseRLControllerNet
    from aw_nas.utils.exception import NasException
    search_space = get_search_space(cls="cnn",
                                    num_cell_groups=2,
                                    num_steps=[3, 6])

    device = "cuda"
    cls = BaseRLControllerNet.get_class_(case["type"])
    net0 = cls(search_space, device, cell_index=0)
    net1 = cls(search_space, device, cell_index=1)

    batch_size = 3
    arch, log_probs, entropies, (hx, cx) = net0.sample(batch_size)
    assert len(hx) == net0.num_lstm_layers
    assert len(cx) == net0.num_lstm_layers
    assert hx[0].shape == (batch_size, net0.controller_hid)
    assert len(arch) == batch_size
    num_actions = len(arch[0][0]) + len(arch[0][1])
    assert log_probs.shape == (batch_size, num_actions)
    assert entropies.shape == (batch_size, num_actions)
    assert len(arch[0][0]) == search_space.num_node_inputs * 3

    batch_size = 3
    arch, log_probs, entropies, (hx, cx) = net1.sample(batch_size)
    assert len(arch[0][0]) == search_space.num_node_inputs * 6

    with pytest.raises(NasException):
        # cannot use shared network when search space have cellwise `num_steps`
        _ = cls(search_space, device, cell_index=None)


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
    agent.step(rollouts, optimizer, "reward")
    for n, v in controller.named_parameters():
        assert (ori_params[n] - v).abs().mean() > 0


# --- test controller differentiable ----
def test_diff_controller():
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)

    assert controller.cg_alphas[0].shape == (
        14, len(search_space.shared_primitives))
    rollouts = controller.sample(3)
    assert isinstance(rollouts[0].genotype, search_space.genotype_type)


def test_diff_controller_use_prob():
    from aw_nas import utils
    import numpy as np
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, use_prob=True)

    assert controller.cg_alphas[0].shape == (
        14, len(search_space.shared_primitives))
    rollouts = controller.sample(3)
    assert np.abs((utils.get_numpy(rollouts[0].sampled[0]) - utils.softmax(rollouts[0].logits[0])))\
             .mean() < 1e-6
    assert isinstance(rollouts[0].genotype, search_space.genotype_type)


def test_diff_controller_force_uniform():
    import numpy as np
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space,
                                device,
                                force_uniform=True,
                                use_prob=True)

    rollouts = controller.sample(1)
    assert np.equal(rollouts[0].sampled[0].data, 1./len(search_space.shared_primitives) * \
                    np.ones((14, len(search_space.shared_primitives)))).all()


def test_diff_controller_cellwise_primitives():
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn",
                                    num_cell_groups=2,
                                    cell_shared_primitives=[
                                        ("none", "avg_pool_3x3",
                                         "max_pool_3x3", "skip_connect"),
                                        ("skip_connect", "avg_pool_3x3",
                                         "dil_conv_3x3")
                                    ])
    device = "cuda"
    controller = DiffController(search_space, device)
    assert controller.cg_alphas[0].shape == (14, 4)
    assert controller.cg_alphas[1].shape == (14, 3)
    rollout = controller.sample(1)[0]
    assert isinstance(rollout.genotype, search_space.genotype_type)
    assert set([conn[0] for conn in rollout.genotype.normal_0]).issubset(
        ["none", "avg_pool_3x3", "max_pool_3x3", "skip_connect"])
    assert set([conn[0] for conn in rollout.genotype.reduce_1
                ]).issubset(["avg_pool_3x3", "dil_conv_3x3", "skip_connect"])
    print(rollout.genotype)


def test_diff_controller_cellwise_num_steps():
    from aw_nas.controller import DiffController

    num_steps = [4, 6]
    num_cell_groups = len(num_steps)
    search_space = get_search_space(cls="cnn",
                                    num_cell_groups=num_cell_groups,
                                    num_steps=num_steps)
    device = "cuda"
    controller = DiffController(search_space, device)
    for i, num_step in enumerate(num_steps):
        assert controller.cg_alphas[i].shape[0] == \
            num_step * (num_step - 1) / 2 + search_space.num_init_nodes * num_step

    rollout = controller.sample(1)[0]
    assert isinstance(rollout.genotype, search_space.genotype_type)
    print(rollout.genotype)


def test_diff_controller_rollout_batch_size():
    import numpy as np
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)

    rollout = controller.sample(1, batch_size=4)[0]
    assert rollout.sampled[0].shape == (14, 4,
                                        len(search_space.shared_primitives))
    assert rollout.logits[0].shape == (14, len(search_space.shared_primitives))
    print(rollout.genotype)


# ---- test controller evo ----
def test_population_controller_avoid_repeat():
    from aw_nas.controller import EvoController
    ss_cfgs = {
        "cell_layout": [0, 1, 0, 1, 0],
        "num_layers": 5,
        "shared_primitives": ["skip_connect", "sep_conv_3x3"],
        "num_init_nodes": 1,
        "num_steps": 3,
        "num_cell_groups": 2,
        "reduce_cell_groups": [1],
    }
    ss = get_search_space("cnn", **ss_cfgs)
    controller = EvoController(ss, "cuda", rollout_type="discrete",
                               avoid_mutate_repeat=True, avoid_mutate_repeat_worst_threshold=3,
                               avoid_repeat_fallback="raise",
                               population_size=100, parent_pool_size=1)
    rollout = controller.sample(n=1)[0]
    rollout.set_perf(1.0)
    controller.set_mode("train")
    # make it the highest reward
    rollouts = []
    for i in range(99):
        rollouts.append(ss.mutate(rollout).set_perf(0.3))
    controller.step([rollout] + rollouts)
    controller.population_size = len(controller.population)
    with pytest.raises(Exception):
        for _ in range(3):
            controller.sample(n=1)

    controller.avoid_repeat_fallback = "return" # let's fallback to return, not rais
    for _ in range(3):
        controller.sample(n=1)


def test_population_controller_mutate():
    from aw_nas.controller import EvoController, ParetoEvoController

    device = "cuda"

    for ss, rollout_type in zip(("cnn", "ofa"), ("discrete", "ofa")):
        search_space = get_search_space(cls=ss)
        controller = EvoController(search_space,
                                   device,
                                   rollout_type=rollout_type,
                                   mode="train",
                                   population_size=1,
                                   parent_pool_size=1,
                                   mutate_kwargs={},
                                   avoid_mutate_repeat=True,
                                   avoid_mutate_repeat_worst_threshold=1,
                                   eval_sample_strategy="population",
                                   elimination_strategy="regularized")

        controller.mode = "train"
        rollouts = controller.sample(1)
        for rollout in rollouts:
            rollout.set_perf(np.random.random(), "reward")
        controller.step(rollouts)
        new_rollout = controller.sample(1)[0]
        assert str(rollouts[0].genotype) != str(new_rollout.genotype)

        controller = ParetoEvoController(search_space,
                                         device,
                                         rollout_type=rollout_type,
                                         mode="train",
                                         init_population_size=1,
                                         perf_names=["reward"],
                                         eval_sample_strategy="all")

        controller.mode = "train"
        rollouts = controller.sample(1)
        for rollout in rollouts:
            rollout.set_perf(np.random.random(), "reward")
        controller.step(rollouts)
        new_rollout = controller.sample(1)[0]
        assert str(rollouts[0].genotype) != str(new_rollout.genotype)

def test_pareto_evo_controller_remove_non_pareto():
    from aw_nas.controller import ParetoEvoController
    device = "cuda"
    for ss, rollout_type in zip(("cnn", "ofa"), ("discrete", "ofa")):
        search_space = get_search_space(cls=ss)
        controller = ParetoEvoController(search_space,
                                         device,
                                         rollout_type=rollout_type,
                                         mode="train",
                                         init_population_size=10,
                                         perf_names=["reward"],
                                         eval_sample_strategy="all")
        rollouts = controller.sample(15)
        for rollout in rollouts:
            rollout.set_perf(np.random.random(), "reward")
        controller.step(rollouts)
        assert 0 < len(controller.population)

@pytest.mark.skip(reason="Not necessary")
def test_pareto_evo_controller_find_opt():
    import numpy as np

    from aw_nas.controller import ParetoEvoController
    from aw_nas.common import SearchSpace
    from aw_nas.rollout import Rollout

    device = "cuda"

    class ToySearchSpace(SearchSpace):
        NAME = "toy"

        def __init__(self, dimension, choice, schedule_cfg=None):
            super(ToySearchSpace, self).__init__(schedule_cfg)

            self.dimension = dimension
            self.choice = choice

        def random_sample(self):
            return ToyRollout(ToyRollout.random_sample_arch(self.dimension,
                self.choice), {}, self)

        def genotype(self, arch):
            return tuple(arch)

        def rollout_from_genotype(self, genotype):
            return ToyRollout(list(genotype), {}, self)

        def plot_arch(self, genotypes, filename, label, **kwargs):
            pass

        def distance(self, arch1, arch2):
            pass

        @classmethod
        def supported_rollout_types(cls):
            return ["toy"]

        def mutate(self, rollout):
            arch = rollout.arch
            mutate_index = np.random.choice(list(range(len(arch))))
            mutate_res = np.random.choice(self.choice)
            while mutate_res == arch[mutate_index]:
                mutate_res = np.random.choice(self.choice)
            arch[mutate_index] = mutate_res
            return ToyRollout(arch, {}, self)


    class ToyRollout(Rollout):
        NAME = "toy"

        def __init__(self, arch, info, search_space, candidate_net=None):
            super(ToyRollout, self).__init__(arch, info, search_space,
                    candidate_net)
            self._genotype = tuple(arch)

        @classmethod
        def random_sample_arch(cls, dimension, choice):
            return np.random.choice(choice, size=dimension)

    dimension = 50
    range_ = 10
    search_space = ToySearchSpace(dimension, np.arange(-range_, range_, 0.1))
    
    def toy_func(x):
        fx = (x * np.sin(np.sqrt(2 * np.pi * np.abs(x)))).mean(-1)
        fy = np.sqrt(range_ ** 2 - fx ** 2) - x.mean(-1) / 10
        return fx, fy

    def evaluate_rollout(rollout):
        x = np.array(rollout.arch).reshape(1, -1)
        fx, fy = toy_func(x)
        fx = float(fx[0])
        fy = float(fy[0])
        rollout.set_perf(fx, "reward")
        rollout.set_perf(fy, "test")
        return rollout

    x = np.random.choice(np.arange(-range_, range_, 0.1), size=(100000, dimension))
    fx, fy = toy_func(x)
    indices = np.argsort(-fx)
    fx = fx[indices]
    fy = fy[indices]

    nx = [fx[0]]
    ny = [fy[0]]
    for xo, yo in zip(fx[1:], fy[1:]):
        if yo > ny[-1]:
            nx += [xo]
            ny += [yo]


    paretos = {}
    for epochs in [100, 200]:#, 300, 500, 800]:
        controller = ParetoEvoController(search_space,
                                     device,
                                     rollout_type="toy",
                                     init_population_size=100,
                                     perf_names=["reward", "test"],
                                     eval_sample_strategy="all")
        controller.mode = "train"

        for _ in range(epochs):
            rollouts = controller.sample(10)
            for rollout in rollouts:
                evaluate_rollout(rollout)
            controller.step(rollouts)

        population = controller.find_pareto_opt()
        pareto = [tuple(x) for x in population.values()]
        pareto = sorted(pareto)
        paretos[epochs] = pareto

    with open("pareto.json", "w") as fw:
        import json
        json.dump({"fx":np.array(fx).tolist(), "fy": np.array(fy).tolist(),
            "nx": nx, "ny": ny, "paretos": paretos}, fw)


@pytest.mark.parametrize("case", [
    {"perf_names": ["reward", "param_size"]},
    {"perf_names": ["reward"]}
])
def test_cars(case):
    from aw_nas.controller.base import BaseController
    ss = get_search_space("cnn")
    kwargs = {
            "population_size": 10,
            "eval_sample_strategy": "n",
            "prefill_population": True,
            "avoid_repeat": True
    }
    kwargs.update(case)
    controller = BaseController.get_class_("cars")(
        ss, "cuda", rollout_type="discrete", **kwargs)
    # eval mode sample, use to train supernet
    controller.set_mode("eval")
    rollouts = controller.sample(n=3)
    controller.set_mode("train")
    with pytest.raises(AssertionError):
        controller.sample(n=3)
    rollouts = controller.sample(n=20)
    random_perfs = np.random.rand(20)
    random_param_sizes = np.random.rand(20) * 10
    [rollout.set_perfs({
        "reward": perf,
        "param_size": param_size
    }) for rollout, perf, param_size in zip(rollouts, random_perfs, random_param_sizes)]
    controller.step(rollouts)
    dominated = set([rollout.genotype for rollout in rollouts]).difference(
        set(controller.population.keys()))
    rollout_perfs = {rollout.genotype: rollout.perf for rollout in rollouts}
    print("Population perfs: ", list(controller.population.values()))
    print("Dominated: ", [(rollout_perfs[geno]["reward"],
                           rollout_perfs[geno]["param_size"]) for geno in dominated])
