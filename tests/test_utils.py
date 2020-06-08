import pytest

EPS = 1e-8

@pytest.mark.parametrize("case",
                         [{
                             "cfg": {
                                 "type": "mul",
                                 "boundary": [6, 11],
                                 "step": 0.1,
                                 "start": 1.
                             },
                             "epochs": [1, 5, 6, 12],
                             "results": [1, 1, 0.1, 0.01]
                         },
                          {
                              "cfg": {
                                  "type": "add",
                                  "every": 2,
                                  "step": -0.3,
                                  "start": 1.,
                                  "min": 0
                              },
                              "epochs": [1, 2, 3, 5, 9],
                              "results": [1, 1, 0.7, 0.4, 0]
                          },
                          {
                              "cfg": {
                                  "type": "value",
                                  "value": [None, 0.1, 1.0],
                                  "boundary": [1, 11, 21]
                              },
                              "epochs": [2, 11, 40],
                              "results": [None, 0.1, 1.0]
                          }])
def test_get_schedule_value(case):
    from aw_nas.utils.common_utils import get_schedule_value

    for epoch, expect in zip(case["epochs"], case["results"]):
        res = get_schedule_value(case["cfg"], epoch)
        if expect is None:
            assert res is None
        else:
            assert res - expect < EPS


def test_wrap_writer_none():
    from aw_nas.utils.vis_utils import WrapWriter
    writer = None
    sub_writer = WrapWriter(writer, "controller")
    assert sub_writer.add_scalar("loss", 3.0) is None


def test_wrap_writer(writer):
    from aw_nas.utils.vis_utils import WrapWriter
    sub_writer = WrapWriter(writer, "controller")
    subsub_writer = sub_writer.get_sub_writer("rl_agent")
    assert sub_writer.add_scalar.__doc__ == writer.add_scalar.__doc__

    assert sub_writer.add_scalar("loss", 3.0) == writer.add_scalar("controller/loss", 3.0)
    v = {"entropy": 1, "regularization": 0.1}
    assert subsub_writer.add_scalars("losses", v) == \
        writer.add_scalars("controller/rl_agent/losses", v)

def test_cache_results():
    import torch
    from aw_nas.utils.common_utils import cache_results
    class A(object):
        @cache_results(cache_params=["x", "y", "z"], key_funcs=id, buffer_size=1)
        def method(self, x, y, z=1):
            str_ = "{} {} {}".format(x, y, z)
            return str_
    a = A()
    x_1 = torch.Tensor([0.1, 0.2, 0.3])
    x_2 = torch.Tensor([0.1, 0.2, 0.3])
    a.method(x_1, 2) # called
    assert A.method.cache_hit_and_miss == [0, 1]
    a.method(x_1, 2) # use cached
    assert A.method.cache_hit_and_miss == [1, 1]
    a.method(x_2, 2) # called
    assert A.method.cache_hit_and_miss == [1, 2]
    assert len(A.method.cache_dict) == 1
    a.method(x_2, 2, 1) # use cached
    assert A.method.cache_hit_and_miss == [2, 2]
    a.method(x_1, 2) # called
    assert A.method.cache_hit_and_miss == [2, 3]

def test_tensor_scheduler():
    import torch
    from aw_nas.utils.torch_utils import init_tensor_scheduler
    tensor = torch.ones(1) * 0.1
    cfg = {
        "type": "CosineAnnealingLR",
        "T_max": 5,
        "eta_min": 0.
    }
    scheduler = init_tensor_scheduler(tensor, cfg)
    for i in range(8):
        scheduler.step(i)
        print(tensor.item())

def test_warmup_cosine_lrscheduler():
    import numpy as np
    import torch
    from torch import optim
    from aw_nas.utils.lr_scheduler import WarmupCosineAnnealingLR

    params = [torch.nn.Parameter(torch.zeros(2, 3))]
    optimizer = optim.SGD(params, 0.01)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, T_max=10)
    expected = list(np.arange(0, 0.01, 0.001))
    for i in range(20):
        scheduler.step(i)
        print(scheduler.get_lr())
        # assert scheduler.get_lr()[0] - expected[i] < 1e-6

    print("2")
    optimizer = optim.SGD(params, 0.01)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=10, every=10, T_max=10)
    for i in range(20):
        scheduler.step(i)
        print(scheduler.get_lr())

def test_registered_supported_rollouts():
    from aw_nas import RegistryMeta, BaseRollout

    simple_trainer_cls = RegistryMeta.get_class("trainer", "simple")
    assert "a_fake_rollout" not in simple_trainer_cls.supported_rollout_types()
    class AFakeRollout(BaseRollout):
        NAME = "a_fake_rollout"
        supported_components = [("trainer", "simple")]

        def set_candidate_net(self, c_net):
            self.c_net = c_net

        def plot_arch(self, *args, **kwargs):
            pass
    assert "a_fake_rollout" in simple_trainer_cls.supported_rollout_types()
