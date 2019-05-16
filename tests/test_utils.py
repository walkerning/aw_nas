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
