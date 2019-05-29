import pytest

class MockUnderlyWriter(object):
    def add_scalar(self, tag, value):
        """Add scalar!"""
        return "add_scalar_{}_{}".format(tag, value)

    def add_scalars(self, main_tag, value):
        """Add scarlas!"""
        return "add_scalars_{}_{}".format(main_tag, value)

@pytest.fixture
def writer():
    return MockUnderlyWriter()

@pytest.fixture
def super_net(request):
    cfg = getattr(request, "param", {})
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import SuperNet
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    net = SuperNet(search_space, device, **cfg)
    return net

@pytest.fixture
def diff_super_net(request):
    cfg = getattr(request, "param", {})
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import DiffSuperNet
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    net = DiffSuperNet(search_space, device, **cfg)
    return net

@pytest.fixture
def rnn_super_net(request):
    cfg = getattr(request, "param", {})
    num_tokens = cfg.pop("num_tokens", 10)
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import RNNSuperNet
    search_space = get_search_space(cls="rnn")
    device = "cuda"
    net = RNNSuperNet(search_space, device, num_tokens, **cfg)
    return net

@pytest.fixture
def rnn_diff_super_net(request):
    cfg = getattr(request, "param", {})
    num_tokens = cfg.pop("num_tokens", 10)
    from aw_nas.common import get_search_space
    from aw_nas.weights_manager import RNNDiffSuperNet
    search_space = get_search_space(cls="rnn")
    device = "cuda"
    net = RNNDiffSuperNet(search_space, device, num_tokens, **cfg)
    return net
