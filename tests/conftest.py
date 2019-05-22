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
