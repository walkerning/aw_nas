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
