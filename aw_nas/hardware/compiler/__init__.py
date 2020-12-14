from aw_nas.utils import getLogger

_LOGGER = getLogger("hardware.compiler")

try:
    from aw_nas.hardware.compiler import dpu
except ImportError as e:
    _LOGGER.warn("Cannot import hardware compiler for dpu: {}\n".format(e))

try:
    from aw_nas.hardware.compiler import xavier
except ImportError as e:
    _LOGGER.warn("Cannot import hardware compiler for xavier: {}\n".format(e))


