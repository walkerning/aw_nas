from aw_nas.utils import getLogger

_LOGGER = getLogger("hardware.compiler")

try:
    from aw_nas.hardware.compiler import dpu
except ImportError as e:
    _LOGGER.warn("Cannot import hardware compiler for dpu: {}\n".format(e))
