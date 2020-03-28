from aw_nas.utils import logger as _logger
_LOGGER = _logger.getChild("hardware.compiler")

try:
    from aw_nas.hardware.compiler import dpu
except ImportError as e:
    _LOGGER.warn("Error importing hardware compiler for dpu: {}\n".format(e))

