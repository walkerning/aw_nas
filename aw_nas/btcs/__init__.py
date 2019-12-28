"""
Built-in tight coupled NAS flows.
"""

from aw_nas.utils import logger as _logger
_LOGGER = _logger.getChild("btc")

try:
    from aw_nas.btcs import nasbench
except ImportError as e:
    _LOGGER.warn(
        ("Error importing module nasbench: {}\n"
         "Should install the NASBench 101 package following "
         "https://github.com/google-research/nasbench").format(e))
