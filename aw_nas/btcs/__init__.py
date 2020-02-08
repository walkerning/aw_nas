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

try:
    from aw_nas.btcs import nasbench_201
except ImportError as e:
    _LOGGER.warn(
        ("Error importing module nasbench_201: {}\n"
         "Should install the NASBench 201 package following "
         "https://github.com/D-X-Y/NAS-Bench-201").format(e))
        
