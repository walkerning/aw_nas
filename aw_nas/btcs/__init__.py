"""
Built-in tight coupled NAS flows.
"""

from aw_nas.utils import getLogger
_LOGGER = getLogger("btc")

try:
    from aw_nas.btcs import nasbench_101
except ImportError as e:
    _LOGGER.warning(
        ("Cannot import module nasbench: {}\n"
         "Should install the NASBench 101 package following "
         "https://github.com/google-research/nasbench").format(e))

try:
    from aw_nas.btcs import nasbench_201
    from aw_nas.btcs import nasbench_201_close
except ImportError as e:
    _LOGGER.warning(
        ("Cannot import module nasbench_201: {}\n"
         "Should install the NASBench 201 package following "
         "https://github.com/D-X-Y/NAS-Bench-201").format(e))

try:
    from aw_nas.btcs import nasbench_301
except ImportError as e:
    _LOGGER.warning(
        ("Cannot import module nasbench_301: {}\n"
         "Should install the NASBench 301 package following "
         "https://github.com/automl/nasbench301.\n"
         "There still exist some bugs in commit 48a5f0ca152b83ae2fa31365116c0fb480466fb1, "
         "by the time of 2020/12/29, if these bugs are not fixed, can temporarily install this: "
         "pip install git+https://github.com/walkerning/nasbench301.git"
         "@65420f5595c0fdfab99fe3e914b04bffebf8fffe").format(e))

from aw_nas.btcs import enas
from aw_nas.btcs import layer2
