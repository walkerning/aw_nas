"""
AW_NAS package.
"""
#pylint: disable=unused-import,unused-wildcard-import,wildcard-import

from pkg_resources import resource_string
__version__ = resource_string(__name__, "VERSION").decode("ascii")

from .common import (
    SearchSpace,
    Rollout,
    CNNSearchSpace,
    RNNSearchSpace,
    get_search_space,
)

from .utils import RegistryMeta

from .base import Component

from .dataset import *

from .trainer import *

from .controller import *

from .weights_manager import *
