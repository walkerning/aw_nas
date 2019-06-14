"""
AW_NAS package.
"""
#pylint: disable=unused-import,unused-wildcard-import,wildcard-import

from pkg_resources import resource_string
__version__ = resource_string(__name__, "VERSION").decode("ascii")

from .utils import RegistryMeta

from .base import Component

from .common import (
    assert_rollout_type,
    SearchSpace,
    BaseRollout,
    Rollout,
    DifferentiableRollout,
    CNNSearchSpace,
    RNNSearchSpace,
    get_search_space,
)

from .dataset import *

from .trainer import *

from .controller import *

from .evaluator import *

from .weights_manager import *

from .objective import *

from . import final
