"""
AW_NAS package.
"""
#pylint: disable=unused-import

from pkg_resources import resource_string

__version__ = resource_string(__name__, "VERSION").decode("ascii")
__version_info__ = __version__.split(".")

from aw_nas.utils import RegistryMeta

from aw_nas.base import Component

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

from aw_nas import dataset
from aw_nas import controller
from aw_nas import evaluator
from aw_nas import weights_manager
from aw_nas import objective
from aw_nas import trainer
from aw_nas import final
from aw_nas import hardware

from aw_nas import btcs
from aw_nas import germ

from .plugin import _reload_plugins, AwnasPlugin
_reload_plugins()
