"""
Rollouts, the inferface of different components in the NAS system.
"""
# pylint: disable=unused-import

from aw_nas.utils.exception import expect
from aw_nas.rollout.base import (
    BaseRollout,
    Rollout,
    DifferentiableRollout
)
from aw_nas.rollout.mutation import (
    MutationRollout,
    CellMutation,
    Population,
    ModelRecord
)
from aw_nas.rollout.dense import (
    DenseMutationRollout,
    DenseMutation,
    DenseDiscreteRollout
)
from aw_nas.rollout.ofa import (
    MNasNetOFASearchSpace,
    MNasNetOFARollout,
    SSDOFASearchSpace,
    SSDOFARollout
)

from aw_nas.rollout.compare import (
    CompareRollout
)

from aw_nas.rollout.general import (
    GeneralSearchSpace,
    GeneralRollout
)

from aw_nas.rollout.wrapper import (
    WrapperSearchSpace,
    WrapperRollout,
    GermWrapperSearchSpace
)

def assert_rollout_type(type_name):
    expect(type_name in BaseRollout.all_classes_(),
           "rollout type {} not registered yet".format(type_name))
    return type_name
