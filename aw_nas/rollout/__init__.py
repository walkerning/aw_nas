"""
Rollouts, the inferface of different components in the NAS system.
"""

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

def assert_rollout_type(type_name):
    expect(type_name in BaseRollout.all_classes_(),
           "rollout type {} not registered yet".format(type_name))
    return type_name
