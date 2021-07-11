"""
Objectives
"""
#pylint: disable=unused-import

from aw_nas.utils import getLogger as _getLogger

from aw_nas.objective.base import BaseObjective
from aw_nas.objective.image import ClassificationObjective, CrossEntropyLabelSmooth
from aw_nas.objective.flops import FlopsObjective
from aw_nas.objective.language import LanguageObjective
from aw_nas.objective.fault_injection import FaultInjectionObjective
from aw_nas.objective.ofa import OFAClassificationObjective
from aw_nas.objective.detection import DetectionObjective
from aw_nas.objective.hardware import HardwareObjective
from aw_nas.objective.zeroshot import ZeroShot
from aw_nas.objective.container import ContainerObjective

try:
    from aw_nas.objective.mmdet_head import MMDetectionHeadObjective
except ImportError as e:
    _getLogger("mmdet_head").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e))

from aw_nas.objective.detection_utils import *
