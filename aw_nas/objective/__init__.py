"""
Objectives
"""
#pylint: disable=unused-import
from aw_nas.objective.base import BaseObjective
from aw_nas.objective.image import ClassificationObjective
from aw_nas.objective.flops import FlopsObjective
from aw_nas.objective.language import LanguageObjective
from aw_nas.objective.fault_injection import FaultInjectionObjective
from aw_nas.objective.ofa import OFAClassificationObjective
from aw_nas.objective.ssd import SSDObjective
from aw_nas.objective.hardware import HardwareObjective

from aw_nas.objective.container import ContainerObjective
