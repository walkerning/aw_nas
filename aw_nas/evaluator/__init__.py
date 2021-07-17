"""
Evaluators: evaluator components is responsible for evaluating the performance of a candidate
network.
"""

#pylint: disable=unused-import, unused-wildcard-import, wildcard-import

from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.evaluator.mepa import MepaEvaluator
from aw_nas.evaluator.shared_weight import *
from aw_nas.evaluator.tune import TuneEvaluator
from aw_nas.evaluator.compare import ArchNetworkEvaluator
from aw_nas.evaluator.arch_network import *
from aw_nas.evaluator.few_shot_shared_weight import *
