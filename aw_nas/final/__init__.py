#pylint: disable=unused-import
from aw_nas.utils import logger as _logger
_LOGGER = _logger.getChild("final")

from .cnn_trainer import CNNFinalTrainer
from .cnn_model import CNNGenotypeModel

from .bnn_model import BNNGenotypeModel

from .rnn_trainer import RNNFinalTrainer
from .rnn_model import RNNGenotypeModel

from .dense import DenseGenotypeModel

from .ofa_model import OFAGenotypeModel

try:
    from .ssd_model import SSDFinalModel, SSDHeadFinalModel
    from .det_trainer import DetectionFinalTrainer
except ImportError as e:
    _LOGGER.warn(
        ("Cannot import module detection: {}\n"
         "Should install EXTRAS_REQUIRE `det`").format(e))


from .general_model import GeneralGenotypeModel
