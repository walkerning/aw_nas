#pylint: disable=unused-import,wrong-import-position

from aw_nas.utils import getLogger
_LOGGER = getLogger("final")

from .cnn_trainer import CNNFinalTrainer
from .cnn_model import CNNGenotypeModel

from .bnn_model import BNNGenotypeModel

from .rnn_trainer import RNNFinalTrainer
from .rnn_model import RNNGenotypeModel

from .dense import DenseGenotypeModel

from .ofa_model import OFAGenotypeModel

from .wrapper_model import WrapperFinalModel
from .wrapper_trainer import WrapperFinalTrainer

try:
    from .det_model import DetectionFinalModel
    from .det_trainer import DetectionFinalTrainer
except ImportError as e:
    _LOGGER.warn(
        ("Cannot import module detection: {}\n"
         "Should install EXTRAS_REQUIRE `det`").format(e))


from .general_model import GeneralGenotypeModel
