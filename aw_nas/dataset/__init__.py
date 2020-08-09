#pylint: disable=unused-import
from aw_nas.utils import logger as _logger
_LOGGER = _logger.getChild("dataset")

from aw_nas.dataset.base import BaseDataset
from aw_nas.dataset import cifar10
from aw_nas.dataset import ptb
from aw_nas.dataset import imagenet
from aw_nas.dataset import tiny_imagenet
from aw_nas.dataset import cifar100
from aw_nas.dataset import svhn

try:
    from aw_nas.dataset import voc
    from aw_nas.dataset import coco
except ImportError as e:
    _LOGGER.warn(
        ("Cannot import module detection: {}\n"
         "Should install EXTRAS_REQUIRE `det`").format(e))


AVAIL_DATA_TYPES = ["image", "sequence"]
