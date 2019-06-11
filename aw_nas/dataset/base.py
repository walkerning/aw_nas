# -*- coding: utf-8 -*-
"""Base class definition of Dataset"""

import os
import abc

from aw_nas import Component, utils

class BaseDataset(Component):
    REGISTRY = "dataset"

    def __init__(self, relative_dir=None):
        """
        Args:
            data_dir (str): The directory to store the datasets,
                by default: `data` directory under the current working directory.
        """
        super(BaseDataset, self).__init__(schedule_cfg=None)

        base_dir = os.environ.get("AWNAS_DATA", os.path.expanduser("~/awnas_data"))
        if relative_dir is None:
            relative_dir = self.NAME #pylint: disable=no-member
        self.data_dir = os.path.join(base_dir, relative_dir)

    @abc.abstractmethod
    def splits(self):
        """
        Returns:
           Dict(str: torch.utils.data.Dataset): A dict from split name to dataset.
        """
        return {}

    @utils.abstractclassmethod
    def data_type(cls):
        """
        The data type of this dataset.
        """
