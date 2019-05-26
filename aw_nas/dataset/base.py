# -*- coding: utf-8 -*-
"""Base class definition of Dataset"""

import abc

from aw_nas import Component

class BaseDataset(Component):
    REGISTRY = "dataset"

    def __init__(self, data_dir="./data"):
        """
        Args:
            data_dir (str): The directory to store the datasets,
                by default: `data` directory under the current working directory.
        """
        super(BaseDataset, self).__init__(schedule_cfg=None)

        self.data_dir = data_dir

    @abc.abstractmethod
    def splits(self):
        """
        Returns:
           Dict(str: torch.utils.data.Dataset): A dict from split name to dataset.
        """
        return {}

    @abc.abstractmethod
    def data_type(self):
        return None
