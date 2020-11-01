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

        base_dir = utils.get_awnas_dir("AWNAS_DATA", "data")
        if relative_dir is None:
            relative_dir = self.NAME #pylint: disable=no-member
        self.data_dir = os.path.join(base_dir, relative_dir)

    def same_data_split_mapping(self):
        """
        If different transforms are used for the same split of data.
        The resulting Dataset will be saved as multiple keys.
        Use `same_data_split_mapping` to map the equivalent relation.

        For example, `return {"train_testTransform": "train"}` means
        "train_testTransform" split use the same data as "train",
        and they should use the same set of randomly sampled indices.

        The default implementation is a commonly-used mapping.
        """
        return {"train_testTransform": "train"}

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
