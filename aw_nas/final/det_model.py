from __future__ import print_function

import torch
from torch import nn

from aw_nas.utils.exception import ConfigException, expect

class HeadModel(nn.Module):
    def __init__(self,
                 device,
                 num_classes=10,
                 extras=None,
                 regression_headers=None,
                 classification_headers=None):
        super(HeadModel, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.extras = extras
        self.regression_headers = regression_headers
        self.classification_headers = classification_headers
        expect(
            None not in [extras, regression_headers, classification_headers],
            "Extras, regression_headers and classification_headers must be provided, "
            "got None instead.",
            ConfigException)

    def forward(self, features):
        expect(isinstance(features, (list, tuple)),
               'features must be a series of feature.', ValueError)
        features = self.extras(features)
        confidences = self.classification_headers(features)
        locations =  self.regression_headers(features)
        return confidences, locations
