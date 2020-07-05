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
                 classification_headers=None,
                 norm=None):
        super(HeadModel, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.extras = extras
        self.regression_headers = regression_headers
        self.classification_headers = classification_headers
        if norm:
            self.norm = norm
        else:
            self.norm = None
        expect(
            None not in [extras, regression_headers, classification_headers],
            "Extras, regression_headers and classification_headers must be provided, "
            "got None instead.",
            ConfigException)

    def forward(self, features):
        expect(isinstance(features, (list, tuple)),
               'features must be a series of feature.', ValueError)
        if self.norm:
            features = self.norm(features)
        features = self.extras(features)

        batch_size = features[0].shape[0]

        confidences = [
            feat.permute(0, 2, 3, 1).contiguous()
            for feat in self.classification_headers(features)
        ]
        locations = [
            feat.permute(0, 2, 3, 1).contiguous()
            for feat in self.regression_headers(features)
        ]

        confidences = torch.cat([t.view(t.size(0), -1) for t in confidences],
                                1).view(batch_size, -1, self.num_classes)
        locations = torch.cat([t.view(t.size(0), -1) for t in locations],
                              1).view(batch_size, -1, 4)
        return confidences, locations
