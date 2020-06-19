
from __future__ import print_function


import torch

from torch import nn
from torchvision.ops import nms

from aw_nas.ops import ops
from aw_nas.utils import weights_init
from aw_nas.utils import box_utils

from aw_nas.utils.exception import ConfigException, expect


class HeadModel(nn.Module):
    
    def __init__(self, device, num_classes=10, extras=None, regression_headers=None, classification_headers=None, norm=None):
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
        expect(None not in [extras, regression_headers, classification_headers], 'Extras, regression_headers and classification_headers must be provided, got None instead.', ConfigException)

        self._init_weights()

    def forward(self, features):
        expect(isinstance(features, (list, tuple)), 'features must be a series of feature.', ValueError)
        if self.norm:
            features = self.norm(features)
        features = self.extras(features)

        batch_size = features[0].shape[0]

        confidences = [feat.permute(0, 2, 3, 1).contiguous() for feat in self.classification_headers(features)]
        locations = [feat.permute(0, 2, 3, 1).contiguous() for feat in self.regression_headers(features)]

        confidences = torch.cat([t.view(t.size(0), -1) for t in confidences], 1).view(batch_size, -1, self.num_classes)
        locations = torch.cat([t.view(t.size(0), -1) for t in locations], 1).view(batch_size, -1, 4)
        return confidences, locations

    def _init_weights(self):
        self.extras.apply(weights_init)
        self.regression_headers.apply(weights_init)
        self.classification_headers.apply(weights_init)


class PredictModel(nn.Module):
    def __init__(self, num_classes, background_label, top_k=200, confidence_thresh=0.01, nms_thresh=0.5, variance=(0.1, 0.2), priors=None):
        super(PredictModel, self).__init__()
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.priors = priors
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, confidences, locations, img_shape):
        priors = self.priors(img_shape).to(confidences.device)
        num = confidences.size(0)  # batch size
        num_priors = priors.size(0)
        output = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(num)]
        confidences = self.softmax(confidences)
        conf_preds = confidences.view(num, num_priors,
                                      self.num_classes).transpose(2, 1)

        for i in range(num):
            decoded_boxes = box_utils.decode(locations[i],
                                             priors, self.variance)
            conf_scores = conf_preds[i].clone()

            for cls_idx in range(1, self.num_classes):
                c_mask = conf_scores[cls_idx].gt(self.confidence_thresh)
                scores = conf_scores[cls_idx][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                box_prob = torch.cat([boxes, scores.view(-1, 1)], 1)
                ids = nms(boxes, scores, self.nms_thresh)
                output[i][cls_idx] = torch.cat((boxes[ids], scores[ids].unsqueeze(1)), 1)
        return output

