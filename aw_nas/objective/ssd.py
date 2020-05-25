# -*- coding: utf-8 -*-

from math import sqrt
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torchvision
try:
    from torchvision.ops import nms
except ModuleNotFoundError:
    from aw_nas.utils import log as _log
    _log.logger.getChild("ssd").warn(
        "Detection task functionalities cannot be used, update torchvision version to >=0.4.0. "
        "current: %s", str(torchvision.__version__))

from aw_nas.objective.base import BaseObjective
from aw_nas.utils.torch_utils import accuracy
from aw_nas.utils import box_utils


class SSDObjective(BaseObjective):
    NAME = "ssd_detection"

    def __init__(self,
                 search_space,
                 num_classes=21,
                 min_dim=300,
                 feature_maps=[19, 10, 5, 3, 2, 1],
                 aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                 steps=[16, 32, 64, 100, 150, 300],
                 scales=[45, 90, 135, 180, 225, 270, 315],
                 clip=True,
                 center_variance=0.1,
                 size_variance=0.2,
                 nms_threshold=0.5):
        super(SSDObjective, self).__init__(search_space)
        self.num_classes = num_classes

        self.priors = PriorBox(min_dim, aspect_ratios, feature_maps, scales,
                               steps, (center_variance, size_variance), clip)
        self.target_transform = TargetTransform(
            0.5, (center_variance, size_variance))
        self.box_loss = MultiBoxLoss(num_classes, nms_threshold, True, 0, True,
                                     3, 1 - nms_threshold, False)
        self.predictor = PredictModel(num_classes,
                                      0,
                                      200,
                                      0.01,
                                      nms_threshold,
                                      priors=self.priors)

        self.all_boxes = [{} for _ in range(self.num_classes)]
        self.cache = []

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def on_epoch_start(self, epoch):
        super(SSDObjective, self).on_epoch_start(epoch)
        self.search_space.on_epoch_start(epoch)

    def batch_transform(self, inputs, outputs, annotations):
        """
        annotations: [-1, 4 + 1 + 1 + 2] boxes + labels + ids + shapes
        """
        for _id, cache in self.cache:
            if _id == hash(str(annotations)):
                return cache
        device = inputs.device
        img_shape = inputs.shape[-1]
        batch_size = inputs.shape[0]
        priors = self.priors.forward(img_shape).to(device)
        num_priors = priors.shape[0]
        location_t = torch.zeros([batch_size, num_priors, 4])
        classification_t = torch.zeros([batch_size, num_priors])

        shapes = []
        for i, (boxes, labels, _id, height, width) in enumerate(annotations):
            boxes = boxes / inputs.shape[-1]
            boxes = boxes.to(device)
            labels = labels.to(device)
            conf_t, loc_t = self.target_transform(boxes, labels, priors)
            location_t[i] = loc_t
            classification_t[i] = conf_t
            shapes.append([_id, height, width])

        cache = classification_t.long().to(device), location_t.to(
            device), shapes
        if len(self.cache) >= 1:
            del self.cache[-1]
        self.cache.append((hash(str(annotations)), cache))
        return cache

    def perf_names(self):
        return ["mAP"]

    def get_acc(self, inputs, outputs, targets, cand_net):
        conf_t, loc_t, shapes = self.batch_transform(inputs, outputs, targets)
        """
        target: [batch_size, anchor_num, 5], boxes + labels
        """
        keep = conf_t > 0
        confidences, regressions = outputs
        return accuracy(confidences[keep], conf_t[keep], topk=(1, 5))

    def get_mAP(self, inputs, outputs, annotations, cand_net):
        """
        Get mAP.
        """
        # FIXME: this method does not actually calculate mAP, but detection boxes
        # of current batch only. After all batch's boxes are calculated, passing them
        # to method `evaluate_detections`
        confidences, regression = outputs
        detections = self.predictor(confidences, regression, inputs.shape[-1])
        for batch_id, (_, _, _id, h, w) in enumerate(annotations):
            for j in range(self.num_classes):
                dets = detections[batch_id][j]
                if len(dets) == 0:
                    continue
                boxes = dets[:, :4]
                boxes[:, 0::2] *= w
                boxes[:, 1::2] *= h
                scores = dets[:, 4].reshape(-1, 1)
                cls_dets = torch.cat((boxes, scores),
                                     dim=1).cpu().detach().numpy()
                self.all_boxes[j][_id] = cls_dets
        return [0.]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        acc = self.get_acc(inputs, outputs, targets, cand_net)
        if not cand_net.training:
            self.get_mAP(inputs, outputs, targets, cand_net)
        return [acc[0].item()]

    def get_reward(self, inputs, outputs, targets, cand_net, final=False):
        return 0.

    def get_loss(self,
                 inputs,
                 outputs,
                 targets,
                 cand_net,
                 add_controller_regularization=True,
                 add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        return sum(self._criterion(inputs, outputs, targets, cand_net))

    def _criterion(self, inputs, outputs, annotations, model):
        conf_t, loc_t, shapes = self.batch_transform(inputs, outputs,
                                                     annotations)
        return self.box_loss(outputs, (conf_t, loc_t))


class TargetTransform(object):
    def __init__(self, iou_threshold, variance):
        self.threshold = iou_threshold
        self.variance = variance

    def __call__(self, boxes, labels, priors):

        num_priors = priors.size(0)
        loc_t = torch.Tensor(1, num_priors, 4)
        conf_t = torch.LongTensor(1, num_priors)
        boxes = boxes.to(torch.float)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        if widths.min() == 0 or heights.min() == 0:
            idx = (widths > 0).__and__(heights > 0)
            boxes = boxes[idx]
            labels = labels[idx]
        if len(boxes) == 0:
            conf_t[0, :] = 0
            return conf_t.squeeze(0), loc_t.squeeze(0)
        box_utils.match(self.threshold,
                        torch.tensor(boxes).float(), priors, self.variance,
                        torch.tensor(labels), loc_t, conf_t, 0)
        loc_t = loc_t.squeeze(0)
        conf_t = conf_t.squeeze(0)
        return conf_t, loc_t


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self,
                 num_classes,
                 overlap_thresh,
                 prior_for_matching,
                 bkg_label,
                 neg_mining,
                 neg_pos,
                 neg_overlap,
                 encode_target,
                 variance=(0.1, 0.2),
                 priors=None,
                 device=None):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = variance
        self.priors = priors
        self.device = device

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        conf_data, loc_data = predictions
        conf_t, loc_t = targets
        num = loc_data.size(0)

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = box_utils.log_sum_exp(batch_conf) - batch_conf.gather(
            1, conf_t.view(-1, 1))

        # Hard Negative Mining
        tmp = pos.reshape(loss_c.shape)
        loss_c[tmp] = 0  # filter out pos boxes for now

        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(
            -1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_c, loss_l


class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self,
                 min_dim=300,
                 aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                 feature_maps=[19, 10, 5, 3, 2, 1],
                 scales=[45, 90, 135, 180, 225, 270, 315],
                 steps=[16, 32, 64, 100, 150, 300],
                 variance=(0.1, 0.2),
                 clip=True,
                 **kwargs):

        super(PriorBox, self).__init__()
        self.min_dim = min_dim  #[height, width]
        self.feature_maps = feature_maps  #[(height, width), ...]
        self.aspect_ratios = aspect_ratios
        self.num_priors = len(aspect_ratios)
        self.clip = clip
        self.scales = [s / min_dim for s in scales]

        if steps:
            self.steps = [step / min_dim for step in steps]
        else:
            self.steps = [(1 / f_h, 1 / f_w) for f_h, f_w in feature_maps]

        self.offset = [step * 0.5 for step in self.steps]

        self.priors_boxes = {}

    def forward(self, image_shape):
        if image_shape in self.priors_boxes:
            return self.priors_boxes[image_shape]

        mean = []
        # l = 0
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), range(f)):
                cx = j * self.steps[k] + self.offset[k]
                cy = i * self.steps[k] + self.offset[k]
                s_k = self.scales[k]
                mean += [cx, cy, s_k, s_k]

                s_k_prime = sqrt(s_k * self.scales[k + 1])
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    if isinstance(ar, int):
                        ar_sqrt = sqrt(ar)
                        mean += [cx, cy, s_k * ar_sqrt, s_k / ar_sqrt]
                        mean += [cx, cy, s_k / ar_sqrt, s_k * ar_sqrt]
                    elif isinstance(ar, list):
                        mean += [cx, cy, s_k * ar[0], s_k * ar[1]]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        self.priors_boxes[image_shape] = output
        return output


class PredictModel(nn.Module):
    def __init__(self,
                 num_classes,
                 background_label,
                 top_k=200,
                 confidence_thresh=0.01,
                 nms_thresh=0.5,
                 variance=(0.1, 0.2),
                 priors=None):
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
        output = [[torch.tensor([]) for _ in range(self.num_classes)]
                  for _ in range(num)]
        confidences = self.softmax(confidences)
        conf_preds = confidences.view(num, num_priors,
                                      self.num_classes).transpose(2, 1)

        for i in range(num):
            decoded_boxes = box_utils.decode(locations[i], priors,
                                             self.variance)
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
                output[i][cls_idx] = torch.cat(
                    (boxes[ids], scores[ids].unsqueeze(1)), 1)
        return output
