# -*- coding: utf-8 -*-
import torch

from aw_nas.objective.base import BaseObjective
from aw_nas.objective.detection_utils import (
    Losses, AnchorsGenerator, Matcher, PostProcessing, Metrics)
from aw_nas.utils.torch_utils import accuracy


class DetectionObjective(BaseObjective):
    NAME = "detection"

    def __init__(self,
                 search_space,
                 num_classes,
                 anchors_generator_type,
                 matcher_type,
                 loss_type,
                 post_processing_type,
                 metrics_type,
                 anchors_generator_cfg={},
                 matcher_cfg={},
                 loss_cfg={},
                 post_processing_cfg={},
                 metrics_cfg={},
                 schedule_cfg=None):
        super(DetectionObjective, self).__init__(search_space,
                                                 schedule_cfg=schedule_cfg)
        # not including background
        self.num_classes = num_classes

        self.anchors = AnchorsGenerator.get_class_(anchors_generator_type)(
            **anchors_generator_cfg)
        self.target_transform = Matcher.get_class_(matcher_type)(**matcher_cfg)
        self.box_loss = Losses.get_class_(loss_type)(num_classes, **loss_cfg)
        self.predictor = PostProcessing.get_class_(post_processing_type)(
            num_classes, anchors=self.anchors, **post_processing_cfg)

        self.metrics = Metrics.get_class_(metrics_type)(**metrics_cfg)

        self.all_boxes = [{} for _ in range(self.num_classes)]
        self.cache = []

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def on_epoch_start(self, epoch):
        super(DetectionObjective, self).on_epoch_start(epoch)
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
        anchors = self.anchors(img_shape).to(device)
        num_anchors = anchors.shape[0]
        location_t = torch.zeros([batch_size, num_anchors, 4])
        confidence_t = torch.zeros([batch_size, num_anchors])

        shapes = []
        for i, anno in enumerate(annotations):
            boxes = anno["boxes"]
            labels = anno["labels"]
            height, width = anno["shape"]
            _id = anno["image_id"]
            boxes = boxes / inputs.shape[-1]
            boxes = boxes.to(device)
            labels = labels.to(device)
            conf_t, loc_t = self.target_transform(boxes, labels, anchors)
            location_t[i] = loc_t
            confidence_t[i] = conf_t
            shapes.append([_id, height, width])

        cache = confidence_t.long().to(device), location_t.to(device), shapes
        if len(self.cache) >= 1:
            del self.cache[-1]
        self.cache.append((hash(str(annotations)), cache))
        return cache

    def perf_names(self):
        return ["mAP"]

    def aggregate_fn(self, perf_name, is_training=True):
        assert perf_name in ["reward", "loss"] + self.perf_names()
        if not is_training and perf_name == "reward":
            return lambda perfs: self.metrics(self.all_boxes)
        return super().aggregate_fn(perf_name, is_training)

    def get_acc(self, inputs, outputs, targets, cand_net):
        conf_t, _, _ = self.batch_transform(inputs, outputs, targets)
        # target: [batch_size, anchor_num, 5], boxes + labels
        keep = conf_t > 0
        confidences, _ = outputs
        return accuracy(confidences[keep], conf_t[keep], topk=(1, 5))

    def get_mAP(self, inputs, outputs, annotations, cand_net):
        """
        Get mAP.
        """
        # This method does not actually calculate mAP, but detection boxes
        # of current batch only. After all batch's boxes are calculated, passing them
        # to method `evaluate_detections`
        confidences, regression = outputs
        detections = self.predictor(confidences, regression, inputs.shape[-1])
        self.metrics.target_to_gt_recs(annotations)
        for batch_id, anno in enumerate(annotations):
            _id = anno["image_id"]
            h, w = anno["shape"]
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

    def get_reward(self, inputs, outputs, targets, cand_net):
        # mAP is actually calculated using aggregate_fn
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
        conf_t, loc_t, _ = self.batch_transform(inputs, outputs,
                                                     annotations)
        return self.box_loss(outputs, (conf_t, loc_t))
