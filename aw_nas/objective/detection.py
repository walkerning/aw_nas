# -*- coding: utf-8 -*-
import os
import torch

from aw_nas.final.base import FinalModel
from aw_nas.objective.base import BaseObjective
from aw_nas.objective.detection_utils import (
    Losses,
    AnchorsGenerator,
    Matcher,
    PostProcessing,
    Metrics,
)
from aw_nas.utils.torch_utils import accuracy


class DetectionObjective(BaseObjective):
    NAME = "detection"

    def __init__(
        self,
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
        soft_losses_cfg={},
        post_processing_cfg={},
        metrics_cfg={},
        schedule_cfg=None,
    ):
        super(DetectionObjective, self).__init__(
            search_space, schedule_cfg=schedule_cfg
        )
        # not including background
        self.num_classes = num_classes

        self.anchors = AnchorsGenerator.get_class_(anchors_generator_type)(
            **anchors_generator_cfg
        )
        self.target_transform = Matcher.get_class_(matcher_type)(**matcher_cfg)
        self.box_loss = Losses.get_class_(loss_type)(num_classes, **loss_cfg)
        self.predictor = PostProcessing.get_class_(post_processing_type)(
            num_classes, anchors=self.anchors, **post_processing_cfg
        )
        self.metrics = Metrics.get_class_(metrics_type)(**metrics_cfg)

        self.soft_losses = None
        self.teacher_net = None

        """
        soft_losses_cfg:
            teacher_cfg: str or dict
                teacher_final_type: str
                teacher_final_cfg: {}
            losses_cfg:
              - type: str
                cfg: {}
              - type: ...

        """
        if soft_losses_cfg.get("losses_cfg") and soft_losses_cfg.get("teacher_cfg"):
            teacher_cfg = soft_losses_cfg.get("teacher_cfg", "supernet")
            if isinstance(teacher_cfg, str):
                if teacher_cfg == "supernet":
                    self.teacher_net = "supernet"
                elif os.path.exists(teacher_cfg):
                    self.teacher_net = torch.load(teacher_cfg, "cpu")
                else:
                    raise ValueError(
                        "Except teacher_cfg to be 'supernet' or the path of the teacher model, got {} instead.".format(
                            teacher_cfg
                        )
                    )
            elif isinstance(teacher_cfg, dict):
                state_dict = teacher_cfg.pop("state_dict", None)
                self.teacher_net = FinalModel.get_class_(
                    teacher_cfg["teacher_final_type"]
                )(teacher_cfg["teacher_final_cfg"])
                if state_dict:
                    self.teacher_net.load_state(state_dict, restrict=True)
            else:
                raise ValueError(
                    "Except teacher_cfg to be a str or dict, got {} instead.".format(
                        teacher_cfg
                    )
                )
            losses_cfg = soft_losses_cfg["losses_cfg"]
            if isinstance(losses_cfg, dict):
                losses_cfg = [losses_cfg]
            self.soft_losses = [
                Losses.get_class_(cfg["type"])(**cfg["cfg"]) for cfg in losses_cfg
            ]

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
        features, _, _ = outputs
        feature_maps = [ft.shape[-2:] for ft in features]
        for _id, cache in self.cache:
            if _id == hash(str(annotations) + str(feature_maps)):
                return cache
        device = inputs.device
        img_shape = inputs.shape[-1]
        batch_size = inputs.shape[0]
        anchors = torch.cat(self.anchors(feature_maps), dim=0).to(device)
        num_anchors = anchors.shape[0]
        location_t = torch.zeros([batch_size, num_anchors, 4])
        confidence_t = torch.zeros([batch_size, num_anchors])

        shapes = []
        for i, anno in enumerate(annotations):
            boxes = anno["boxes"] / inputs.shape[-1]
            labels = anno["labels"]
            height, width = anno["shape"]
            _id = anno["image_id"]
            boxes = boxes.to(device)
            labels = labels.to(device)
            conf_t, loc_t = self.target_transform(boxes, labels, anchors)
            location_t[i] = loc_t
            confidence_t[i] = conf_t
            shapes.append([_id, height, width])

        cache = confidence_t.long().to(device), location_t.to(device), shapes
        if len(self.cache) >= 2:
            del self.cache[0]
        self.cache.append((hash(str(annotations) + str(feature_maps)), cache))
        return cache

    def perf_names(self):
        return ["mAP", "FLOPs"]

    def aggregate_fn(self, perf_name, is_training=True):
        assert perf_name in ["reward", "loss"] + self.perf_names()
        if not is_training and perf_name in ("reward", "mAP"):
            return lambda perfs: self.metrics(self.all_boxes)
        return super().aggregate_fn(perf_name, is_training)

    def get_acc(self, inputs, outputs, annotations, cand_net):
        conf_t, _, _ = self.batch_transform(inputs, outputs, annotations)
        # target: [batch_size, anchor_num, 5], boxes + labels
        keep = conf_t > 0
        _, confidences, _ = outputs
        confidences = torch.cat(
            [
                conf.permute(0, 2, 3, 1).reshape(
                    conf.shape[0], -1, self.num_classes + 1
                )
                for conf in confidences
            ],
            dim=1,
        )
        return accuracy(confidences[keep], conf_t[keep], topk=(1, 5))

    def get_mAP(self, inputs, outputs, annotations, cand_net):
        """
        Get mAP.
        """
        # This method does not actually calculate mAP, but detection boxes
        # of current batch only. After all batch's boxes are calculated, passing them
        # to method `evaluate_detections`
        features, confidences, regressions = outputs
        confidences = torch.cat(
            [
                conf.permute(0, 2, 3, 1).reshape(
                    conf.shape[0], -1, self.num_classes + 1
                )
                for conf in confidences
            ],
            dim=1,
        )
        regressions = torch.cat(
            [
                reg.permute(0, 2, 3, 1).reshape(reg.shape[0], -1, 4)
                for reg in regressions
            ],
            dim=1,
        )
        outputs = features, confidences, regressions
        detections = self.predictor(*outputs)
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
                cls_dets = torch.cat((boxes, scores), dim=1).cpu().detach().numpy()
                self.all_boxes[j][_id] = cls_dets
        return [0.0]

    def get_perfs(self, inputs, outputs, annotations, cand_net):
        acc = self.get_acc(inputs, outputs, annotations, cand_net)
        if not cand_net.training:
            self.get_mAP(inputs, outputs, annotations, cand_net)
        if hasattr(cand_net, "super_net"):
            flops = cand_net.super_net.total_flops / 1e9
            cand_net.super_net.reset_flops()
        elif hasattr(cand_net, "total_flops"):
            flops = cand_net.total_flops / 1e9
            cand_net.total_flops = 0
        else:
            flops = 0
        return [acc[0].item(), -flops]

    def get_reward(self, inputs, outputs, annotations, cand_net):
        # mAP is actually calculated using aggregate_fn
        return 0.0

    def get_loss(
        self,
        inputs,
        outputs,
        annotations,
        cand_net,
        add_controller_regularization=True,
        add_evaluator_regularization=True,
    ):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        return sum(self._criterion(inputs, outputs, annotations, cand_net).values())

    def _criterion(self, inputs, outputs, annotations, cand_net):
        _, confidences, regressions = outputs
        confidences = torch.cat(
            [
                conf.permute(0, 2, 3, 1).reshape(
                    conf.shape[0], -1, self.num_classes + 1
                )
                for conf in confidences
            ],
            dim=1,
        )
        regressions = torch.cat(
            [
                reg.permute(0, 2, 3, 1).reshape(reg.shape[0], -1, 4)
                for reg in regressions
            ],
            dim=1,
        )
        outputs = _, confidences, regressions

        conf_t, loc_t, _ = self.batch_transform(inputs, outputs, annotations)
        indices, normalizer = self.box_loss.filter_samples(outputs, (conf_t, loc_t))
        hard_losses = self.box_loss(outputs, (conf_t, loc_t), indices, normalizer)
        soft_losses = {}
        if self.soft_losses is not None and cand_net.training:
            if self.teacher_net == "supernet":
                teacher_net = cand_net.super_net
            else:
                self.teacher_net = self.teacher_net.to(inputs.device)
                teacher_net = self.teacher_net
            teacher_net.train()
            with torch.no_grad():
                soft_target = teacher_net(inputs)
            [
                soft_losses.update(soft_loss(outputs, soft_target, indices, normalizer))
                for soft_loss in self.soft_losses
            ]
        return {**hard_losses, **soft_losses}
