# -*- coding: utf-8 -*-
import torch

from aw_nas.objective.base import BaseObjective
from aw_nas.objective.detection_utils import Metrics
from aw_nas.utils.torch_utils import accuracy, collect_results_gpu, get_dist_info
from aw_nas import utils

try:
    from mmcv.utils.config import ConfigDict
    from mmdet.models.builder import build_head
    from mmdet.core import bbox2result
except ImportError as e:
    utils.getLogger("det_neck").warn(
        "Cannot import mmdet_head, detection NAS might not work: {}".format(e)
    )


class MMDetectionHeadObjective(BaseObjective):
    NAME = "mmdetection_head"

    def __init__(
        self,
        search_space,
        num_classes,
        head_type,
        metrics_type,
        head_cfg,
        metrics_cfg={},
        teacher=None,
        schedule_cfg=None,
    ):
        super(MMDetectionHeadObjective, self).__init__(
            search_space, schedule_cfg=schedule_cfg
        )
        # not including background
        self.num_classes = num_classes

        self.head = build_head(
            ConfigDict({"type": head_type, "num_classes": num_classes, **head_cfg})
        )

        self.teacher = None
        if isinstance(teacher, str):
            self.teacher = torch.load(teacher, "cpu")
            self.soft_loss_cls = Losses.get_class_("adaptive_distillation_loss")(
                beta=1.5, gamma=1.0, temperature=1.0
            )

        self.metrics = Metrics.get_class_(metrics_type)(**metrics_cfg)

        self.all_boxes = [{} for _ in range(self.num_classes)]
        self.results = []
        self.annotations = []
        self.cache = []

        self.is_training = False

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def on_epoch_start(self, epoch):
        super(MMDetectionHeadObjective, self).on_epoch_start(epoch)
        self.search_space.on_epoch_start(epoch)

    def perf_names(self):
        return ["mAP", "FLOPs"]

    def _aggregate_results(self):
        results = collect_results_gpu(self.results)
        annotations = collect_results_gpu(self.annotations)
        self.results = []
        self.annotations = []
        self.all_boxes = [{} for _ in range(self.num_classes)]
        rank, _ = get_dist_info()
        if rank == 0:
            self.metrics.target_to_gt_recs(annotations)
            for out, anno in zip(results, annotations):
                _id = anno["image_id"]
                for j, o in enumerate(out):
                    if len(o) == 0:
                        continue
                    self.all_boxes[j][_id] = o
            return self.metrics(self.all_boxes)
        return 0.0

    def aggregate_fn(self, perf_name, is_training=True):
        assert perf_name in ["reward", "loss"] + self.perf_names()
        if not is_training and perf_name in ("mAP",):
            return lambda perfs: self._aggregate_results()
        return super().aggregate_fn(perf_name, is_training)

    def get_acc(self, inputs, outputs, annotations, cand_net):
        return torch.tensor(0.0), torch.tensor(0.0)
        conf_t, _, _ = self.batch_transform(inputs, outputs, annotations)
        # target: [batch_size, anchor_num, 5], boxes + labels
        keep = conf_t > 0
        _, confidences, _ = outputs
        return accuracy(confidences[keep], conf_t[keep], topk=(1, 5))

    def get_mAP(self, inputs, outputs, annotations, cand_net):
        """
        Get mAP.
        """
        # This method does not actually calculate mAP, but detection boxes
        # of current batch only. After all batch's boxes are calculated, passing them
        # to method `evaluate_detections`
        bboxes = self.head.get_bboxes(*outputs, img_metas=annotations, rescale=True)
        results = [
            bbox2result(box.detach().cpu(), label, self.num_classes)
            for box, label in bboxes
        ]
        self.results.extend(results)
        self.annotations.extend(annotations)
        return [0.0]

    def get_anchors(self, featmap_sizes, img_metas, device="cpu"):
        anchor = self.anchors(featmap_sizes)
        valid = [torch.ones(a.shape[0]).to(torch.bool).to(device) for a in anchor]
        return (
            [
                [a.to(device) * img_meta["img_shape"][0] for a in anchor]
                for img_meta in img_metas
            ],
            [valid] * len(img_metas),
        )

    def get_perfs(self, inputs, outputs, annotations, cand_net):
        if not self.is_training:
            self.get_mAP(inputs, outputs, annotations, cand_net)
        if hasattr(cand_net, "super_net"):
            flops = cand_net.super_net.total_flops / 1e9
            cand_net.super_net.reset_flops()
        elif hasattr(cand_net, "total_flops"):
            flops = cand_net.total_flops / 1e9
            cand_net.total_flops = 0
        else:
            flops = 0
        return [0.0, -flops]

    def get_reward(self, inputs, outputs, annotations, cand_net):
        # mAP is actually calculated using aggregate_fn
        # return self.get_perfs(inputs, outputs, annotations, cand_net)[0]
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
        device = inputs.device
        gt_bboxes = [
            anno["gt_bboxes"].data.to(device).to(torch.float32) for anno in annotations
        ]
        gt_labels = [
            anno["gt_labels"].data.to(device).to(torch.long) for anno in annotations
        ]
        losses = self.head.loss(
            *outputs,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=annotations,
            gt_bboxes_ignore=None
        )

        loss = {k: sum(l) for k, l in losses.items()}
        if self.teacher is not None:
            loss.update(
                {"soft_loss": self.soft_loss(inputs, outputs, annotations, cand_net)}
            )
        return loss

    def set_mode(self, mode):
        assert mode in ("train", "eval")

        if mode == "train":
            self.head.train()
            self.is_training = True
        else:
            self.head.eval()
            self.is_training = False

    def soft_loss(self, inputs, outputs, annotations, cand_net):
        img_metas = annotations
        cls_scores, bbox_pred, centerness_pred = outputs
        if self.teacher is not None:
            self.teacher = self.teacher.to(inputs.device)
            featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
            assert len(featmap_sizes) == self.head.anchor_generator.num_levels

            device = inputs.device
            anchor_list, valid_flag_list = self.head.get_anchors(
                featmap_sizes, img_metas, device=device
            )
            label_channels = (
                self.head.cls_out_channels if self.head.use_sigmoid_cls else 1
            )

            cls_reg_targets = self.head.get_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=None,
                gt_labels_list=gt_labels,
                label_channels=label_channels,
            )

            (
                anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_pos,
                num_total_neg,
            ) = cls_reg_targets

            with torch.no_grad():
                soft_cls, soft_reg, soft_center = self.teacher(inputs)

            (losses_cls,) = multi_apply(
                self.soft_loss_single,
                anchor_list,
                outputs[0],
                outputs[1],
                soft_cls,
                soft_reg,
                labels_list,
                num_total_samples=num_total_pos,
            )
            return sum(losses_cls)

        return torch.tensor(0.0).to(inputs.device)

    def soft_loss_single(
        self,
        anchors,
        cls_score,
        bbox_pred,
        labels,
        bbox_targets,
        hard_labels,
        num_total_samples,
    ):

        anchors = anchors.reshape(-1, 4)
        cls_score = (
            cls_score.permute(0, 2, 3, 1)
            .reshape(-1, self.head.cls_out_channels)
            .contiguous()
        )
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        centerness_targets = centerness_targets.permute(0, 2, 3, 1).reshape(-1)
        labels = (
            labels.permute(0, 2, 3, 1)
            .reshape(-1, self.head.cls_out_channels)
            .contiguous()
        )
        hard_labels = hard_labels.reshape(-1)

        bg_class_ind = self.num_classes
        pos_inds = (
            ((hard_labels >= 0) & (hard_labels < bg_class_ind)).nonzero().squeeze(1)
        )

        if len(pos_inds) > 0:
            # classification loss
            loss_cls = sum(
                self.soft_loss_cls(
                    (None, cls_score),
                    (None, labels),
                    (None, pos_inds),
                    (1.0, len(pos_inds)),
                ).values()
            )

        else:
            loss_cls = cls_score.sum() * 0

        return (loss_cls,)
