import torch
import torch.nn.functional as F

from aw_nas.objective.detection_utils.base import Losses
from aw_nas.utils import box_utils, getLogger


__all__ = ["MultiBoxLoss", "FocalLoss"]


class MultiBoxLoss(Losses):
    NAME = "multibox_loss"
    """SSD Weighted Loss Function
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

    def __init__(self, num_classes, neg_pos=3, loc_coef=1.0, schedule_cfg=None):
        super(MultiBoxLoss, self).__init__(schedule_cfg)
        self.num_classes = num_classes
        self.negpos_ratio = neg_pos
        self.loc_coef = loc_coef

    def filter_samples(self, predictions, targets):
        _, conf_data, loc_data = predictions
        conf_t, _ = targets
        batch_size = loc_data.size(0)

        pos_idx = conf_t > 0
        num_pos = pos_idx.sum(dim=1, keepdim=True)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_anchors,4]

        batch_conf = conf_data.view(-1, self.num_classes + 1)
        loss_c = box_utils.log_sum_exp(batch_conf) - batch_conf.gather(
            1, conf_t.view(-1, 1)
        )
        # Hard Negative Mining
        tmp = pos_idx.reshape(loss_c.shape)
        loss_c[tmp] = 0  # filter out pos boxes for now

        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(
            self.negpos_ratio * num_pos.to(torch.long), max=pos_idx.size(1) - 1
        )
        neg_idx = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        cls_idx = (pos_idx + neg_idx).gt(0)
        N = num_pos.data.sum()
        return (cls_idx, pos_idx), (N, N)

    def forward(self, predictions, targets, indices, normalizer):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_anchors,num_classes)
                loc shape: torch.size(batch_size,num_anchors,4)
                anchors shape: torch.size(num_anchors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        _, conf_data, loc_data = predictions
        conf_t, loc_t = targets

        loc_t = torch.autograd.Variable(loc_t, requires_grad=False)
        conf_t = torch.autograd.Variable(conf_t, requires_grad=False)

        cls_indices, pos_indices = indices
        cls_normalizer, pos_normalizer = normalizer
        conf_data = conf_data[cls_indices]
        conf_t = conf_t[cls_indices]
        loc_p = loc_data[pos_indices].view(-1, 4)
        loc_t = loc_t[pos_indices].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")
        loss_c = F.cross_entropy(conf_data, conf_t, reduction="sum")
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        loss_l /= cls_normalizer
        loss_c /= pos_normalizer
        return {"cls_loss": loss_c, "reg_loss": loss_l * self.loc_coef}


class FocalLoss(Losses):
    NAME = "focal_loss"

    def __init__(
        self,
        num_classes,
        alpha,
        gamma,
        background_label=0,
        loc_coef=1.0,
        schedule_cfg=None,
    ):
        super().__init__(schedule_cfg)
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.background_label = background_label
        self.loc_coef = loc_coef

    def filter_samples(self, predictions, targets):
        conf_t, _ = targets
        reserved_indices = conf_t >= 0
        positive_indices = conf_t > 0
        num_pos_per_batch = positive_indices.sum(-1).to(torch.float)
        has_pos_batch = num_pos_per_batch > 0
        reserved_indices[~has_pos_batch] = False
        positive_indices[~has_pos_batch] = False

        normalizer = torch.unsqueeze(num_pos_per_batch, -1)
        normalizer = normalizer.repeat(1, positive_indices.shape[-1])
        cls_normalizer = normalizer[reserved_indices]
        reg_normalizer = normalizer[positive_indices]
        return (reserved_indices, positive_indices), (cls_normalizer, reg_normalizer)

    def forward(self, predicts, targets, indices, normalizer):
        _, logits, regressions = predicts
        conf_t, loc_t = targets
        confidences = logits.sigmoid()
        confidences = torch.clamp(confidences, 1e-4, 1.0 - 1e-4)
        batch_size = confidences.shape[0]
        device = confidences.device

        reserved_indices, positive_indices = indices
        cls_normalizer, reg_normalizer = normalizer
        confidences = confidences[reserved_indices]
        regressions = regressions[positive_indices]

        # convert label to one-hot encoding.
        conf_t = conf_t[reserved_indices]
        assert 0 < conf_t.max() < confidences.shape[-1], (
            "The number of classes exceeds the number of predicted classes, "
            "please ensure the correction of configuration."
        )
        conf_t = (
            torch.zeros_like(confidences)
            .to(device)
            .scatter_(1, conf_t.reshape(-1, 1), 1)
            .to(torch.float)
        )
        conf_t[:, 0] = self.background_label
        loc_t = loc_t[positive_indices]

        alpha_factor = torch.ones_like(confidences) * self.alpha
        alpha_factor = torch.where(
            torch.eq(conf_t, 1.0), alpha_factor, 1.0 - alpha_factor
        )
        focal_weight = torch.where(
            torch.eq(conf_t, 1.0), 1.0 - confidences, confidences
        )
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        bce = -(
            conf_t * torch.log(confidences)
            + (1.0 - conf_t) * torch.log(1.0 - confidences)
        )
        classification_losses = focal_weight * bce
        classfication_loss = (
            classification_losses / torch.unsqueeze(cls_normalizer, 1)
        ).sum() / batch_size

        regression_diff = torch.abs(loc_t - regressions)
        regression_losses = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0,
        )
        regression_loss = (
            regression_losses.mean(-1) / reg_normalizer
        ).sum() / batch_size

        return {
            "cls_loss": classfication_loss,
            "reg_loss": regression_loss * self.loc_coef,
        }


def sigmoid_cross_entropy_with_logits(logits, labels):
    return (
        torch.where(logits > 0, logits, torch.zeros_like(logits))
        - logits * labels
        + torch.log(1 + torch.exp(-torch.abs(logits)))
    )


class AdaptiveDistillationLoss(Losses):
    """
    ref: [Learning Efficient Detector with Semi-supervised Adaptive Distillation]
    (https://arxiv.org/abs/1901.00366)
    """

    NAME = "adaptive_distillation_loss"
    SCHEDULABLE_ATTRS = ["loss_coef"]

    def __init__(self, beta, gamma, temperature, loss_coef=1.0, schedule_cfg=None):
        super().__init__(schedule_cfg)
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.loss_coef = loss_coef

    def forward(self, predictions, targets, indices, normalizer):
        if self.loss_coef == 0:
            return {"adaptive_soft_loss": torch.tensor(0.0).to(predictions[-1].device)}
        _, positive_indices = indices
        _, pos_normalizer = normalizer

        logit_s = predictions[1]
        logit_t = targets[1]
        logit_s = logit_s[positive_indices] / self.temperature
        logit_t = logit_t[positive_indices] / self.temperature

        conf_t = logit_t.sigmoid()
        soft_cross_entropy = sigmoid_cross_entropy_with_logits(logit_s, conf_t)
        teacher_entropy = sigmoid_cross_entropy_with_logits(logit_t, conf_t)

        kullback_leiber_dist = -teacher_entropy + soft_cross_entropy
        adaptive_weight = 1 - torch.exp(
            -kullback_leiber_dist - self.beta * teacher_entropy
        )
        if self.gamma != 1.0:
            adaptive_weight = torch.pow(adaptive_weight, self.gamma)

        adaptive_soft_loss = adaptive_weight * kullback_leiber_dist
        return {
            "adaptive_soft_loss": (adaptive_soft_loss.sum(-1) / pos_normalizer).sum()
            * self.loss_coef
        }
