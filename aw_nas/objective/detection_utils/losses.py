
import torch
import torch.nn.functional as F

from aw_nas.objective.detection_utils.base import Losses
from aw_nas.utils import box_utils

__all__ = ["MultiBoxLoss"]

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
    def __init__(self,
                 num_classes,
                 neg_mining=True,
                 neg_pos=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos

    def forward(self, predictions, targets):
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
        conf_data, loc_data = predictions
        conf_t, loc_t = targets
        num = loc_data.size(0)

        loc_t = torch.autograd.Variable(loc_t, requires_grad=False)
        conf_t = torch.autograd.Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_anchors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction="sum")

        if self.do_neg_mining:
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes + 1)
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
                -1, self.num_classes + 1)
            conf_t = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, conf_t, reduction="sum")

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_c, loss_l
