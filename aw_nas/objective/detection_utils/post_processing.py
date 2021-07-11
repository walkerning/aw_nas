import torch
import torchvision

try:
    from torchvision.ops import nms
except ModuleNotFoundError:
    from aw_nas.utils import log as _log

    _log.logger.getChild("detection").warn(
        "Detection task functionalities cannot be used, update torchvision version to >=0.4.0. "
        "current: %s",
        str(torchvision.__version__),
    )

from aw_nas.objective.detection_utils.base import PostProcessing
from aw_nas.utils import box_utils

__all__ = ["SSDPostProcessing"]


class SSDPostProcessing(PostProcessing):
    NAME = "ssd_post_processing"

    def __init__(
        self,
        num_classes,
        top_k=200,
        confidence_threshold=0.01,
        nms_threshold=0.5,
        variance=(0.1, 0.2),
        apply_prob_type="softmax",
        anchors=None,
        schedule_cfg=None,
    ):
        super(SSDPostProcessing, self).__init__(schedule_cfg)
        self.num_classes = num_classes
        self.top_k = top_k
        self.confidence_thresh = confidence_threshold
        self.nms_thresh = nms_threshold
        self.variance = variance
        self.anchors = anchors
        if apply_prob_type == "softmax":
            self.prob_fn = torch.nn.Softmax(dim=-1)
        elif apply_prob_type == "sigmoid":
            self.prob_fn = torch.sigmoid
        else:
            raise ValueError(
                "Except apply_prob_type is one of 'sigmoid' or 'softmax', "
                "got {} instead.".format(apply_prob_type)
            )

    def __call__(self, features, confidences, locations):
        feature_maps = [ft.shape[-2:] for ft in features]
        anchors = torch.cat(self.anchors(feature_maps)).to(confidences.device)
        num = confidences.size(0)  # batch size
        num_anchors = anchors.size(0)
        output = [
            [torch.tensor([]) for _ in range(self.num_classes)] for _ in range(num)
        ]
        confidences = self.prob_fn(confidences)
        conf_preds = confidences.view(num, num_anchors, self.num_classes + 1).transpose(
            2, 1
        )

        for i in range(num):
            decoded_boxes = box_utils.decode(locations[i], anchors, self.variance)
            conf_scores = conf_preds[i].clone()

            for cls_idx in range(self.num_classes):
                c_mask = conf_scores[cls_idx + 1].gt(self.confidence_thresh)
                scores = conf_scores[cls_idx + 1][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids = nms(boxes, scores, self.nms_thresh)
                output[i][cls_idx] = torch.cat(
                    (boxes[ids], scores[ids].unsqueeze(1)), 1
                )
        return output
