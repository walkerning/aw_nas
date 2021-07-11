from aw_nas.weights_manager.headers.classifiers import Classifier
from aw_nas.weights_manager.wrapper import BaseHead


__all__ = ["AnchorHead"]


class AnchorHead(BaseHead):
    NAME = "anchor_based"

    def __init__(
        self,
        device,
        feature_channel_nums,
        num_classes,
        aspect_ratios,
        num_layers=1,
        has_background=True,
        use_separable_conv=True,
        pretrained_path=None,
        schedule_cfg=None,
    ):
        super(AnchorHead, self).__init__(device, feature_channel_nums, schedule_cfg)
        self.num_classes = num_classes
        multi_ratios = [len(r) * 2 + 2 for r in aspect_ratios]
        self.regression = Classifier(4, feature_channel_nums, multi_ratios)
        self.classification = Classifier(
            num_classes + int(has_background), feature_channel_nums, multi_ratios
        )
        self.device = device
        self.pretrained_path = pretrained_path

    def forward(self, features):
        return self.classification(features), self.regression(features)
