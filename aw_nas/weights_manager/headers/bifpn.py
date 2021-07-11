from aw_nas.weights_manager.wrapper import BaseHead

from .classifiers import BiFPNClassifier

__all__ = ["BiFPNHead"]


class BiFPNHead(BaseHead):
    NAME = "bifpn_head"

    def __init__(
        self,
        device,
        num_classes,
        feature_channels,
        bifpn_out_channels,
        activation="swish",
        num_layers=4,
        has_backgroud=True,
        schedule_cfg=None,
    ):
        super(BiFPNHeader).__init__(schedule_cfg)
        self.num_classes = num_classes

        num_anchors = 9
        self.reg = BiFPNClassifier(
            bifpn_out_channels, num_anchors, 4, num_layers, activation
        )
        self.cls = BiFPNClassifier(
            bifpn_out_channels,
            num_anchors,
            num_classes + int(has_background),
            num_layers,
            activation,
        )

        self.device = device
        self.pretrained_path = pretrained_path

    def forward(self, features):
        return self.cls(features), self.reg(features)
