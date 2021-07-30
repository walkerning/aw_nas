from __future__ import print_function

import torch
from torch import nn
from aw_nas.final.base import FinalModel
from aw_nas.weights_manager.necks.base import BaseNeck
from aw_nas.weights_manager.wrapper import BaseHead
from aw_nas.rollout.wrapper import WrapperRollout
from aw_nas.utils import RegistryMeta


class DetectionFinalModel(FinalModel):
    NAME = "det_final_model"
    SCHEDULABLE_ATTRS = []

    def __init__(self,
                 search_space,
                 device,
                 genotypes,
                 backbone_type,
                 backbone_cfg,
                 neck_type,
                 neck_cfg,
                 head_type,
                 head_cfg,
                 feature_levels=[4, 5],
                 supernet_state_dict=None,
                 schedule_cfg=None):
        super(DetectionFinalModel, self).__init__(schedule_cfg=schedule_cfg)
        self.search_space = search_space
        self.device = device
        self.feature_levels = feature_levels

        self.backbone = RegistryMeta.get_class('final_model',
                                               backbone_type)(search_space,
                                                              device,
                                                              **backbone_cfg)

        feature_channels = self.backbone.get_feature_channel_num(
            feature_levels)
        self.neck = BaseNeck.get_class_(
            neck_type)(
            search_space,
            device,
            None,
            feature_channels,
            **neck_cfg)

        neck_feature_channels = self.neck.get_feature_channel_num()

        self.head = BaseHead.get_class_(head_type)(
                device, neck_feature_channels, **head_cfg)

        if supernet_state_dict:
            self.load_supernet_state_dict(supernet_state_dict)
        rollout = search_space.rollout_from_genotype(genotypes)
        self.finalize(rollout)

        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def finalize(self, rollout):
        b_rollout, n_rollout = self._extract_backbone_and_neck_rollout(rollout)
        self.backbone = self.backbone.finalize(b_rollout)
        self.neck = self.neck.finalize(n_rollout)
        return self

    def load_supernet_state_dict(self, supernet_state_dict, strict=True):
        model_state = torch.load(supernet_state_dict, "cpu")
        model_state = model_state.get("weights_manager", model_state)
        backbone_state = {
            k[9:]: v
            for k, v in model_state.items() if k.startswith("backbone")
        }
        neck_state = {
            k[5:]: v
            for k, v in model_state.items() if k.startswith("neck")
        }
        head_state = {
            k[5:]: v
            for k, v in model_state.items() if k.startswith("head")
        }
        self.backbone.load_supernet_state_dict(backbone_state,
                                               filter_regex=r".*classifier.*")
        self.neck.load_state_dict(neck_state, strict=strict)
        self.head.load_state_dict(head_state, strict=strict)
        return self

    def set_hook(self):
        for _, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1) * \
                    module.kernel_size[0] * module.kernel_size[1] * \
                    outputs.size(2) * outputs.size(3) / module.groups
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    @classmethod
    def supported_data_types(cls):
        return ["image"]
    
    @staticmethod
    def _extract_backbone_and_neck_rollout(rollout):
        if isinstance(rollout, WrapperRollout):
            return rollout.backbone, rollout.neck
        return rollout, None

    @staticmethod
    def _pickout_features(features, feature_levels):
        features_shape = {f.shape[-1]: i for i, f in enumerate(features)}
        features = [features[i] for i in sorted(features_shape.values())]
        assert len(features) >= max(feature_levels) + 1
        return [features[level] for level in feature_levels]

    def forward(self, inputs):
        features = self.backbone.extract_features(inputs)
        features = self._pickout_features(features, self.feature_levels)
        features = self.neck(features)
        return self.head(features)

