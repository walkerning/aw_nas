# -*- coding: utf-8 -*-
"""
A cell-based model whose architecture is described by a genotype.
"""

from __future__ import print_function

import torch
from torch import nn

from aw_nas import ops
from aw_nas.final.base import FinalModel
from aw_nas.final.det_model import HeadModel
from aw_nas.ops import MobileNetV3Block, SeperableConv2d
from aw_nas.utils import RegistryMeta, logger
from aw_nas.utils.exception import ConfigException, expect


class SSDNorm(nn.Module):
    def __init__(self, channel, scale):
        super(SSDNorm, self).__init__()
        self.l2norm = ops.L2Norm(channel, scale)

    def forward(self, features):
        ft0_l2norm = self.l2norm(features[0])
        return [ft0_l2norm] + features[1:]


class Extras(nn.Module):
    def __init__(self, expansions, channels):
        super(Extras, self).__init__()
        self.blocks = nn.ModuleList([
            MobileNetV3Block(exp,
                             in_channels,
                             out_channels,
                             stride=2,
                             affine=True,
                             kernel_size=3,
                             activation='relu') for exp, in_channels,
            out_channels in zip(expansions, channels[:-1], channels[1:])
        ])

    def forward(self, features):
        out = features[-1]
        for block in self.blocks:
            out = block(out)
            features.append(out)
        return features


class Classifier(nn.Module):
    def __init__(self, num_classes, channels, ratios):
        super(Classifier, self).__init__()
        self.convs = nn.ModuleList([
            SeperableConv2d(in_channels,
                            out_channels=ratio * num_classes,
                            kernel_size=3,
                            padding=1)
            for in_channels, ratio in zip(channels, ratios)
        ])

    def forward(self, features):
        return [conv(ft) for ft, conv in zip(features, self.convs)]


class SSDHeadFinalModel(FinalModel):
    NAME = "ssd_head_final_model"

    def __new__(cls,
                device,
                num_classes,
                feature_channels,
                expansions=[0.5, 0.5, 0.5, 0.5],
                channels=[512, 256, 256, 64],
                aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                pretrained_path=None,
                schedule_cfg=None):
        """
        feature_channels: the channels of 2 feature_maps (C4, C5) extracted from backbone
        channels: extras feature_maps(C6, C7, C8, C9) channels
        """
        head_channels = feature_channels + channels

        extras = Extras(expansions, head_channels[1:])
        multi_ratio = [len(r) * 2 + 2 for r in aspect_ratios]
        regression_headers = Classifier(4, head_channels, multi_ratio)
        classification_headers = Classifier(num_classes + 1, head_channels,
                                            multi_ratio)
        expect(None not in [extras, regression_headers, classification_headers],
            "Extras, regression_headers and classification_headers must be provided, "
            "got None instead.", ConfigException)
        head = HeadModel(device,
                         num_classes=num_classes + 1,
                         extras=extras,
                         regression_headers=regression_headers,
                         classification_headers=classification_headers)
        if pretrained_path:
            mismatch = head.load_state_dict(torch.load(pretrained_path, "cpu"),
                                            strict=False)
            logger.info(mismatch)
        return head

    @classmethod
    def supported_data_types(cls):
        return ["image"]


class SSDFinalModel(FinalModel):
    NAME = "ssd_final_model"
    SCHEDULABLE_ATTRS = []

    def __init__(self,
                 search_space,
                 device,
                 backbone_type,
                 backbone_cfg,
                 feature_levels=[4, 5],
                 supernet_state_dict=None,
                 head_type='ssd_head_final_model',
                 head_cfg={},
                 num_classes=10,
                 schedule_cfg=None):
        super(SSDFinalModel, self).__init__(schedule_cfg=schedule_cfg)
        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.feature_levels = feature_levels

        genotypes = backbone_cfg.pop("genotypes")
        self.backbone = RegistryMeta.get_class('final_model',
                                               backbone_type)(search_space,
                                                              device,
                                                              **backbone_cfg)

        feature_channels = self.backbone.get_feature_channel_num(
            feature_levels)
        self.head = SSDHeadFinalModel(device, num_classes, feature_channels,
                                      **head_cfg)

        if supernet_state_dict:
            self.load_supernet_state_dict(supernet_state_dict)
        self.finalize(genotypes)

        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def finalize(self, genotypes):
        self.backbone.finalize(genotypes)
        return self

    def load_supernet_state_dict(self, supernet_state_dict, strict=True):
        model_state = torch.load(supernet_state_dict, "cpu")
        model_state = model_state.get("weights_manager", model_state)
        backbone_state = {
            k[9:]: v
            for k, v in model_state.items() if k.startswith("backbone")
        }
        head_state = {
            k[5:]: v
            for k, v in model_state.items() if k.startswith("head")
        }
        self.backbone.load_supernet_state_dict(backbone_state,
                                               filter_regex=r".*classifier.*")
        self.head.load_state_dict(head_state, strict=strict)
        return self.backbone, self.head

    def set_hook(self):
        for _, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += 2* inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    outputs.size(2) * outputs.size(3) / module.groups
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def forward(self, inputs):
        features, _ = self.backbone.extract_features(inputs, [4, 5])
        confidences, locations = self.head(features)
        return confidences, locations
