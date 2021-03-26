# -*- coding: utf-8 -*-
"""
A cell-based model whose architecture is described by a genotype.
"""

from __future__ import print_function

import re

import torch
from torch import nn

from aw_nas.common import genotype_from_str
from aw_nas.final.base import FinalModel
from aw_nas.weights_manager.ofa_backbone import BaseBackboneArch
from aw_nas.germ import GermWeightsManager


class OFAGenotypeModel(FinalModel):
    NAME = "ofa_final_model"

    def __init__(self, search_space, device,
                 genotypes=None,
                 backbone_type="mbv2_backbone",
                 backbone_cfg=None,
                 supernet_state_dict=None,
                 filter_regex=None,
                 schedule_cfg=None):
        super(OFAGenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        self.backbone_cfg = backbone_cfg or {}

        self.backbone = BaseBackboneArch.get_class_(backbone_type)(
            device=self.device, **backbone_cfg)
        self.load_supernet_state_dict(supernet_state_dict, filter_regex)
        self.genotypes = genotypes
        if genotypes:
            rollout = search_space.rollout_from_genotype(genotypes)
            self.finalize(rollout)

        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def forward(self, inputs):
        res = self.backbone(inputs)
        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops/1.e6)
            self._flops_calculated = True
        return res

    def extract_features(self, inputs,  drop_connect_rate=0.0):
        return self.backbone.extract_features(inputs, drop_connect_rate=drop_connect_rate)

    def get_feature_channel_num(self, p_level):
        return self.backbone.get_feature_channel_num(p_level)

    def load_state_dict(self, model, strict=True):
        keys = model.keys()
        for key in keys:
            if key.startswith('backbone'):
                return super().load_state_dict(model, strict)
            else:
                return self.backbone.load_state_dict(model, strict)

    def load_supernet_state_dict(self, supernet_state_dict, filter_regex=None):
        """
        supernet_state_dict includes all params and weights of FlexibileArch
        """
        if supernet_state_dict is not None:
            if isinstance(supernet_state_dict, dict):
                state_dict = supernet_state_dict
            else:
                state_dict = torch.load(supernet_state_dict, map_location="cpu")
            state_dict = state_dict.get("weights_manager", state_dict)
            if filter_regex is not None:
                regex = re.compile(filter_regex)
                state_dict = {k: v for k, v in state_dict.items() if not regex.match(k)}
            mismatch = self.load_state_dict(state_dict, strict=filter_regex is None)
            self.logger.info("loading supernet: " + str(mismatch))
        return self

    def finalize(self, rollout):
        self.backbone = self.backbone.finalize(rollout.depth, rollout.width, rollout.kernel)
        return self

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

    def layer_idx_to_named_modules(self, idx):
        stage_idx, block_idx = idx
        prefix = "backbone.cells.{}.{}".format(stage_idx, block_idx)
        m = self
        for name in prefix.split('.'):
            m = getattr(m, name)
        for n, _ in m.named_modules():
            if not n:
                yield prefix
            yield '.'.join([prefix, n])
