# -*- coding: utf-8 -*-
"""
OFA super net.
"""

import torch
from torch import nn

from aw_nas.common import assert_rollout_type
from aw_nas import utils
from aw_nas.utils import data_parallel
from aw_nas.utils.common_utils import make_divisible
from aw_nas.utils.exception import expect
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.weights_manager.ofa_backbone import BaseBackboneArch
from aw_nas.weights_manager.wrapper import BaseBackboneWeightsManager

try:
    from torch.nn import SyncBatchNorm
    convert_sync_bn = SyncBatchNorm.convert_sync_batchnorm
except ImportError:
    utils.getLogger("weights_manager.ofa").warn(
        "Import convert_sync_bn failed! SyncBatchNorm might not work!")
    convert_sync_bn = lambda m: m

__all__ = ["OFACandidateNet", "OFASupernet"]


class OFASupernet(BaseBackboneWeightsManager, nn.Module):
    NAME = "ofa_supernet"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        backbone_type="mbv2_backbone",
        backbone_cfg={},
        num_classes=10,
        schedule_cfg=None,
    ):
        super(OFASupernet, self).__init__(
            search_space, device, rollout_type, schedule_cfg
        )
        nn.Module.__init__(self)
        self.backbone = BaseBackboneArch.get_class_(backbone_type)(
            device, schedule_cfg=schedule_cfg, **backbone_cfg
        )

        self.reset_flops()
        self.set_hook()

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def forward(self, inputs, rollout=None):
        return self.backbone.forward_rollout(inputs, rollout)

    def extract_features(self, inputs, rollout=None):
        return self.backbone.extract_features(inputs, rollout)

    def get_feature_channel_num(self, p_levels):
        return self.backbone.get_feature_channel_num(p_levels)

    def set_hook(self):
        for name, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += (
                    inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * inputs[0].size(2)
                    * inputs[0].size(3)
                    / (module.stride[0] * module.stride[1] * module.groups)
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        model = OFACandidateNet(self, rollout)
        return model

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("ofa"), assert_rollout_type("ssd_ofa")]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def save(self, path):
        torch.save(
            {
                "epoch": self.epoch,
                "state_dict": self.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # apply the gradients
        optimizer.step()

    def step_current_gradients(self, optimizer):
        optimizer.step()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def finalize(self, rollout):
        pass


class OFACandidateNet(CandidateNet):
    """
    The candidate net for SuperNet weights manager.
    """

    def __init__(self, super_net, rollout):
        super(OFACandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.search_space = super_net.search_space

        self._flops_calculated = False
        self.total_flops = 0
        self.rollout = rollout

    def get_device(self):
        return self._device

    def forward(self, inputs):
        out = self.super_net.forward(inputs, self.rollout)
        return out
