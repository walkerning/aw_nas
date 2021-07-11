# -*- coding: utf-8 -*-
"""
WrapperWeightsManager, BaseBackboneWeightsManager
"""

import abc

import torch
from torch import nn

from aw_nas import Component, ops
from aw_nas.utils import expect
from aw_nas.common import BaseRollout
from aw_nas.rollout.wrapper import WrapperSearchSpace, WrapperRollout
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.weights_manager.necks.base import BaseNeck
from aw_nas.utils.parallel_utils import parallelize

__all__ = [
    "BaseHead", "BaseBackboneWeightsManager",
    "WrapperCandidateNet", "WrapperWeightsManager",
    "ClassificationHead"
]

class BaseHead(Component, nn.Module):
    REGISTRY = "head"

    def __init__(self, device, feature_channel_nums, schedule_cfg=None):
        super().__init__(schedule_cfg)
        nn.Module.__init__(self)

        self.device = device
        self.feature_channel_nums = feature_channel_nums


class BaseBackboneWeightsManager(BaseWeightsManager):
    @abc.abstractmethod
    def extract_features(self, inputs, rollout):
        """
        Return the internal features maps (the last feature map of each spartial dimension).
        """

    @abc.abstractmethod
    def get_feature_channel_num(self, feature_levels=None):
        """
        Get a list of channel numbers of feature maps of certain feature_levels.
        By default, the channel numbers of all feature levels are returned.
        """

    @abc.abstractmethod
    def finalize(self, rollout):
        """
        In-place finalize model, and return self
        """


class WrapperCandidateNet(CandidateNet):
    def __init__(self, super_net, rollout, eval_no_grad=True):
        super(WrapperCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = super_net
        self._device = self.super_net.device
        self.rollout = rollout

    def forward(self, inputs): #pylint: disable=arguments-differ
        return self.super_net(inputs, self.rollout)

    def _forward_with_params(self, *args, **kwargs): #pylint: disable=arguments-differ
        raise NotImplementedError()

    def get_device(self):
        return self._device
    

@parallelize()
class WrapperWeightsManager(BaseWeightsManager, nn.Module):
    """
    A ``wrapper'' (shared weights) weights manager that delegate the calls to subcomponents:
    backbone, neck and head.

    `backbone` and `neck` are instances of weights manager (supernets),
    and can accept rollout to specify architecture.
    `head` is a `BaseHead` instance, usually not searchable. Currently,
    only the `forward` method is required to be implemented.

    Note that we do not automatically generate the available default configs
    for the backbone/neck search spaces in `gen-sample-config`
    in wrapper search space, weights manager.

    The concrete meaning of `feature_levels` depends on specific backbone/neck/head implementations.

    NOTE: Use wrapper weights manager to run examples/mloss/enas/enas_search.yaml,
    7% slower than the original one: 75.6s/epoch (6epoch 7min34s) v.s. 70.5s/epoch (6epoch 7min3s)
    """

    NAME = "wrapper"

    def __init__(self, search_space, device, rollout_type,
                 backbone_type=None, backbone_cfg=None,
                 neck_type=None, neck_cfg=None,
                 head_type=None, head_cfg=None,
                 feature_levels=[-1],
                 max_grad_norm=None,
                 schedule_cfg=None):
        super().__init__(search_space, device, rollout_type, schedule_cfg)
        nn.Module.__init__(self)

        expect(backbone_type is not None)
        expect(head_type is not None)

        # check search space type
        if isinstance(search_space, WrapperSearchSpace):
            backbone_ss = search_space.backbone
            backbone_rollout_type = search_space.backbone_rollout_type
            neck_ss = search_space.neck
            neck_rollout_type = search_space.neck_rollout_type
            expect(self.rollout_type == "wrapper",
                   "WrapperSearchSpace corresponds to `wrapper` rollout")
        else:
            # search_space should be compatible with backbone weights manager
            backbone_ss = search_space
            backbone_rollout_type = rollout_type
            # neck weights manager must accept `search_space=None` and `forward(rollout=None)`
            # or neck is None
            neck_ss = None
            neck_rollout_type = None

        # initialize backbone/neck/head
        self.backbone = BaseBackboneWeightsManager.get_class_(backbone_type)(
            backbone_ss, self.device, backbone_rollout_type, **(backbone_cfg or {}))
        feature_channel_nums = self.backbone.get_feature_channel_num(feature_levels)
        if neck_type is not None:
            self.neck = BaseNeck.get_class_(neck_type)(
                neck_ss, self.device, neck_rollout_type, feature_channel_nums, **(neck_cfg or {}))
            feature_channel_nums = self.neck.get_feature_channel_num()
        else:
            self.neck = None
        self.head = BaseHead.get_class_(head_type)(
            self.device, feature_channel_nums, **(head_cfg or {}))

        # other configs
        self.max_grad_norm = max_grad_norm
        # the features that need to be passed from backbone to neck
        self.feature_levels = feature_levels

        self.to(self.device)

        self.reset_flops()
        self.set_hook()

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

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

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, inputs, rollout): #pylint: disable=arguments-differ
        b_rollout, n_rollout = self._extract_backbone_and_neck_rollout(rollout)
        features = self.backbone.extract_features(inputs, b_rollout)
        features = self._pickout_features(features, self.feature_levels)
        if self.neck is not None:
            features = self.neck.forward_rollout(n_rollout, features)
        return self.head(features)

    def assemble_candidate(self, rollout):
        """
        Assemble a wrapper candidate net.
        """
        return WrapperCandidateNet(self, rollout)

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def save(self, path):
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    @classmethod
    def supported_rollout_types(cls):
        """
        Return all the rollout types for class check.
        Whether the sub rollout types are supported by backbone/neck
        is checked in the `__init__` method.
        """
        return list(BaseRollout.all_classes_().keys())

    @classmethod
    def supported_data_types(cls):
        return ["image"]


# simple classification head
class ClassificationHead(BaseHead):
    """
    Classification head. By default, global avg pooling and one linear layer.
    TODO: support multiple layers and so on
    """
    NAME = "classification"

    def __init__(self, device, feature_channel_nums,
                 num_classes=10, dropout_rate=0., schedule_cfg=None):
        super().__init__(device, feature_channel_nums, schedule_cfg)

        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()

        # only use the final feature map
        self.classifier = nn.Linear(self.feature_channel_nums[-1], self.num_classes)
        self.to(self.device)

    def forward(self, features):
        out = self.global_pooling(features[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
