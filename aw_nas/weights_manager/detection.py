"""
Super net for detection tasks.
"""
import six

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aw_nas import utils
from aw_nas.common import assert_rollout_type
from aw_nas.utils import data_parallel
from aw_nas.utils.torch_utils import _to_device
from aw_nas.utils.common_utils import make_divisible, nullcontext
from aw_nas.utils import DistributedDataParallel
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.weights_manager.detection_header import DetectionHeader

try:
    from aw_nas.utils.SynchronizedBatchNormPyTorch.sync_batchnorm import (
        convert_model as convert_sync_bn,
    )
except ImportError:
    def convert_sync_bn(m): return m

__all__ = ["DetectionBackboneSupernet"]


class DetectionBackboneSupernet(BaseWeightsManager, nn.Module):
    NAME = "det_supernet"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        feature_levels=[3, 4, 5],
        search_backbone_type="ofa_supernet",
        search_backbone_cfg={},
        head_type="ssd_header",
        head_cfg={},
        num_classes=21,
        multiprocess=False,
        gpus=tuple(),
        schedule_cfg=None,
    ):
        super(DetectionBackboneSupernet, self).__init__(
            search_space, device, rollout_type, schedule_cfg
        )
        nn.Module.__init__(self)
        self.backbone = BaseWeightsManager.get_class_(search_backbone_type)(
            search_space, device, rollout_type,
            multiprocess=False,
            gpus=gpus,
            **search_backbone_cfg
        )
        self.multiprocess = multiprocess
        self.gpus = gpus

        self.feature_levels = feature_levels
        backbone_stage_channel = self.backbone.backbone.get_feature_channel_num(
            feature_levels)
        cfg_channels = head_cfg.get("feature_channels", backbone_stage_channel)

        self.head = DetectionHeader.get_class_(head_type)(
            device,
            num_classes,
            cfg_channels,
            **head_cfg
        )

        self.reset_flops()
        self.set_hook()
        self._parallelize()

    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self).to(self.device)
            object.__setattr__(
                self, "parallel_model", DistributedDataParallel(
                    net, self.gpus, find_unused_parameters=True).to(self.device)
            )
        else:
            self.to(self.device)

    def forward(self, inputs, rollout=None):
        features, out = self.backbone.extract_features(
            inputs, self.feature_levels, rollout)
        features, confidences, regression = self.head.forward_rollout(features, rollout)
        return features, confidences, regression

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
        return DetectionBackboneCandidateNet(self, rollout)

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


class DetectionBackboneCandidateNet(CandidateNet):
    def __init__(self, super_net, rollout, gpus=tuple()):
        super(DetectionBackboneCandidateNet, self).__init__()
        self.super_net = super_net
        self.rollout = rollout
        self._device = self.super_net.device
        self.gpus = gpus
        self.multiprocess = self.super_net.multiprocess

    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net(inputs, self.rollout)

    def forward(self, inputs, single=False):
        if self.multiprocess:
            out = self.super_net.parallel_model.forward(inputs, self.rollout)
        elif len(self.gpus) > 1:
            out = data_parallel(
                self, (inputs,), self.gpus, module_kwargs={"single": True}
            )
        else:
            return self._forward(inputs)
        return out

    def gradient(self,
                 data,
                 criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                 parameters=None,
                 eval_criterions=None,
                 mode="train",
                 zero_grads=True,
                 return_grads=True,
                 **kwargs):
        """Get the gradient with respect to the candidate net parameters.

        Args:
            parameters (optional): if specificied, can be a dict of param_name: param,
            or a list of parameter name.
        Returns:
            grads (dict of name: grad tensor)
        """
        self._set_mode(mode)

        if return_grads:
            active_parameters = dict(self.named_parameters())
            if parameters is not None:
                _parameters = dict(parameters)
                _addi = set(_parameters.keys()).difference(active_parameters)
                assert not _addi,\
                    ("Cannot get gradient of parameters that are not active "
                     "in this candidate net: {}")\
                    .format(", ".join(_addi))
            else:
                _parameters = active_parameters
        inputs, targets = data
        batch_size = inputs.size(0)
        min_image_size = min(self.super_net.search_space.image_size_choice)
        cur_image_size = self.rollout.image_size
        ratio = (min_image_size / cur_image_size) ** 2
        mini_batch_size = make_divisible(batch_size * ratio, 8)
        inputs = F.interpolate(inputs, (cur_image_size, cur_image_size),
                               mode="bilinear", align_corners=False)
        if zero_grads:
            self.zero_grad()
        for i in range(
                0, batch_size // mini_batch_size +
            int(batch_size % mini_batch_size != 0), mini_batch_size
        ):
            mini_inputs = inputs[i: i + mini_batch_size]
            mini_targets = targets[i: i + mini_batch_size]
            outputs = self.forward_data(mini_inputs, mini_targets, **kwargs)
            loss = criterion(mini_inputs, outputs, mini_targets)

            loss.backward()

        if not return_grads:
            grads = None
        else:
            grads = [(k, v.grad.clone()) for k, v in six.iteritems(_parameters)
                     if v.grad is not None]

        if eval_criterions:
            eval_res = utils.flatten_list(
                [c(mini_inputs, outputs, mini_targets) for c in eval_criterions])
            return grads, eval_res
        return grads

    def eval_queue(self,
                   queue,
                   criterions,
                   steps=1,
                   mode="eval",
                   aggregate_fns=None,
                   **kwargs):
        self._set_mode(mode)

        aggr_ans = []
        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for _ in range(steps):
                data = next(queue)
                # print("{}/{}\r".format(i, steps), end="")
                data = _to_device(data, self.get_device())
                outputs = self.forward_data(data[0], **kwargs)
                self._set_mode("eval")  # mAP only is calculated in "eval" mode
                ans = utils.flatten_list(
                    [c(data[0], outputs, data[1]) for c in criterions])
                aggr_ans.append(ans)
                self._set_mode(mode)
        aggr_ans = np.asarray(aggr_ans).transpose()
        if aggregate_fns is None:
            # by default, aggregate batch rewards with MEAN
            aggregate_fns = [lambda perfs: np.mean(perfs) if len(perfs) > 0 else 0.]\
                * len(aggr_ans)
        return [aggr_fn(ans) for aggr_fn, ans in zip(aggregate_fns, aggr_ans)]
