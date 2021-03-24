
import copy

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from aw_nas import ops
from aw_nas.ops import FlexibleDepthWiseConv, FlexiblePointLinear, FlexibleBatchNorm2d
from aw_nas.weights_manager import FlexibleBlock
from aw_nas.weights_manager.detection_header import DetectionHeader

class FlexibleSepConv(FlexibleBlock):
    NAME = "sep_conv"

    def __init__(self, in_channels, out_channels, norm=True, kernel_sizes=[3]):
        super(FlexibleSepConv, self).__init__()

        self.depthwise_conv = FlexibleDepthWiseConv(in_channels, kernel_sizes,
                                                      stride=1, bias=False)
        #self.depthwise_conv = ops.Conv2dStaticSamePadding(in_channels, in_channels, kernel_sizes[0], stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = FlexiblePointLinear(in_channels, out_channels, bias=True)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.reset_mask()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)
        return x

    def set_mask(self, mask):
        pass

    def finalize(self):
        return self


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, activation="swish", onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList([
                FlexibleSepConv(in_channels, in_channels, norm=False) for i in range(num_layers)
            ])

        self.bn_list = nn.ModuleList([
            nn.ModuleList([
                FlexibleBatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)
            ]) 
            for j in range(5)
        ])
        self.header = FlexibleSepConv(in_channels, num_anchors * num_classes, norm=False)
        self.act = ops.get_op(activation)() 

    def forward(self, inputs, post_process=False):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.act(feat)
            feat = self.header(feat)
            feats.append(feat)
        return feats

class FlexibleBiFPNBlock(FlexibleBlock):
    NAME = "bifpn_block"

    def __init__(self, num_channels, conv_channels, first_time=False,
            epsilon=1e-4, attention=True, activation="swish", schedule_cfg=None):
        super().__init__(schedule_cfg)
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.first_time = first_time
        self.epsilon = epsilon
        self.attention = attention

        self.conv_up = nn.ModuleDict(OrderedDict({
            str(k): FlexibleSepConv(num_channels, num_channels) for k in range(6, 2, -1)
        }))
        self.conv_down = nn.ModuleDict(OrderedDict({
            str(k): FlexibleSepConv(num_channels, num_channels) for k in range(4, 8)
        }))

        self.downsample = nn.ModuleDict(OrderedDict({
            #str(k): ops.MaxPool2dStaticSamePadding((3, 3), (2, 2)) for k in range(4, 8)
            str(k): nn.MaxPool2d((3, 3), (2, 2), padding=(1,1)) for k in range(4, 8)
        }))

        self.swish = ops.get_op(activation)()
        #self.relu_fn = ops.get_op("relu6")

        self.weights_1 = nn.ParameterDict({
            str(k): nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) 
            for k in range(3, 7)
        })

        self.weights_2 = nn.ParameterDict({
            str(k): nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True) 
            for k in range(4, 7)
        })
        self.weights_2["7"] = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        if self.first_time:
            self.down_channel = nn.ModuleDict({str(k): nn.Sequential(
                FlexiblePointLinear(conv_channels[i], num_channels, bias=True),
                FlexibleBatchNorm2d(num_channels, momentum=0.01, eps=1e-3)
            ) for i, k in enumerate(range(3, 6))})

            self.down_channel_2 = nn.ModuleDict({str(k): nn.Sequential(
                FlexiblePointLinear(conv_channels[i], num_channels, bias=True),
                FlexibleBatchNorm2d(num_channels, momentum=0.01, eps=1e-3)
            ) for i, k in enumerate(range(4, 6), 1)})

            self.p5_to_p6 = nn.Sequential(
                FlexiblePointLinear(conv_channels[2], num_channels, bias=True),
                FlexibleBatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                #ops.MaxPool2dStaticSamePadding((3, 3), (2, 2)),
                nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1)),
            )

            self.p6_to_p7 = nn.Sequential(
                #ops.MaxPool2dStaticSamePadding((3, 3), (2, 2)),
                nn.MaxPool2d((3, 3), (2, 2), padding=(1, 1)),
            )


        self.reset_mask()

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        if self.attention:
            return self._forward_rollout(features, rollout, self._fast_attention)
        return self._forward_rollout(features, rollout, self._no_attention)

    def _fast_attention(self, weight, *args):
        weight = ops.get_op("relu6")(inplace=False)(weight)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        assert len(weight) == len(args)
        return sum([w * arg for w, arg in zip(weight, args)])

    def _no_attention(self, weight, *args):
        return sum(args)

    def _forward_rollout(self, features, rollout=None, attention_fn=None):
        if attention_fn is None:
            attention_fn = self._fast_attention

        if self.first_time:
            p_in = {}
            p_in[3], p_in[4], p_in[5] = features
            p4, p5 = p_in[4], p_in[5]
            
            p_in[6] = self.p5_to_p6(p_in[5])
            p_in[7] = self.p6_to_p7(p_in[6])

            for i, down in sorted(self.down_channel.items()):
                i = int(i)
                p_in[i] = down(p_in[i])
        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p_in = {k: f for k, f in enumerate(features, 3)}

        # Connections for P6_0 and P7_0 to P6_1 respectively
        prev = p_in[7]
        up = {}
        for i, conv in sorted(self.conv_up.items(), reverse=True):
            ii = int(i)
            up[ii] = conv(self.swish(attention_fn(self.weights_1[i], p_in[ii], F.interpolate(prev, p_in[ii].shape[-2:], mode="nearest"))))
            prev = up[ii]

        if self.first_time:
            p_in[4] = self.down_channel_2["4"](p4)
            p_in[5] = self.down_channel_2["5"](p5)

        outs = {3: prev}
        for i, conv in sorted(self.conv_down.items()):
            ii = int(i)
            if ii in up:
                outs[ii] = conv(self.swish(attention_fn(self.weights_2[i], p_in[ii], up[ii], self.downsample[i](prev))))
            else:
                outs[ii] = conv(self.swish(attention_fn(self.weights_2[i], p_in[ii], self.downsample[i](prev))))
            prev = outs[ii]

        return tuple([v for k, v in sorted(outs.items())])

    def set_mask(self, expansions, kernel_sizes):
        assert len(expansions) == len(kernel_sizes) == len(self.blocks)
        for block, exp, kernel in zip(self.blocks, expansions, kernel_sizes):
            block.set_mask(exp, kernel)

    def finalize(self):
        return self

class FlexibleBiFPNExtras(FlexibleBlock):
    def __init__(self, in_channels, out_channels, activation="swish", attention=True, repeat=3):
        super(FlexibleBiFPNExtras, self).__init__()

        self.blocks = nn.Sequential(*[
            FlexibleBiFPNBlock(out_channels, in_channels, first_time=i==0, epsilon=1e-4,
                activation=activation, attention=attention) for i in range(repeat)   
        ])

    def forward(self, features):
        return self.forward_rollout(features)

    def forward_rollout(self, features, rollout=None):
        return self.blocks(features)

    def finalize(self):
        self.blocks = nn.Sequential(*[m.finalize() for m in self.blocks])
        return self


class BiFPN(DetectionHeader):
    NAME = "bifpn_header"

    def __init__(self, device, num_classes, feature_channels, bifpn_out_channels, 
                 activation="swish", attention=True, repeat=3, num_layers=4, pretrained_path=None, schedule_cfg=None):
        super().__init__(schedule_cfg)
        self.num_classes = num_classes
        self.extras = FlexibleBiFPNExtras(feature_channels, bifpn_out_channels, activation, attention, repeat)

        num_anchors = 9
        self.regression_headers = Classifier(bifpn_out_channels, num_anchors, 4, num_layers, activation)
        self.classification_headers = Classifier(bifpn_out_channels,
                num_anchors, num_classes + 1, num_layers, activation)
                                    
        self.device = device
        self.pretrained_path = pretrained_path

    def forward(self, features):
        return self.extras(features)

    def finalize(self, rollout):
        finalized_model = copy.deepcopy(self)
        finalized_model.extras = finalized_model.extras.finalize()
        return finalized_model

