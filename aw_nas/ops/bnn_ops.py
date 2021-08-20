"""
BNN operations.
"""
# pylint: disable=arguments-differ,useless-super-delegation,invalid-name

import torch
from torch import nn
import torch.nn.functional as F

from aw_nas.ops import register_primitive, ConvBNReLU, Identity

# -------- Activatio Binarization Function --------

class BirealBinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # refered the birealnet implemention https://github.com/liuzechun/Bi-Real-net/tree/master/pytorch_implementation/BiReal18_34
        out_forward = torch.sign(x)
        # out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (
            1 - mask1.type(torch.float32)
        )
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (
            1 - mask2.type(torch.float32)
        )
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class StraightThroughBinaryActivation(torch.autograd.Function):
    # -- the method dict --
    @staticmethod
    def forward(ctx, inputs, method=torch.tensor([0.0])):
        ctx.save_for_backward(inputs, method)
        # BinaryActivatioFunction
        #   -1: float
        #    0: sign()
        if method == 0:
            inputs = inputs.sign()
        elif method == -1:
            inputs = inputs
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        # straight through the grad
        inputs, method = ctx.saved_tensors
        return grad_output


# ---- XNOR modules ----
class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fp_weight, binarize_cfgs):

        # === break down the binarize_cfgs ===
        scale = binarize_cfgs["bi_w_scale"]
        method = binarize_cfgs["bi_act_method"]
        STE_method = binarize_cfgs["STE_method"]
        old_x = x
        # === binarize the activation ===
        # === Activation Type ==========
        # 0 - normal sign with staight through
        # 1 - bireal activtion
        # -1 - no binarization
        if method == 1:
            x = BirealBinaryActivation.apply(x)
        else:
            x = StraightThroughBinaryActivation.apply(x, method)


        #  === binarize the weight ===
        # zero-mean the weight
        # fp_weight.data = fp_weight.data - fp_weight.data.mean(1, keepdim=True)

        # for binary weight scale:
        # -1 - no binary for weight
        # 0 - no_scale
        # 1 - tensor_wise scaling
        # 2 - channel_wise scaling
        # 3 - bireal (channel-wise scaling, but use clamp of the fp_weight for backward)

        if not scale == -1:
            fp_weight.data.clamp_(
                -1, 1
            )  # clamp the real-weights first, need to discover whether to use it, birealnet didnt use this
        if scale == 0:
            mean_val = fp_weight.abs().mean()
            bi_weight = fp_weight.sign()
        elif scale == 1:
            mean_val = fp_weight.abs().mean()
            bi_weight = fp_weight.sign() * mean_val
        elif scale == 2 or scale == 3:
            # mean_val = torch.mean(torch.mean(torch.mean(abs(fp_weight),dim=3),dim=2,keepdim=True),dim=1,keepdim=True)  # the scaling factor calc in bireal's implemention
            mean_val = (
                fp_weight.abs().view(fp_weight.shape[0], -1).mean(1, keepdim=True)
            )
            bi_weight = fp_weight.sign() * mean_val.view(-1, 1, 1, 1)
        elif scale == -1:
            # normally not used, no binarize for weight
            mean_val = fp_weight.abs().mean()
            bi_weight = fp_weight
        else:
            raise NotImplementedError(
                "the scale method for binary is not implemented yet."
            )
        ctx.save_for_backward(old_x, fp_weight, bi_weight, mean_val, scale, STE_method)
        return x, bi_weight

    @staticmethod
    def backward(ctx, g_x, g_bi_weight):
        x, fp_weight, bi_weight, mean_val, scale, STE_method = ctx.saved_tensors
        clip_value = 1.3
        g_x[x.ge(clip_value)] = 0
        g_x[x.le(-clip_value)] = 0
        if scale == 0:
            g_bi_weight[fp_weight.ge(clip_value)] = 0
            g_bi_weight[fp_weight.le(-clip_value)] = 0
            g_fp_weight = g_bi_weight
        elif scale == 1:
            g_bi_weight[fp_weight.ge(clip_value * mean_val)] = 0
            g_bi_weight[fp_weight.le(-clip_value * mean_val)] = 0
            g_fp_weight = g_bi_weight
        elif scale == 2:
            proxy = fp_weight.abs().sign()
            proxy[fp_weight.abs() > 1] = 0  # mask out gradients
            binary_grad = g_bi_weight * mean_val.view(-1, 1, 1, 1) * proxy

            mean_grad = bi_weight.data.sign() * g_bi_weight
            mean_grad = mean_grad.view(mean_grad.shape[0], -1).mean(1).view(-1, 1, 1, 1)
            mean_grad = mean_grad * bi_weight.data.sign()

            g_fp_weight = binary_grad + mean_grad
            g_fp_weight = (
                g_fp_weight * fp_weight[0].nelement() * (1 - 1 / fp_weight.size(1))
            )  # gradients w.r.t full-precision weights before centralizing
        elif scale == 3:
            # in original bireal implemention
            # clipped_weights = torch.clamp(real_weights, 1.0, -1.0) as the binary_weight's backward
            g_bi_weight[fp_weight.ge(1.0)] = 0
            g_bi_weight[fp_weight.le(-1.0)] = 0
            g_fp_weight = g_bi_weight

        elif scale == -1:
            # use full precision weight, no need to process binary-weight's grad
            g_fp_weight = g_bi_weight
        else:
            raise NotImplementedError(
                "the scale method for binary is not implemented yet."
            )
        return g_x, g_fp_weight, None, None


class BinaryConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        dilation=1,
        # the binariy related cfgs, should be fed in with binary-cfg-dict
        dropout_ratio=0,  # TODO: implement droppath
        bi_w_scale=2,  # the weight-scaling
        bi_act_method=0,  # the type of bi-activation
        bias=False,# w/o bias
        STE_method=0,
    ):
        super(BinaryConv2d, self).__init__()
        (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.dropout_ratio,
            self.bi_w_scale,
            self.bi_act_method,
            self.use_bias,
            self.STE_method,
        ) = (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dropout_ratio,
            bi_w_scale,
            bi_act_method,
            bias,
            STE_method,
        )
        self.full_precision = nn.Parameter(
            torch.zeros(
                [
                    self.out_channels,
                    self.in_channels // self.groups,
                    self.kernel_size,
                    self.kernel_size,
                ]
            )
        )
        torch.nn.init.xavier_normal_(self.full_precision.data)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros([self.out_channels]).cuda())
        else:
            self.bias = None
        # self.offset = self.full_precision.shape[1]*self.full_precision.shape[2]*self.full_precision.shape[3]

    def forward(self, x):
        # x, bi_weight = self.binarize(x, self.full_precision, torch.tensor([self.scale]), torch.tensor(self.method))
        binarize_cfgs = {
            "bi_w_scale": torch.tensor([self.bi_w_scale]),
            "bi_act_method": torch.tensor([self.bi_act_method]),
            "STE_method": torch.tensor([self.STE_method]),
        }
        x, bi_weight = Binarize.apply(x, self.full_precision, binarize_cfgs)
        x = F.conv2d(
            x,
            bi_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        # x = (x+self.offset) / 2
        return x


class SkipConnectV2(nn.Module):
    """
    plain double concat of avgpool-2d & concat when stride==2
    supported cases:
        - stride=1/2 | c_in == c_out
        - stride=2 | c_in == 2*c_out
    """

    def __init__(self, C, C_out, stride, affine, conv_ds=False, conv_ds_mode="normal"):
        super(SkipConnectV2, self).__init__()
        self.stride = stride
        self.conv_ds = conv_ds
        self.expansion = C_out // C
        self.conv_ds_mode = conv_ds_mode
        if stride == 2:
            if self.conv_ds and self.conv_ds_mode == "normal":
                # support arbitary chs
                self.op1 = nn.AvgPool2d(2)
                self.op2 = XNORGroupConv(
                    C,
                    C_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    affine=affine,
                    shortcut=False,
                )
            elif not self.conv_ds:
                assert C_out == 2 * C or C_out == C
                self.op1 = nn.AvgPool2d(2)
                if self.expansion == 2:
                    self.op2 = nn.AvgPool2d(2)

        if stride == 1:
            self.op = Identity()

    def forward(self, x):
        if self.stride == 1:
            x1 = self.op(x)
            return x1
        elif self.stride == 2:
            if self.conv_ds:
                x1 = self.op1(x)
                x2 = self.op2(x1)
                return x2
            else:
                if self.expansion == 2:
                    x1 = self.op1(x)
                    x2 = self.op2(x)
                    return torch.cat([x1, x2], 1)
                elif self.expansion == 1:
                    x1 = self.op1(x)
                    return x1


"""
xnor-conv-bn-relu should be the basic building block and contain all the possiible ops

It should contain the args:
    - ch disalignment arrangement
    - shortcut_op_type(simple parameter-free shorcut)
    - reduction_op_type(avgpool2d / factorized reduce)
    - layer-order
    - relu
    - group
    - dilation
    - binary_conv_cfgs
        - bi_w_scale
        - bi_act_method
        - bias(zero-mean)

"""


class BinaryConvBNReLU(nn.Module):
    """the basic binary conv bn relu block"""

    def __init__(
        # basic conv-bn-relu interfaces
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        affine=True,
        relu=True,
        # conv-related interfaces to support different types of conv
        # expansion=1, # FIXME: expansion should be calulated with the chs
        group=1,
        dilation=1,
        # binary layers cfgs
        shortcut=True,
        shortcut_op_type="simple",  # simple/conv-op
        reduction_op_type="conv",  # conv/factorized
        layer_order="conv_bn_relu",
        binary_conv_cfgs={
            "bi_w_scale": 1,
            "bi_act_method": 0,
            "bias": False,
            "STE_method": 0,
        },
    ):
        super(BinaryConvBNReLU, self).__init__()
        (
            self.C_in,
            self.C_out,
            self.kernel_size,
            self.stride,
            self.padding,
            self.affine,
            self.relu,
            self.group,
            self.dilation,
            self.use_shortcut,
            self.shortcut_op_type,
            self.reduction_op_type,
            self.layer_order,
            self.binary_conv_cfgs,
        ) = (
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            affine,
            relu,
            group,
            dilation,
            shortcut,
            shortcut_op_type,
            reduction_op_type,
            layer_order,
            binary_conv_cfgs,
        )

        self.expansion = self.C_out // self.C_in
        """
        currently only support bianry conv with the same size(stride==1 c_in==c_out)
        or with the standard reduction practice(stride==2 c_out=2*c_in)
        in other cases, shortcut should be False / or no shortcut will be applied
        """
        if stride == 1:
            if shortcut:
                assert self.expansion == 1
        elif stride == 2:
            if shortcut:
                assert self.expansion == 2 or self.expansion == 1

        assert self.group == 1
        assert self.dilation == 1

        group_dim = C_out // stride
        assert padding == int((kernel_size - 1) / 2)
        # the 'relu' in layer_order is just a placeholder, when relu is false, no relu will be applied regardless of its name in layer_order
        assert self.layer_order in ["conv_bn_relu", "bn_conv_relu"]
        # supported shortcut_op_type
        #  - simple: "identity" for normal and avgpool2d for strided(2x in ch to meet width)
        #  - conv: use another conv op for shortcut
        # assert self.shortcut_op_type == "simple" or "conv" in self.shortcut_op_type
        assert self.shortcut_op_type in ["simple"]
        #  currently donot support conv ds for basic BinaryConvBNReLU, add it outside instead
        assert self.reduction_op_type in ["conv", "factorized"]

        # Initialize OPs
        self.bn = nn.BatchNorm2d(
            C_in if self.layer_order == "bn_conv_relu" else C_out,
            affine=True,
            momentum=0.9,
            eps=1e-05
        )
        self.relu = nn.ReLU(inplace=False) if self.relu else None
        if self.stride == 2 and self.reduction_op_type == "factorized":
            self.convs = [
                BinaryConv2d(
                    C_in,
                    group_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=group,
                    bias=False,
                    #**self.binary_conv_cfgs,
                )
                for _ in range(stride)
            ]
            self.convs = nn.ModuleList(self.convs)
        else:
            self.conv = BinaryConv2d(
                C_in,
                C_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=group,
                dilation=dilation,
                **self.binary_conv_cfgs,
            )
        # shortcut ops
        if self.shortcut_op_type == "simple":
            # SkipConnectV2 is identity() when stride==1 and avgpool with ch duplicate & concat with stride==2
            self.shortcut = SkipConnectV2(
                C_in,
                C_out,
                stride,
                affine,
                conv_ds=False,
            )
        else:
            raise AssertionError(
                "Currently base BinaryConvBNReLU only support simple shortcut"
            )

    def forward(self, x):

        if self.layer_order == "bn_conv_relu":
            x = self.bn(x)

        if self.stride == 2 and self.reduction_op_type == "factorized":
            mod = x.size(2) % self.stride
            if mod != 0:
                pad = self.stride - mod
                x = F.pad(x, (pad, 0, pad, 0), "constant", 0)
            out_l = []
            for i, conv in enumerate(self.convs):
                out_ = conv(x[:, :, i:, i:])
                out_l.append(out_)
            out = torch.cat(out_l, dim=1)
            if self.use_shortcut:
                out += self.shortcut(x)
        else:
            out = self.conv(x)
            if self.use_shortcut:
                out += self.shortcut(x)

        if self.layer_order == "conv_bn_relu":
            out = self.bn(out)
        if self.relu:
            out = self.relu(out)
        return out


class ResNetDownSample(nn.Module):
    def __init__(self, stride):
        super(ResNetDownSample, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class BinaryResNetBlock(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        affine,
        relu=True,
        kernel_size=3,
        downsample="conv",
        # --- the binary cfgs ---
        binary_cfgs={},
    ):
        super(BinaryResNetBlock, self).__init__()
        (
            self.C_in,
            self.C_out,
            self.stride,
            self.affine,
            self.relu,
            self.kernel_size,
            self.downsample,
            self.binary_cfgs,
        ) = (
            C_in,
            C_out,
            stride,
            affine,
            relu,
            kernel_size,
            downsample,
            binary_cfgs,
        )
        padding = int((kernel_size - 1) / 2)
        self.op_1 = BinaryConvBNReLU(
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            affine,
            relu,
            **binary_cfgs,
        )
        self.op_2 = BinaryConvBNReLU(
            C_out,
            C_out,
            kernel_size,
            1,  # the 2nd block keep the same dim
            padding,
            affine,
            relu,
            **binary_cfgs,
        )
        # define the skip_op
        assert downsample in ["conv", "avgpool", "binary_conv"]
        if downsample == "conv":
            self.skip_op = (
                Identity()
                if stride == 1
                else ConvBNReLU(C_in, C_out, 1, stride, 0, affine=affine)
            )
        elif downsample == "avgpool":
            self.skip_op = Identity() if stride == 1 else ResNetDownSample(stride)
        elif downsample == "binary_conv":
            self.skip_op = (
                Identity()
                if stride == 1
                else BinaryConvBNReLU(
                    C_in, C_out, 3, stride, 1, affine=affine, **binary_cfgs
                )
            )

    def forward(self, inputs):
        inner = self.op_1(inputs) + self.skip_op(inputs)
        out = self.op_2(inner) + inner
        return out


# this should align with the defalut values when defining the `BinaryConvBNReLU`
xnor_ResNet_cfgs = {
    "shortcut": True,
    "shortcut_op_type": "simple",
    "reduction_op_type": "factorized",
    # "layer_order": "conv_bn_relu",
    "layer_order": "bn_conv_relu",
    "binary_conv_cfgs": {
        "bi_w_scale": 2,
        "bi_act_method": 0,
        "bias": False,
        "STE_method": 1,
    },
}

register_primitive(
    "xnor_resnet_block",
    lambda C, C_out, stride, affine: BinaryResNetBlock(
        C,
        C_out,
        stride=stride,
        affine=affine,
        relu=True,
        #downsample="conv",
        binary_cfgs=xnor_ResNet_cfgs,
    ),
)

bireal_binary_cfgs = {
    "shortcut": True,
    "shortcut_op_type": "simple",
    "reduction_op_type": "conv",
    "layer_order": "conv_bn_relu",
    "binary_conv_cfgs": {
        "bi_w_scale": 3,
        "bi_act_method": 3,
        "bias": False,
    },
}

register_primitive(
    "bireal_resnet_block",
    lambda C, C_out, stride, affine: BinaryResNetBlock(
        C,
        C_out,
        stride=stride,
        affine=affine,
        relu=True,
        downsample="conv",
        binary_cfgs=bireal_binary_cfgs,
    ),
)


class BinaryVggBlock(nn.Module):
    """
    the binary vgg block
    the skip_op is avgpool2d for stride=2 & idenity for stride=1
    """

    def __init__(
        self,
        C_in,
        C_out,
        stride,
        affine,
        relu=True,
        kernel_size=3,
        downsample="conv",
        binary_cfgs={},
    ):
        super(BinaryVggBlock, self).__init__()
        (
            self.C_in,
            self.C_out,
            self.stride,
            self.affine,
            self.relu,
            self.kernel_size,
            self.downsample,
            self.binary_cfgs,
        ) = (
            C_in,
            C_out,
            stride,
            affine,
            relu,
            kernel_size,
            downsample,
            binary_cfgs,
        )
        binary_cfgs["shortcut"] = False  # vgg block has no shortcut
        padding = int((kernel_size - 1) / 2)
        self.op = BinaryConvBNReLU(
            C_in,
            C_out,
            kernel_size,
            1,  # stride=1
            padding,
            affine,
            relu,
            **binary_cfgs,
        )
        # assert downsample in ["conv","avgpool","binary_conv"]
        assert downsample == "avgpool"
        if stride == 2:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif stride == 1:
            self.pool = nn.Identity()

    def forward(self, inputs):
        return self.pool(self.op(inputs))


register_primitive(
    "xnor_vgg_block",
    lambda C, C_out, stride, affine: BinaryVggBlock(
        C,
        C_out,
        stride=stride,
        affine=affine,
        relu=True,
        downsample="avgpool",
        binary_cfgs=binary_cfgs,
    ),
)

register_primitive(
    "bireal_vgg_block",
    lambda C, C_out, stride, affine: BinaryVggBlock(
        C,
        C_out,
        stride=stride,
        affine=affine,
        relu=True,
        downsample="avgpool",
        binary_cfgs=bireal_binary_cfgs,
    ),
)


# ============ register the base binary operations ==============

binary_cfgs = {
    "shortcut": True,
    "shortcut_op_type": "simple",
    "reduction_op_type": "factorized",
    # "layer_order": "conv_bn_relu",
    "layer_order": "bn_conv_relu",
    "binary_conv_cfgs": {
        "bi_w_scale": 1,
        "bi_act_method": 0,
        "bias": False,
    },
}

binary_cfgs_no_shortcut = binary_cfgs
binary_cfgs_no_shortcut["shortcut"] = False

register_primitive(
    "xnor_conv_3x3_noskip",
    lambda C, C_out, stride, affine: BinaryConvBNReLU(
        C, C_out, 3, stride, 1, affine=affine, **binary_cfgs_no_shortcut
    ),
)

register_primitive(
    "xnor_conv_1x1_noskip",
    lambda C, C_out, stride, affine: BinaryConvBNReLU(
        C, C_out, 1, stride, 0, affine=affine, **binary_cfgs_no_shortcut
    ),
)

# ====== register the candidate operations for binary search ======

# use factorized when stride==2
binary_cfgs = {
    "shortcut": True,
    "shortcut_op_type": "simple",
    "reduction_op_type": "factorized",
    # "layer_order": "conv_bn_relu",
    "layer_order": "bn_conv_relu",
    "binary_conv_cfgs": {
        "bi_w_scale": 1,
        "bi_act_method": 0,
        "bias": False,
    },
}

binary_cfgs_no_shortcut = binary_cfgs
binary_cfgs_no_shortcut["shortcut"] = False

# the origianl version of xnor_conv_3x3 conditional skip, when encounter stride=2 conv, no shortcut is applied
register_primitive(
    "xnor_conv_3x3_cond_skip_v0",
    lambda C_in, C_out, stride, affine: BinaryConvBNReLU(
        C_in, C_out, 3, stride, 1, affine=affine, relu=True, **binary_cfgs
    )
    if stride == 1
    else BinaryConvBNReLU(
        C_in, C_out, 3, stride, 1, affine=affine, relu=True, **binary_cfgs_no_shortcut
    ),
)

# newer version of the xnor_conv_3x3 conditional skip, when encounter stride=2 conv(assert that the c-out=2*c-in), use factorized-reduce(could be with shortcut)
register_primitive(
    "xnor_conv_3x3_cond_skip",
    lambda C, C_out, stride, affine: BinaryConvBNReLU(
        C, C_out, 3, stride, 1, affine=affine, **binary_cfgs
    ),
)

# the skip connection op for binary search, identity() when stride=2, duplicate and concat when expansion & stride == 2
register_primitive(
    "xnor_skip_connect",
    lambda C, C_out, stride, affine: SkipConnectV2(
        C, C_out, stride, affine, conv_ds=False
    ),
)

# ==== register the cell-wise skip-connect op ====

binary_cfgs = {
    "shortcut": False,
    "shortcut_op_type": "simple",
    "reduction_op_type": "factorized",
    # "layer_order": "conv_bn_relu",
    "layer_order": "bn_conv_relu",
    "binary_conv_cfgs": {
        "bi_w_scale": 1,
        "bi_act_method": 0,
        "bias": False,
    },
}

# plain shortcut when stride == 1, and stride=2 binary conv with no shortcut
# for cell-wise shortcut
register_primitive(
    "xnor_conv_3x3_skip_connect",
    lambda C, C_out, stride, affine: Identity()
    if stride == 1
    else BinaryConvBNReLU(C, C_out, 3, stride, 1, affine=affine, **binary_cfgs),
)

# ---- binary NIN ----
class NIN(nn.Module):
    def __init__(self, C, C_out, stride, affine, binary_cfgs=binary_cfgs):

        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True)

        self.conv2_1 = XNORConvBNReLU(
            192, 160, kernel_size=1, stride=1, padding=0, **binary_cfgs
        )
        self.conv2_2 = XNORConvBNReLU(
            160, 96, kernel_size=1, stride=1, padding=0, **binary_cfgs
        )

        self.conv3_1 = XNORConvBNReLU(
            96,
            192,
            kernel_size=5,
            stride=1,
            padding=2,
            **binarize_cfgs,
        )
        self.conv3_2 = XNORConvBNReLU(
            192,
            192,
            kernel_size=1,
            stride=1,
            padding=0,
            **binarize_cfgs,
        )
        self.conv3_3 = XNORConvBNReLU(
            192,
            192,
            kernel_size=1,
            stride=1,
            padding=0,
            **binarize_cfgs,
        )

        self.conv4_1 = XNORConvBNReLU(
            192,
            192,
            kernel_size=3,
            stride=1,
            padding=1,
            **binarize_cfgs,
        )
        self.conv4_2 = XNORConvBNReLU(
            192,
            192,
            kernel_size=1,
            stride=1,
            padding=0,
            **binarize_cfgs,
        )

        self.bn5 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True)
        self.conv5 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m.weight, "data"):
                    m.weight.data.zero_().add_(1.0)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(m.weight, "data"):
                    m.weight.data.clamp_(min=0.01)

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = F.relu(self.conv5(self.bn5(x)))
        x = F.avg_pool2d(x, kernel_size=8)
        x = x.squeeze()
        return x


register_primitive("xnor_nin", NIN)
