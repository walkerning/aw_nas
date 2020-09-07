"""
BNN operations.
"""
# pylint: disable=arguments-differ,useless-super-delegation,invalid-name

import torch
from torch import nn
import torch.nn.functional as F

from aw_nas.ops import register_primitive, ConvBNReLU, Identity

# ---- binary activation ----
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        inputs = inputs.sign()
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        return grad_input


class Fp32Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        return grad_input


# ---- XNOR modules ----
class xnor_binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fp_weight, fp32_act, scale):

        # fp_weight.data = fp_weight.data - fp_weight.data.mean(1, keepdim=True)
        fp_weight.data.clamp_(-1, 1)

        old_x = x
        if fp32_act:
            x = Fp32Activation.apply(x)
        else:
            x = BinaryActivation.apply(x)
        # scale

        if scale:
            mean_val = fp_weight.abs().view(fp_weight.shape[0], -1).mean(1, keepdim=True)
            bi_weight = fp_weight.sign() * mean_val.view(-1, 1, 1,1)
        else:
            mean_val = fp_weight.abs().mean()
            # bi_weight = fp_weight.sign() * mean_val
            bi_weight = fp_weight.sign()

        ctx.save_for_backward(old_x, fp_weight, bi_weight, mean_val, scale)
        return x, bi_weight

    @staticmethod
    def backward(ctx, g_x, g_bi_weight):
        x, fp_weight, bi_weight, mean_val, scale = ctx.saved_tensors
        clip_value=1.3
        g_x[x.ge(clip_value)] = 0
        g_x[x.le(-clip_value)] = 0

        if scale:
            proxy = fp_weight.abs().sign()
            proxy[fp_weight.abs() > 1] = 0 # mask out gradients
            binary_grad = g_bi_weight * mean_val.view(-1, 1, 1, 1) * proxy

            mean_grad = bi_weight.data.sign() * g_bi_weight
            mean_grad = mean_grad.view(mean_grad.shape[0], -1).mean(1).view(-1, 1, 1, 1)
            mean_grad = mean_grad * bi_weight.data.sign()

            g_fp_weight = binary_grad + mean_grad
            g_fp_weight = (
                g_fp_weight * fp_weight[0].nelement() * (1 - 1 / fp_weight.size(1))
            ) # gradients w.r.t full-precision weights before centralizing
        else:
            # g_bi_weight[bi_weight.ge(clip_value*mean_val)] = 0
            # g_bi_weight[bi_weight.le(-clip_value*mean_val)] = 0
            g_bi_weight[bi_weight.ge(clip_value)] = 0
            g_bi_weight[bi_weight.le(-clip_value)] = 0
            g_fp_weight = g_bi_weight

        return g_x, g_fp_weight, None, None


class XNORConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        dilation=1,
        dropout_ratio=0,
        fp32_act=False,
        scale=False,
    ):
        super(XNORConv2d, self).__init__()
        (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.dropout_ratio,
            self.fp32_act,
            self.scale,
        ) = (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dropout_ratio,
            fp32_act,
            scale
        )
        self.binarize = xnor_binarize.apply
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
        self.full_precision.data.normal_(0, 0.05)
        self.bias = nn.Parameter(torch.zeros([self.out_channels]).cuda())

    def forward(self, x):
        x, bi_weight = self.binarize(x, self.full_precision, self.fp32_act, torch.tensor([self.scale]))
        x = F.conv2d(
            x, bi_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        return x


class XNORFactorizedReduce(nn.Module):
    def __init__(self, C, C_out, stride, affine):
        super(XNORFactorizedReduce, self).__init__()
        assert C_out == 2*C
        assert stride == 2
        kernel_size=3
        padding=1
        # self.op1 = nn.AvgPool2d(2)
        self.op2 = XNORGroupConv(C, C, kernel_size, 2, padding, affine)
        self.op3 = XNORGroupConv(C, C, kernel_size, 2, padding, affine)

    def forward(self, x):
        # x = self.op1(x)
        x1 = self.op2(x)
        x2 = self.op3(x)
        return torch.cat([x1,x2],1)

# the plain xnor-conv without shortcut(also surpport different channel size)
class XNORConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        affine=True,
        groups=1,
        dropout_ratio=0,
        fp32_act=False,
    ):
        super(XNORConvBNReLU, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1, affine=True)
        self.conv = XNORConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dropout_ratio,
            fp32_act,
        )

    def forward(self, x):
        out = self.bn(x)
        out = self.conv(out)
        out = F.relu(out)
        return out


class XNORGroupConv(nn.Module):
    # the XNOR group conv for binaryNN search

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, expansion=1, group=1, shortcut=True): # FIXME: default using the shortcut for binary op may not be proper
        super(XNORGroupConv, self).__init__()
        if shortcut: # only use shortcut when C-in and C-out are the same
            self.use_shortcut = True
        else:
            self.use_shortcut = False
        if self.use_shortcut:
            if expansion is not 1:
                raise Exception("when enabling shortcut, so donot support expansion")
            if C_in != C_out:
                raise Exception("when enabling shortcut, only surpport c-in == c-out")
            if stride == 1:
                self.shortcut = Identity()
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=2)
        else:
            pass
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_in*expansion, affine=affine),
            XNORConv2d(C_in, C_out*expansion, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=group),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        if self.use_shortcut:
            out = self.op(x)+self.shortcut(x)
        else:
            out = self.op(x)
        return out

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)

class XNORDilConv(nn.Module):
    # the XNOR group conv for binaryNN search

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, expansion=1, group=1):
        super(XNORDilConv, self).__init__()
        if expansion is not 1:
            raise Exception("binary block has inner connection, so donot support expansion")
        if stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = nn.AvgPool2d(kernel_size=2) # maybe avgpool2d will cause grad loss
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_in*expansion, affine=affine),
            XNORConv2d(C_in, C_in*expansion, kernel_size=kernel_size, stride=stride,
                      dilation=dilation , padding=padding, groups=group),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        # FIXME: not sure how to add shortcut with reduction of dilated conv
        return self.op(x)+self.shortcut(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)

register_primitive("xnor_conv_3x3",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 3, stride, 1, affine=affine, group=1),
)

register_primitive("xnor_conv_3x3_noskip",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 3, stride, 1, affine=affine, group=1, shortcut=False),
)
register_primitive("cond_xnor_conv_3x3_noskip",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 3, stride, 1, affine=affine, group=1, shortcut=False)\
                   if stride > 1 or C != C_out else Identity()
)

register_primitive("xnor_conv_5x5",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 5, stride, 2, affine=affine, group=1),
)

register_primitive("xnor_conv_5x5_noskip",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 5, stride, 2, affine=affine, group=1, shortcut=False),
)

register_primitive("xnor_conv_1x1",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 1, stride, 0, affine=affine, group=1),
)

register_primitive("xnor_group_conv_3x3",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 3, stride, 1, affine=affine, group=3),
)

register_primitive("xnor_group_conv_5x5",
                   lambda C, C_out, stride, affine: XNORGroupConv(
                       C, C_out, 5, stride, 2, affine=affine, group=3),
)

register_primitive("xnor_dil_conv_3x3",
                   lambda C, C_out, stride, affine: XNORDilConv(
                       C, C_out, 3, stride, 2, 2, affine=affine),
)

register_primitive("xnor_dil_conv_5x5",
                   lambda C, C_out, stride, affine: XNORDilConv(
                       C, C_out, 5, stride, 4, 2, affine=affine),
)

# the preporcess node in the supernet, without the shortcut, and c-in != c-out
register_primitive("xnor_conv_bn_3x3",
                    lambda C, C_out, stride, affine: XNORConvBNReLU(
                        C, C_out, 3, stride, 1, affine=affine),
)

register_primitive("xnor_conv_bn_1x1",
                    lambda C, C_out, stride, affine: XNORConvBNReLU(
                        C, C_out, 1, stride, 0, affine=affine),
)

register_primitive("xnor_factorized_reduce",
                   lambda C, C_out, stride, affine: XNORFactorizedReduce(
                       C, C_out, stride, affine),
)


# ---- the bireal modules ---
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BirealActivation(nn.Module):
    def __init__(self):
        super(BirealActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
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
        out = out_forward.detach() - x.detach() + x

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, C_in, c_out, kernel_size=3, stride=1, padding=1, bias=None):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.number_of_weights = C_in * c_out * kernel_size * kernel_size
        self.shape = (c_out, C_in, kernel_size, kernel_size)
        self.weights = nn.Parameter(
            torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True
        )

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = real_weights.abs().mean().reshape([1, 1, 1, 1]).detach()
        binary_weights_no_grad = scaling_factor * BinaryActivation.apply(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = (
            binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        )
        y = F.conv2d(
            x, binary_weights, stride=self.stride, padding=self.padding, bias=self.bias
        )

        return y


class BinaryConvBNReLU(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        affine,
        bias=None,
        activation=BinaryActivation,
    ):
        super(BinaryConvBNReLU, self).__init__()
        self.bi_conv = HardBinaryConv(
            C_in,
            C_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bi_act = activation
        self.op = nn.Sequential(
            self.bi_act, self.bi_conv, nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)


#  ----- DoReFa modules -------
def uniform_quantize(k):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            if k == 32:
                out = inputs
            elif k == 1:
                out = torch.sign(inputs)
            else:
                n = float(2 ** k - 1)
                out = torch.round(inputs * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)
        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
        return activation_q


class DorefaConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        w_bit=1,
        a_bit=32,
    ):
        super(DorefaConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            w_bit,
        )
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, inputs, order=None):
        weight_q = self.quantize_fn(self.weight)
        return F.conv2d(
            inputs,
            weight_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class DorefaConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        affine=True,
        groups=1,
    ):
        super(DorefaConvBNReLU, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1, affine=True)
        self.conv = DorefaConv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups
        )
        self.act = activation_quantize_fn(a_bit=self.conv.a_bit)

    def forward(self, x):
        x = self.act(x)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return out


# ----- binary resnet blocks: BiReal, XNOR, DoReFa ------
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
        C,
        C_out,
        stride,
        affine,
        kernel_size=3,
        block="bireal",
        act=BinaryActivation,
        downsample="conv",
        fp32_act=False,
    ):
        super(BinaryResNetBlock, self).__init__()
        self.stride = stride
        padding = int((kernel_size - 1) / 2)
        self.activation = act

        if block == "bireal":
            self.op_1 = BinaryConvBNReLU(
                C,
                C_out,
                kernel_size,
                stride,
                padding,
                affine=affine,
                bias=None,
                activation=self.activation,
            )
            self.op_2 = BinaryConvBNReLU(
                C_out,
                C_out,
                kernel_size,
                1,
                padding,
                affine=affine,
                bias=None,
                activation=self.activation,
            )
        elif block == "xnor":
            self.op_1 = XNORConvBNReLU(
                C,
                C_out,
                kernel_size,
                stride,
                padding,
                affine=affine,
                groups=1,
                dropout_ratio=0,
                fp32_act=fp32_act,
            )
            self.op_2 = XNORConvBNReLU(
                C_out,
                C_out,
                kernel_size,
                1,
                padding,
                affine=affine,
                groups=1,
                dropout_ratio=0,
                fp32_act=fp32_act,
            )
        elif block == "dorefa":
            self.op_1 = DorefaConvBNReLU(
                C, C_out, kernel_size, stride, padding, affine=affine, groups=1
            )
            self.op_2 = DorefaConvBNReLU(
                C_out, C_out, kernel_size, 1, padding, affine=affine, groups=1
            )
        if downsample == "conv":
            self.skip_op = (
                Identity()
                if stride == 1
                else ConvBNReLU(C, C_out, 1, stride, 0, affine=affine)
            )
        elif downsample == "avgpool":
            self.skip_op = Identity() if stride == 1 else ResNetDownSample(stride)

    def forward(self, inputs):
        inner = self.op_1(inputs) + self.skip_op(inputs)
        out = self.op_2(inner) + inner
        return out

register_primitive(
    "bireal_resnet_block",
    lambda C, C_out, stride, affine: BinaryResNetBlock(
        C, C_out, stride=stride, affine=affine, block="bireal", act=BirealActivation()
    ),
)

register_primitive(
    "xnor_resnet_block",
    lambda C, C_out, stride, affine: BinaryResNetBlock(
        C, C_out, stride=stride, affine=affine, block="xnor", fp32_act=False
    ),
)

register_primitive(
    "dorefa_resnet_block",
    lambda C, C_out, stride, affine: BinaryResNetBlock(
        C, C_out, stride=stride, affine=affine, block="dorefa"
    ),
)

register_primitive(
    "xnor_resnet_block_pool_downsample",
    lambda C, C_out, stride, affine: BinaryResNetBlock(
        C, C_out, stride=stride, affine=affine, block="xnor", downsample="avgpool"
    ),
)

# --- binary vgg blocks: BiReal, XNOR ----
class BinaryVggBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, block="xnor"):
        super(BinaryVggBlock, self).__init__()
        if block == "xnor":
            self.op = XNORConvBNReLU(
                C, C_out, 3, 1, 1, affine=affine, groups=1, dropout_ratio=0
            )
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, inputs):
        return self.pool(self.op(inputs))


register_primitive(
    "bireal_vgg_block",
    lambda C, C_out, stride, affine: BinaryVggBlock(
        C, C_out, stride, affine=affine, block="bireal"
    ),
)
register_primitive(
    "xnor_vgg_block",
    lambda C, C_out, stride, affine: BinaryVggBlock(
        C, C_out, stride, affine=affine, block="xnor"
    ),
)

# ---- binary NIN ----
class NIN(nn.Module):
    def __init__(self, C, C_out, stride, affine):
        super(NIN, self).__init__()

        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True)

        self.conv2_1 = XNORConvBNReLU(192, 160, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = XNORConvBNReLU(160, 96, kernel_size=1, stride=1, padding=0)

        self.conv3_1 = XNORConvBNReLU(
            96, 192, kernel_size=5, stride=1, padding=2, dropout_ratio=0.5
        )
        self.conv3_2 = XNORConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = XNORConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)

        self.conv4_1 = XNORConvBNReLU(
            192, 192, kernel_size=3, stride=1, padding=1, dropout_ratio=0.5
        )
        self.conv4_2 = XNORConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)

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
