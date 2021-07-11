import torch
from torch import nn
import torch.nn.functional as F
from aw_nas.ops import register_primitive, ConvBNReLU, Identity, SEModule, get_op, BNReLU
from aw_nas.utils import make_divisible, drop_connect

class VggBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine):
        super(VggBlock, self).__init__()
        self.conv_bn = nn.Sequential(nn.Conv2d(C, C_out, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(C_out, affine=affine))
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, inputs):
        return self.pool(F.relu(self.conv_bn(inputs)))

    def forward_one_step(self, context=None, inputs=None):
        _, op_ind = context.next_op_index
        if op_ind == 0:
            assert inputs is not None
            out = self.conv_bn(inputs)
            context.current_op.append(out)
            context.last_conv_module = self.conv_bn[0]
        elif op_ind == 1:
            out = self.pool(F.relu(context.current_op[-1]))
            context.previous_op.append(out)
            context.current_op = []
            context.flag_inject(False)
        else:
            raise Exception("Unexpected op_ind")
        return out, context

class MobileNetBlock(nn.Module):
    def __init__(self, expansion, C, C_out, stride, affine, kernel_size=3, relu6=False):
        super(MobileNetBlock, self).__init__()
        C_inner = self.C_inner = make_divisible(expansion * C, 8)
        self.stride = stride
        self.relu6 = relu6
        self.activation = F.relu6 if self.relu6 else F.relu

        self.conv1 = nn.Conv2d(C, C_inner,
                               kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(C_inner, affine=affine)
        padding = int((kernel_size - 1) / 2)
        self.conv2 = nn.Conv2d(C_inner, C_inner,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=C_inner, bias=False)
        self.bn2 = nn.BatchNorm2d(C_inner, affine=affine)
        self.conv3 = nn.Conv2d(C_inner, C_out,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(C_out, affine=affine)

        self.shortcut = nn.Sequential()
        self.has_conv_shortcut = False
        if stride == 1 and C != C_out:
            self.has_conv_shortcut = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(C, C_out, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, inputs):
        out = self.activation(self.bn1(self.conv1(inputs)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(inputs) if self.stride == 1 else out
        return out

    def forward_one_step(self, context=None, inputs=None):
        activation = F.relu6 if self.relu6 else F.relu
        _, op_ind = context.next_op_index
        if op_ind == 0:
            assert inputs is not None
            context.current_op.append(inputs) # save inputs
            out = self.bn1(self.conv1(inputs))
            context.current_op.append(out)
            context.last_conv_module = self.conv1
        elif op_ind == 2:
            out = self.bn2(self.conv2(activation(context.current_op[-1])))
            context.current_op.append(out)
            context.last_conv_module = self.conv2
        elif op_ind == 3:
            out = self.bn3(self.conv3(activation(context.current_op[-1])))
            context.current_op.append(out)
            if not self.stride == 1:
                # return out
                context.previous_op.append(out)
                context.current_op = []
                context.last_conv_module = self.conv3
            elif not self.has_conv_shortcut:
                # return out + shortcut
                context.previous_op.append(out + self.shortcut(context.current_op[0]))
                context.last_conv_module = self.conv3
                context.current_op = []
        elif op_ind == 4:
            # has_conv_shortcut
            out = self.shortcut(context.current_op[0])
            context.current_op.append(out)
            context.last_conv_module = self.shortcut[0]
        elif op_ind == 5:
            out = context.current_op[-1] + context.current_op[-2]
            context.previous_op.append(out)
            context.current_op = []
            context.flag_inject(False)
        else:
            raise Exception("Unexpected op_ind")
        return out, context

class ResNetBlockSplit(nn.Module):
    def __init__(self, C, C_out, stride, affine, act='relu'):
        super(ResNetBlockSplit, self).__init__()
        self.act = act
        self.op_1_1 = ConvBNReLU(C, C_out,
                                 3, stride, 1, affine=affine, relu=False)
        self.op_1_2 = ConvBNReLU(C, C_out,
                                 3, stride, 1, affine=affine, relu=False)
        self.op_2_1 = ConvBNReLU(C_out, C_out,
                                 3, 1, 1, affine=affine, relu=False)
        self.op_2_2 = ConvBNReLU(C_out, C_out,
                                 3, 1, 1, affine=affine, relu=False)
        self.skip_op = Identity() if stride == 1 else ConvBNReLU(C, C_out,
                                                                 1, stride, 0,
                                                                 affine=affine, relu=False)

    def forward(self, inputs):
        activation == F.relu
        if self.act == 'hardtanh':
            activation = F.hardtanh
        elif self.act == 'sigmoid':
            activation = F.sigmoid
        inner = activation(self.op_1_1(inputs) + self.op_1_2(inputs))
        out = self.op_2_1(inner) + self.op_2_2(inner)
        out_skip = self.skip_op(inputs)
        return activation(out + out_skip)

    def forward_one_step(self, context=None, inputs=None):
        activation == F.relu
        if self.act == 'hardtanh':
            activation = F.hardtanh
        elif self.act == 'sigmoid':
            activation = F.sigmoid
        _, op_ind = context.next_op_index
        if op_ind == 0:
            assert inputs is not None
            context.current_op.append(inputs) # save inputs
            out = self.op_1_1(inputs)
            context.current_op.append(out)
        elif op_ind == 2:
            out = self.op_1_2(context.current_op[0])
            context.current_op.append(out)
        elif op_ind == 3:
            out = activation(context.current_op[-1] + context.current_op[-2])
            context.current_op.append(out)
            context.flag_inject(False)
        elif op_ind == 4:
            out = self.op_2_1(context.current_op[-1])
            context.current_op.append(out)
        elif op_ind == 5:
            out = self.op_2_2(context.current_op[-2])
            context.current_op.append(out)
        else:
            assert op_ind == 6
            skip_out = self.skip_op(context.current_op[0])
            out = activation(context.current_op[-1] + context.current_op[-2] + skip_out)
            context.current_op = []
            context.previous_op.append(out)
            context.flag_inject(False)
        return out, context

class ResNetDownSample(nn.Module):
    def __init__(self, stride):
        super(ResNetDownSample, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, kernel_size=3, act="relu", downsample="conv"):
        super(ResNetBlock, self).__init__()
        self.stride = stride
        padding = int((kernel_size - 1) / 2)
        self.act = act
        self.activation = F.relu
        if self.act == 'hardtanh':
            self.activation = F.hardtanh
        elif self.act == 'sigmoid':
            self.activation = F.sigmoid

        self.op_1 = ConvBNReLU(C, C_out,
                               kernel_size, stride, padding, affine=affine, relu=False)
        self.op_2 = ConvBNReLU(C_out, C_out,
                               kernel_size, 1, padding, affine=affine, relu=False)
        if downsample == "conv":
            self.skip_op = Identity() if stride == 1 else ConvBNReLU(C, C_out,
                                                                     1, stride, 0,
                                                                     affine=affine, relu=False)
        elif downsample == "avgpool":
            self.skip_op = Identity() if stride == 1 else ResNetDownSample(stride)

    def forward(self, inputs):
        inner = self.activation(self.op_1(inputs))
        out = self.op_2(inner)
        out_skip = self.skip_op(inputs)
        return self.activation(out + out_skip)

    def forward_one_step(self, context=None, inputs=None):
        activation = F.relu
        if self.act == 'hardtanh':
            activation = F.hardtanh
        elif self.act == 'sigmoid':
            activation = F.sigmoid
        _, op_ind = context.next_op_index
        if op_ind == 0:
            assert inputs is not None
            context.current_op.append(inputs) # save inputs
            out = self.op_1(inputs)
            context.current_op.append(out)
            context.last_conv_module = self.op_1.op[0]
        elif op_ind == 2:
            out = self.op_2(F.relu(context.current_op[-1]))
            context.current_op.append(out)
            context.last_conv_module = self.op_2.op[0]
        elif op_ind == 3:
            out = self.skip_op(context.current_op[0]) # inputs
            context.current_op.append(out)
            if self.stride == 1: # skip connection is just identity
                context.flag_inject(False)
            else:
                context.last_conv_module = self.skip_op.op[0]
        else:
            assert op_ind == 4
            out = activation(context.current_op[-1] + context.current_op[-2])
            context.current_op = []
            context.previous_op.append(out)
            context.flag_inject(False)
        return out, context

class ResNetBottleneckBlock(nn.Module):
    """
    References: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    For resnet-50, resnet-101, resnet-152
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, affine=True, downsample="conv1x1"):
        super(ResNetBottleneckBlock, self).__init__()
        mid_planes = planes // self.expansion
        self.conv1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes, affine=affine)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes, affine=affine)
        self.conv3 = nn.Conv2d(mid_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            if downsample == "conv1x1":
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes),
                )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBlock_bnreluconv(nn.Module):
    """
    Reference: https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
    For WideResNet
    """
    def __init__(self, C, C_out, stride, affine, kernel_size=3, dropout_rate=0.0):
        super(ResNetBlock_bnreluconv, self).__init__()
        self.C = C
        self.C_out = C_out
        self.dropout_rate = dropout_rate

        self.op_1 = BNReLU(C, C, affine=affine)
        self.op_2 = ConvBNReLU(C, C_out, kernel_size, stride, padding=1, affine=affine, relu=True)
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.op_3 = nn.Conv2d(C_out, C_out, kernel_size, stride=1, padding=1, bias=False)
        self.downsample = None
        if stride != 1 or C != C_out:
            self.downsample = nn.Conv2d(
                C, C_out, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, inputs):
        out = self.op_1(inputs)
        out = self.dropout(self.op_2(out))
        out = self.op_3(out)
        identity = inputs if self.downsample is None else self.downsample(inputs)
        return out + identity


class DenseBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, act='relu', bc_mode=True, bc_ratio=4.0,
                 dropblock_rate=0.0):
        super(DenseBlock, self).__init__()
        growth_rate = self.growth_rate = C_out - C
        self.bc_mode = bc_mode
        self.dropblock_rate = dropblock_rate

        if bc_mode:
            inner_c = int(bc_ratio * growth_rate)
            self.bc_bn = nn.BatchNorm2d(C, affine=affine)
            self.bc_conv = nn.Conv2d(C, inner_c, kernel_size=1, bias=False)
        else:
            inner_c = C
        self.bn = nn.BatchNorm2d(inner_c, affine=affine)
        self.conv = nn.Conv2d(inner_c, growth_rate, kernel_size=3, padding=1, bias=False)

        self.activation = F.relu
        if act == "sigmoid":
            self.activation = F.sigmoid
        elif act == "hardtanh":
            self.activation = F.hardtanh

    def forward(self, x):
        if self.bc_mode:
            out = self.bc_conv(self.activation(self.bc_bn(x)))
        else:
            out = x
        out = self.conv(self.activation(self.bn(out)))
        if self.training and self.dropblock_rate > 0.:
            # optionally drop miniblock
            keep_prob = 1. - self.dropblock_rate
            mask = x.new(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            out.div_(keep_prob)
            out.mul_(mask)
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, C, C_out, stride, affine, act='relu'):
        assert stride == 2 and affine, "standard densenet use stride=2 and affine=True"
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(C)
        self.conv = nn.Conv2d(C, C_out, kernel_size=1, bias=False)
        self.activation = F.relu
        if act == "sigmoid":
            self.activation = F.sigmoid
        elif act == "hardtanh":
            self.activation = F.hardtanh

    def forward(self, x):
        out = self.conv(self.activation(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


# ---------------------- OFA Blocks --------------------------
class MobileNetV2Block(nn.Module):
    def __init__(self, expansion, C, C_out, 
                stride, 
                kernel_size,
                affine,
                activation="relu",
                inv_bottleneck=None,
                depth_wise=None,
                point_linear=None,
                ):
        super(MobileNetV2Block, self).__init__()
        self.expansion = expansion
        self.C = C
        self.C_out = C_out 
        self.C_inner = make_divisible(C * expansion, 8)
        self.stride = stride
        self.kernel_size = kernel_size
        self.act_fn = get_op(activation)()
        
        self.inv_bottleneck = None
        if expansion != 1:
            self.inv_bottleneck = inv_bottleneck or nn.Sequential(
                nn.Conv2d(C, self.C_inner, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.C_inner),
                self.act_fn
            )
        
        self.depth_wise = depth_wise or nn.Sequential(
            nn.Conv2d(self.C_inner, self.C_inner, self.kernel_size, stride, padding=self.kernel_size
                // 2, bias=False, groups=self.C_inner),
            nn.BatchNorm2d(self.C_inner),
            self.act_fn
        )

        self.point_linear = point_linear or nn.Sequential(
            nn.Conv2d(self.C_inner, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.shortcut = nn.Sequential()
        self.has_conv_shortcut = False
        if stride == 1 and C == C_out:
            self.has_conv_shortcut = True

    def forward(self, inputs, drop_connect_rate=0.0):
        out = inputs
        if self.inv_bottleneck:
            out = self.inv_bottleneck(out)
        out = self.depth_wise(out)
        out = self.point_linear(out)
        if self.has_conv_shortcut:
            if drop_connect_rate > 0:
                out = drop_connect(out, p=drop_connect_rate, training=self.training)
            out = out + self.shortcut(inputs)
        return out


class MobileNetV3Block(nn.Module):
    def __init__(self, expansion, C, C_out, 
                 stride, 
                 kernel_size,
                 affine,
                 activation="relu",
                 use_se=False,
                 inv_bottleneck=None,
                 depth_wise=None,
                 point_linear=None,
                 se=None,
                 ):
        super(MobileNetV3Block, self).__init__()
        self.expansion = expansion
        self.C = C
        self.C_out = C_out 
        self.C_inner = make_divisible(C * expansion, 8)
        self.stride = stride
        self.kernel_size = kernel_size
        self.act_fn = get_op(activation)()
        self.use_se = use_se
        
        self.inv_bottleneck = None
        if expansion != 1:
            self.inv_bottleneck = inv_bottleneck or nn.Sequential(
                nn.Conv2d(C, self.C_inner, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.C_inner),
                self.act_fn
            )
        
        self.depth_wise = depth_wise or nn.Sequential(
            nn.Conv2d(self.C_inner, self.C_inner, self.kernel_size, stride, self.kernel_size // 2, groups=self.C_inner, bias=False),
            nn.BatchNorm2d(self.C_inner),
            self.act_fn
        )

        self.point_linear = point_linear or nn.Sequential(
            nn.Conv2d(self.C_inner, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.se = None
        if self.use_se:
            self.se = se or SEModule(self.C_inner)
        
        self.shortcut = None
        if stride == 1 and C == C_out:
            self.shortcut = nn.Sequential()

    def forward(self, inputs, drop_connect_rate=0.0):
        out = inputs
        if self.inv_bottleneck:
            out = self.inv_bottleneck(out)
        out = self.depth_wise(out)
        if self.se:
            out = self.se(out)
        out = self.point_linear(out)
        if self.shortcut is not None:
            if drop_connect_rate > 0:
                out = drop_connect(out, p=drop_connect_rate, training=self.training)
            out = out + self.shortcut(inputs)
        return out


# ------------------ DPU Friendly operators --------------------
class FusedConvBlock(nn.Module):
    def __init__(self, expansion, C, C_out, 
                stride, 
                kernel_size,
                affine,
                activation="relu",
                ):
        super(FusedConvBlock, self).__init__()
        self.expansion = expansion
        self.C = C
        self.C_out = C_out 
        self.C_inner = make_divisible(C * expansion, 8)
        self.stride = stride
        self.kernel_size = kernel_size
        self.act_fn = get_op(activation)
        
        self.fused_conv = nn.Sequential(
            nn.Conv2d(self.C, self.C_inner, self.kernel_size, stride, padding=self.kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.C_inner),
            self.act_fn()
        )

        self.point_linear = nn.Sequential(
            nn.Conv2d(self.C_inner, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.shortcut = nn.Sequential()
        self.has_conv_shortcut = False
        if stride == 1 and C == C_out:
            self.has_conv_shortcut = True

    def forward(self, inputs, drop_connect_rate=0.0):
        out = self.fused_conv(inputs)
        out = self.point_linear(out)
        if self.has_conv_shortcut:
            if drop_connect_rate > 0:
                out = drop_connect(out, p=drop_connect_rate, training=self.training)
            out = out + self.shortcut(inputs)
        return out


register_primitive("mobilenet_block_6_relu6",
                   lambda C, C_out, stride, affine: MobileNetBlock(6, C, C_out,
                                                                   stride, affine, True))
register_primitive("mobilenet_block_1_relu6",
                   lambda C, C_out, stride, affine: MobileNetBlock(1, C, C_out, stride,
                                                                   affine, True))
register_primitive("mobilenet_block_6",
                   lambda C, C_out, stride, affine: MobileNetBlock(6, C, C_out, stride, affine))
register_primitive("mobilenet_block_6_5x5",
                   lambda C, C_out, stride, affine: MobileNetBlock(6, C, C_out, stride, affine, 5))
register_primitive("mobilenet_block_3",
                   lambda C, C_out, stride, affine: MobileNetBlock(3, C, C_out, stride, affine))
register_primitive("mobilenet_block_3_5x5",
                   lambda C, C_out, stride, affine: MobileNetBlock(3, C, C_out, stride, affine, 5))
register_primitive("mobilenet_block_1",
                   lambda C, C_out, stride, affine: MobileNetBlock(1, C, C_out, stride, affine))

register_primitive("resnet_block_1x1",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine,
                                                                kernel_size=1))
register_primitive("resnet_block",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine))
register_primitive("resnet_block_pool_downsample",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine, downsample="avgpool"))

register_primitive("resnet_block_5x5",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine,
                                                                kernel_size=5))
register_primitive("resnet_block_split",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine))
register_primitive("resnet_block_hardtanh",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine,
                                                                act='hardtanh'))
register_primitive("resnet_block_split_hardtanh",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine,
                                                                     act='hardtanh'))
register_primitive("resnet_block_sigmoid",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine,
                                                                act='sigmoid'))
register_primitive("resnet_block_split_sigmoid",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine,
                                                                     act='sigmoid'))
register_primitive("resnet_bottleneck_block",
                   lambda C, C_out, stride, affine: ResNetBottleneckBlock(
                       C, C_out, stride=stride, affine=affine))
register_primitive("wideresnet_block_3x3",
                   lambda C, C_out, stride, affine: ResNetBlock_bnreluconv(
                       C, C_out, stride, affine, kernel_size=3, dropout_rate=0.))
register_primitive("vgg_block",
                   lambda C, C_out, stride, affine: VggBlock(C, C_out, stride, affine=affine))
register_primitive("dense_block",
                   lambda C, C_out, stride, affine: DenseBlock(C, C_out, stride, affine=affine))
register_primitive("dense_reduce_block",
                   lambda C, C_out, stride, affine: Transition(C, C_out, stride, affine=affine))


register_primitive("mobilenet_v2_block",
                   lambda C, C_out, stride, affine, expansion, kernel_size, activation: MobileNetV2Block(expansion, C, C_out, stride, kernel_size, affine=affine, activation=activation))
register_primitive("mobilenet_v3_block",
                   lambda C, C_out, stride, affine, expansion, kernel_size, activation, use_se: MobileNetV3Block(expansion, C, C_out, stride, kernel_size, affine=affine, activation=activation, use_se=use_se))
register_primitive("fused_conv_block",
                   lambda C, C_out, stride, affine, expansion, kernel_size, activation: FusedConvBlock(expansion, C, C_out, stride, kernel_size, affine=affine, activation=activation))
