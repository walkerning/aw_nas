import torch
from torch import nn
import torch.nn.functional as F
from aw_nas.ops import register_primitive, ConvBNReLU, Identity

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
        elif op_ind == 1:
            out = self.pool(F.relu(context.current_op[-1]))
            context.previous_op.append(out)
            context.current_op = []
            context.flag_inject(False)
        else:
            raise Exception("Unexpected op_ind")
        return out, context

class MobileNetBlock(nn.Module):
    def __init__(self, expansion, C, C_out, stride, affine, relu6=False):
        super(MobileNetBlock, self).__init__()
        C_inner = self.C_inner = int(expansion * C)
        self.stride = stride
        self.relu6 = relu6

        self.conv1 = nn.Conv2d(C, C_inner,
                               kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(C_inner, affine=affine)
        self.conv2 = nn.Conv2d(C_inner, C_inner,
                               kernel_size=3, stride=stride,
                               padding=1, groups=C_inner, bias=False)
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
        activation = F.relu6 if self.relu6 else F.relu
        out = activation(self.bn1(self.conv1(inputs)))
        out = activation(self.bn2(self.conv2(out)))
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
        elif op_ind == 2:
            out = self.bn2(self.conv2(activation(context.current_op[-1])))
            context.current_op.append(out)
        elif op_ind == 3:
            out = self.bn3(self.conv3(activation(context.current_op[-1])))
            context.current_op.append(out)
            if not self.stride == 1:
                # return out
                context.previous_op.append(out)
                context.current_op = []
            elif not self.has_conv_shortcut:
                # return out + shortcut
                context.previous_op.append(out + self.shortcut(context.current_op[0]))
                context.current_op = []
        elif op_ind == 4:
            # has_conv_shortcut
            out = self.shortcut(context.current_op[0])
            context.current_op.append(out)
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

class ResNetBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, act='relu'):
        super(ResNetBlock, self).__init__()
        self.act = act
        self.op_1 = ConvBNReLU(C, C_out,
                               3, stride, 1, affine=affine, relu=False)
        self.op_2 = ConvBNReLU(C_out, C_out,
                               3, 1, 1, affine=affine, relu=False)
        self.skip_op = Identity() if stride == 1 else ConvBNReLU(C, C_out,
                                                                 1, stride, 0,
                                                                 affine=affine, relu=False)

    def forward(self, inputs):
        activation = F.relu
        if self.act == 'hardtanh':
            activation = F.hardtanh
        elif self.act == 'sigmoid':
            activation = F.sigmoid
        inner = activation(self.op_1(inputs))
        out = self.op_2(inner)
        out_skip = self.skip_op(inputs)
        return activation(out + out_skip)

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
        elif op_ind == 2:
            out = self.op_2(activation(context.current_op[-1]))
            out_skip = self.skip_op(context.current_op[0]) # inputs
            out = out + out_skip
            context.current_op.append(out)
        else:
            assert op_ind == 3
            out = activation(context.current_op[-1])
            context.current_op = []
            context.previous_op.append(out)
            context.flag_inject(False)
        return out, context

class DenseBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, act='relu'):
        super(DenseBlock, self).__init__()
        growth_rate = C_out - C
        self.bn1 = nn.BatchNorm2d(C, affine=affine)
        self.conv1 = nn.Conv2d(C, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate, affine=affine)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.activation = F.relu
        if act == "sigmoid":
            self.activation = F.sigmoid
        elif act == "hardtanh":
            self.activation = F.hardtanh

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, C, C_out, stride, affine, act='relu'):
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

register_primitive("mobilenet_block_6_relu6",
                   lambda C, C_out, stride, affine: MobileNetBlock(6, C, C_out, stride, affine, True))
register_primitive("mobilenet_block_1_relu6",
                   lambda C, C_out, stride, affine: MobileNetBlock(1, C, C_out, stride, affine, True))
register_primitive("mobilenet_block_6",
                   lambda C, C_out, stride, affine: MobileNetBlock(6, C, C_out, stride, affine))
register_primitive("mobilenet_block_1",
                   lambda C, C_out, stride, affine: MobileNetBlock(1, C, C_out, stride, affine))
register_primitive("resnet_block",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine))
register_primitive("resnet_block_split",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine))
register_primitive("resnet_block_hardtanh",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine, act='hardtanh'))
register_primitive("resnet_block_split_hardtanh",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine, act='hardtanh'))
register_primitive("resnet_block_sigmoid",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine, act='sigmoid'))
register_primitive("resnet_block_split_sigmoid",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine, act='sigmoid'))
register_primitive("vgg_block",
                   lambda C, C_out, stride, affine: VggBlock(C, C_out, stride, affine=affine))
register_primitive("dense_block",
                   lambda C, C_out, stride, affine: DenseBlock(C, C_out, stride, affine=affine))
register_primitive("dense_reduce_block",
                   lambda C, C_out, stride, affine: Transition(C, C_out, stride, affine=affine))
