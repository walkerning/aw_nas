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
    def __init__(self, expansion, C, C_out, stride, affine, kernel_size=3):
        super(MobileNetBlock, self).__init__()
        C_inner = self.C_inner = int(expansion * C)
        self.stride = stride

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
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(inputs) if self.stride == 1 else out
        return out

    def forward_one_step(self, context=None, inputs=None):
        _, op_ind = context.next_op_index
        if op_ind == 0:
            assert inputs is not None
            context.current_op.append(inputs) # save inputs
            out = self.bn1(self.conv1(inputs))
            context.current_op.append(out)
        elif op_ind == 2:
            out = self.bn2(self.conv2(F.relu(context.current_op[-1])))
            context.current_op.append(out)
        elif op_ind == 3:
            out = self.bn3(self.conv3(F.relu(context.current_op[-1])))
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
    def __init__(self, C, C_out, stride, affine):
        super(ResNetBlockSplit, self).__init__()
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
        inner = F.relu(self.op_1_1(inputs) + self.op_1_2(inputs))
        out = self.op_2_1(inner) + self.op_2_2(inner)
        out_skip = self.skip_op(inputs)
        return F.relu(out + out_skip)

    def forward_one_step(self, context=None, inputs=None):
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
            out = F.relu(context.current_op[-1] + context.current_op[-2])
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
            out = F.relu(context.current_op[-1] + context.current_op[-2] + skip_out)
            context.current_op = []
            context.previous_op.append(out)
            context.flag_inject(False)
        return out, context

class ResNetBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, kernel_size=3):
        super(ResNetBlock, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.op_1 = ConvBNReLU(C, C_out,
                               kernel_size, stride, padding, affine=affine, relu=False)
        self.op_2 = ConvBNReLU(C_out, C_out,
                               kernel_size, 1, padding, affine=affine, relu=False)
        self.skip_op = Identity() if stride == 1 else ConvBNReLU(C, C_out,
                                                                 1, stride, 0,
                                                                 affine=affine, relu=False)

    def forward(self, inputs):
        inner = F.relu(self.op_1(inputs))
        out = self.op_2(inner)
        out_skip = self.skip_op(inputs)
        return F.relu(out + out_skip)

    def forward_one_step(self, context=None, inputs=None):
        _, op_ind = context.next_op_index
        if op_ind == 0:
            assert inputs is not None
            context.current_op.append(inputs) # save inputs
            out = self.op_1(inputs)
            context.current_op.append(out)
        elif op_ind == 2:
            out = self.op_2(F.relu(context.current_op[-1]))
            out_skip = self.skip_op(context.current_op[0]) # inputs
            out = out + out_skip
            context.current_op.append(out)
        else:
            assert op_ind == 3
            out = F.relu(context.current_op[-1])
            context.current_op = []
            context.previous_op.append(out)
            context.flag_inject(False)
        return out, context

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
register_primitive("resnet_block_5x5",
                   lambda C, C_out, stride, affine: ResNetBlock(C, C_out, stride, affine=affine,
                                                                kernel_size=5))
register_primitive("resnet_block_split",
                   lambda C, C_out, stride, affine: ResNetBlockSplit(C, C_out,
                                                                     stride, affine=affine))
register_primitive("vgg_block",
                   lambda C, C_out, stride, affine: VggBlock(C, C_out, stride, affine=affine))
