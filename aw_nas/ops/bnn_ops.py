# some bnn op
"""
NN operations.
"""
#pylint: disable=arguments-differ,useless-super-delegation,invalid-name

import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from aw_nas.utils.common_utils import get_sub_kernel, make_divisible, _get_channel_mask
from aw_nas.utils.exception import ConfigException, expect
from torch import nn

from aw_nas.ops import register_primitive, ConvBNReLU, Identity, SEModule, get_op
from aw_nas.utils import make_divisible, drop_connect

# The XNOR-Net

class BinaryActivation(torch.autograd.Function):
  @staticmethod
  def forward(self, input):
    self.save_for_backward(input)
    size = input.size()
    input = input.sign()
    return input
  @staticmethod
  def backward(self, grad_output):
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input.ge(1)] = 0
    grad_input[input.le(-1)] = 0
    return grad_input

class Fp32Activation(torch.autograd.Function):
  @staticmethod
  def forward(self, input):
    self.save_for_backward(input)
    size = input.size()
    return input
  @staticmethod
  def backward(self, grad_output):
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input.ge(1)] = 0
    grad_input[input.le(-1)] = 0
    return grad_input



class XNORConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, dropout_ratio=0, fp32_act=True):
    super(XNORConv2d, self).__init__()
    self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.groups, self.dropout_ratio, self.fp32_act = in_channels, out_channels, kernel_size, stride, padding, groups, dropout_ratio, fp32_act
    if dropout_ratio !=0:
      self.dropout = nn.Dropout(dropout_ratio)
    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = groups)
    self.conv.weight.data.normal_(0, 0.05)
    self.conv.bias.data.zero_()
    self.act = BinaryActivation(self.fp32_act)

    self.full_precision = nn.Parameter(torch.zeros(self.conv.weight.size()))
    self.full_precision.data.copy_(self.conv.weight.data)
    
  def forward(self, x):
    self.full_precision.data = self.full_precision.data - self.full_precision.data.mean(1, keepdim = True)
    self.full_precision.data.clamp_(-1, 1)

    # import ipdb; ipdb.set_trace()
    if self.fp32_act:
        x = Fp32Activation.apply(x)
    else:
        x = BinaryActivation.apply(x)
    if self.dropout_ratio!=0:
      x = self.dropout(x)
    # import ipdb; ipdb.set_trace()
    self.mean_val = self.full_precision.abs().view(self.out_channels, -1).mean(1, keepdim=True)

    self.conv.weight.data.copy_(self.full_precision.data.sign() * self.mean_val.view(-1, 1, 1, 1))
    x = self.conv(x)

    return x

  def copy_grad(self):
   #  import ipdb; ipdb.set_trace()
    proxy = self.full_precision.abs().sign()
    proxy[self.full_precision.data.abs()>1] = 0
    binary_grad = self.conv.weight.grad * self.mean_val.view(-1, 1, 1, 1) * proxy
    # binary_grad = self.conv.weight.grad * self.full_precision.abs() * proxy

    mean_grad = self.conv.weight.data.sign() * self.conv.weight.grad
    mean_grad = mean_grad.view(self.out_channels, -1).mean(1).view(-1, 1, 1, 1)
    mean_grad = mean_grad * self.conv.weight.data.sign()

    self.full_precision.grad = binary_grad + mean_grad
    self.full_precision.grad = self.full_precision.grad * self.full_precision.data[0].nelement() * (1-1/self.full_precision.data.size(1))

class XNORConvBNReLU(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding,affine=True,groups=1, dropout_ratio=0, fp32_act=True):
    super(XNORConvBNReLU, self).__init__()
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1, affine=True)
    # self.bn.weight.fill_(1.0)
    # self.bn.bias.zero_()
    self.conv = XNORConv2d(in_channels, out_channels, kernel_size, stride, padding, groups, dropout_ratio, fp32_act)

  def forward(self, x): # this is dirty, later would delete param & flops args
    out = self.conv(x)
    out = self.bn(out)
    out = F.relu(out)
    return out

# class XNORLinear(nn.Module):
#   def __init__(self, in_features, out_features, dropout_ratio=0):
#     super(XNORLinear, self).__init__()
#     self.in_features, self.out_features, self.dropout_ratio = in_features, out_features, dropout_ratio
#     if dropout_ratio !=0:
#       self.dropout = nn.Dropout(dropout_ratio)
#     self.linear = nn.Linear(in_features = in_features, out_features = out_features)
#     self.linear.weight.normal_(0, 2.0/self.linear.weight[0].nelement())
#     self.full_precision = nn.Parameter(torch.zeros(self.linear.weight.size()))
#     self.full_precision.data.copy_(self.linear.weight.data)
# 
#   def forward(self, x, params, flops):
#     self.full_precision.data = self.full_precision.data - self.full_precision.data.mean(1, keepdim = True)
#     self.full_precision.data.clamp_(-1, 1)
# 
#     x = BinaryActivation.apply(x)
#     if self.dropout_ratio != 0:
#       x = self.dropout(x)
#     self.mean_val = self.full_precision.abs().view(self.out_features, -1).mean(1, keepdim=True)
# 
#     self.linear.weight.data.copy_(self.full_precision.data.sign() * self.mean_val.view(-1, 1))
#     x = self.linear(x)
# 
#     params = params + self.linear.weight.nelement() / 32
#     flops = flops + self.linear.weight.nelement() / 64
# 
#     return x, params, flops
# 
#   def copy_grad(self):
#     proxy = self.full_precision.abs().sign()
#     proxy[self.full_precision.data.abs()>1] = 0
#     binary_grad = self.linear.weight.grad * self.mean_val.view(-1, 1) * proxy
# 
#     mean_grad = self.linear.weight.data.sign() * self.linear.weight.grad
#     mean_grad = mean_grad.view(self.out_features, -1).mean(1).view(-1, 1)
#     mean_grad = mean_grad * self.linear.weight.data.sign()
# 
#     self.full_precision.grad = binary_grad + mean_grad
#     self.full_precision.grad = self.full_precision.grad * self.full_precision.data[0].nelement() * (1-1/self.full_precision.data.size(1))
# 
# class BNLinearReLU(nn.Module):
#   def __init__(self, in_features, out_features, dropout_ratio=0):
#     super(BNLinearReLU, self).__init__()
#     self.bn = nn.BatchNorm1d(in_features, eps=1e-4, momentum=0.1, affine=True)
#     self.bn.weight.zero_().add_(1.0)
#     self.bn.bias.zero_()
#     self.linear = XNORLinear(in_features, out_features, dropout_ratio)
#     
#   def forward(self, x, params, flops):
#     out = self.bn(x)
#     params = params + self.linear.in_features
#     flops = flops + self.linear.in_features
#     out, params, flops = self.linear(out, params, flops)
#     out = F.relu(out)
#     return out, params, flops
# 

# ================ The bi-real net ================
# FIXME: Maybe Merge later


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BirealActivation(nn.Module):
    def __init__(self):
        super(BirealActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        # out = out_forward.detach() - out3.detach() + out3
        out = out_forward.detach() - x.detach() + x

        return out


class HardBinaryConv(nn.Module):
    def __init__(self, C_in, c_out, kernel_size=3, stride=1, padding=1, bias=None): 
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bias = bias # the shape of bias
        self.number_of_weights = C_in * c_out * kernel_size * kernel_size
        self.shape = (c_out, C_in, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = real_weights.abs().mean().reshape([1,1,1,1]).detach()
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # print(scaling_factor, flush=True)
        # scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * BinaryActivation.apply(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # print(binary_weights, flush=True)
        # binary_weights = BinaryActivation().apply(torch.clamp(real_weights, -1.0, 1.0))*scaling_factor # FIXME: concerns some problem with grad here
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, bias=self.bias)

        return y

class BinaryConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine,  bias=None, activation=BinaryActivation):
        super(BinaryConvBNReLU, self).__init__()
        self.bi_conv = HardBinaryConv(C_in, C_out, kernel_size=kernel_size,stride=stride, padding=padding, bias=bias)
        self.bi_act = activation
        self.op = nn.Sequential(
            self.bi_act,
            self.bi_conv,
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)

class ResNetDownSample(nn.Module):
    def __init__(self, stride):
        super(ResNetDownSample, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)

class BinaryResNetBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, kernel_size=3, block="bireal",act=BinaryActivation,downsample="conv", fp32_act=True):
        super(BinaryResNetBlock, self).__init__()
        self.stride = stride
        padding = int((kernel_size - 1) / 2)
        self.activation = act
        # tha act is only used when using bireal-block

        if block == "bireal":
            self.op_1 = BinaryConvBNReLU(C, C_out,
                                   kernel_size, stride, padding, affine=affine, bias=None, activation=self.activation)
            self.op_2 = BinaryConvBNReLU(C_out, C_out,
                                   kernel_size, 1, padding, affine=affine, bias=None, activation=self.activation)
        elif block == "xnor":
            self.op_1 = XNORConvBNReLU(C, C_out,
                                   kernel_size, stride, padding, affine=affine,groups=1, dropout_ratio=0, fp32_act=fp32_act)
            self.op_2 = XNORConvBNReLU(C_out, C_out,
                                   kernel_size, 1, padding, affine=affine,groups=1, dropout_ratio=0, fp32_act=fp32_act)
        elif block == "dorefa":
            self.op_1 = DorefaConvBNReLU(C, C_out,
                                   kernel_size, stride, padding, affine=affine,groups=1)
            self.op_2 = DorefaConvBNReLU(C_out, C_out,
                                   kernel_size, 1, padding, affine=affine,groups=1)

        # DBEUG: test correctness
        # self.activation = Identity()
        # self.op_1 = ConvBNReLU(C, C_out,
        #                        kernel_size, stride, padding, affine=affine)
        # self.op_2 = ConvBNReLU(C_out, C_out, 
        #                        kernel_size, 1, padding, affine=affine)
        # the conv on skip-op is not binarized!
        if downsample == "conv":
            self.skip_op = Identity() if stride == 1 else ConvBNReLU(C, C_out,
                                                                     1, stride, 0,
                                                                     affine=affine)
        elif downsample == "avgpool":
            self.skip_op = Identity() if stride == 1 else ResNetDownSample(stride)

    def forward(self, inputs):
        # for birealnet add an extra skip connection after op1
        inner = self.op_1(inputs)+self.skip_op(inputs)
        out = self.op_2(inner)+inner
        return out

register_primitive("bireal_resnet_block", 
                   lambda C, C_out, stride, affine: BinaryResNetBlock(C, C_out, stride=stride, affine=affine,block="bireal",act=BirealActivation()))
register_primitive("xnor_resnet_block", 
                   lambda C, C_out, stride, affine: BinaryResNetBlock(C, C_out, stride=stride, affine=affine,block="xnor"))
register_primitive("xnor_resnet_block_biact", 
                   lambda C, C_out, stride, affine: BinaryResNetBlock(C, C_out, stride=stride, affine=affine,block="xnor",fp32_act=False))

register_primitive("dorefa_resnet_block", 
                   lambda C, C_out, stride, affine: BinaryResNetBlock(C, C_out, stride=stride, affine=affine,block="dorefa"))
register_primitive("xnor_resnet_block_pool_downsample", 
                   lambda C, C_out, stride, affine: BinaryResNetBlock(C, C_out, stride=stride, affine=affine,block="xnor",downsample="avgpool"))





class BinaryVggBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine, block="xnor"):
        super(BinaryVggBlock, self).__init__()
        if block=="xnor":
            self.op = XNORConvBNReLU(C, C_out,
                                   3, 1, 1, affine=affine, groups=1, dropout_ratio=0)
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, inputs):
        return self.pool(self.op(inputs))

register_primitive("bireal_vgg_block",
                   lambda C, C_out, stride, affine: BinaryVggBlock(C, C_out, stride, affine=affine,block="bireal"))
register_primitive("xnor_vgg_block",
                   lambda C, C_out, stride, affine: BinaryVggBlock(C, C_out, stride, affine=affine,block="xnor"))



class NIN(nn.Module):
    def __init__(self, C, C_out, stride, affine):
        super(NIN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 192, kernel_size = 5, stride = 1, padding = 2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum = 0.1, affine = True)

        self.conv2_1 = XNORConvBNReLU(192, 160, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = XNORConvBNReLU(160, 96, kernel_size=1, stride=1, padding=0)

        self.conv3_1 = XNORConvBNReLU(96, 192, kernel_size=5, stride=1, padding=2, dropout_ratio=0.5)
        # self.conv3_2 = XNORConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)
        # self.conv3_3 = XNORConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)

        self.conv4_1 = XNORConvBNReLU(192, 192, kernel_size=3, stride=1, padding=1, dropout_ratio=0.5)
        # self.conv4_2 = XNORConvBNReLU(192, 192, kernel_size=1, stride=1, padding=0)

        self.bn5 = nn.BatchNorm2d(192, eps = 1e-4, momentum = 0.1, affine = True)
        self.conv5 = nn.Conv2d(192, 10, kernel_size = 1, stride = 1, padding = 0)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
                    
    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min = 0.01)
                    
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = F.max_pool2d(x, kernel_size = 3, stride = 2, padding = 1)

        x = self.conv3_1(x)
        # x = self.conv3_2(x)
        # x = self.conv3_3(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv4_1(x)
        # x = self.conv4_2(x)

        x = F.relu(self.conv5(self.bn5(x)))
        x = F.avg_pool2d(x, kernel_size = 8)
       #  x = x.squeeze()
        return x

register_primitive("nin",
                   lambda C, C_out, stride, affine: NIN(C, C_out, stride, affine))


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
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
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q

class DorefaConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, w_bit=1, a_bit=32):
      super(DorefaConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, w_bit)
      self.w_bit = w_bit
      self.a_bit = a_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

class DorefaConvBNReLU(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding,affine=True,groups=1):
    super(DorefaConvBNReLU, self).__init__()
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1, affine=True)
    # self.bn.weight.fill_(1.0)
    # self.bn.bias.zero_()
    self.conv = DorefaConv2d(in_channels, out_channels, kernel_size, stride, padding, groups)
    self.act = activation_quantize_fn(a_bit=self.conv.a_bit)

  def forward(self, x): # this is dirty, later would delete param & flops args
    x = self.act(x)
    out = self.conv(x)
    out = self.bn(out)
    out = F.relu(out)
    return out



# class BiRealNet(nn.Module):
# 
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
#         super(BiRealNet, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
# 
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.AvgPool2d(kernel_size=2, stride=stride),
#                 conv1x1(self.inplanes, planes * block.expansion),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
# 
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
# 
#         return nn.Sequential(*layers)
# 
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.maxpool(x)
# 
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
# 
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
# 
#         return x
# 
# 
# def birealnet18(pretrained=False, **kwargs):
#     """Constructs a BiRealNet-18 model. """
#     model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
#     return model
# 
# 
# def birealnet34(pretrained=False, **kwargs):
#     """Constructs a BiRealNet-34 model. """
#     model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
#     return model
# 


