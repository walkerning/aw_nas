"""
NN operations.
"""
#pylint: disable=arguments-differ,useless-super-delegation,invalid-name

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from aw_nas.utils.common_utils import get_sub_kernel, make_divisible, _get_channel_mask
from aw_nas.utils.exception import expect


def get_avg_pool_with_size(size):
    f = lambda C, C_out, stride, affine: nn.AvgPool2d(
        size, stride=stride, padding=(size-1)//2, count_include_pad=False)
    return f

def get_max_pool_with_size(size):
    f = lambda C, C_out, stride, affine: nn.MaxPool2d(size, stride=stride, padding=(size-1)//2)
    return f

def conv_7x1_1x7(C, C_out, stride, affine):
    assert C == C_out
    return nn.Sequential(
        # C_out is ignored
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    )

class Hswish(nn.Module):
    def __init__(self, inplace):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) * (1 / 6.)

class Hsigmoid(nn.Module):
    def __init__(self, inplace):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) * (1 / 6.)

PRIMITVE_FACTORY = {
    "none" : lambda C, C_out, stride, affine: Zero(stride),
    "avg_pool_2x2" : get_avg_pool_with_size(2),
    "avg_pool_3x3" : get_avg_pool_with_size(3),
    "max_pool_3x3" : get_max_pool_with_size(3),
    "skip_connect" : lambda C, C_out, stride, affine: Identity() if stride == 1 \
      else FactorizedReduce(C, C_out, stride=stride, affine=affine),
    "res_reduce_block": lambda C, C_out, stride, affine: ResFactorizedReduceBlock(
        C, C_out, stride=stride, affine=affine),

    "sep_conv_3x3" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 3, stride, 1, affine=affine),
    "sep_conv_3x3_exp3" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 3, stride, 1, affine=affine, expansion=3),
    "sep_conv_3x3_exp6" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 3, stride, 1, affine=affine, expansion=6),
    "sep_conv_5x5" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 5, stride, 2, affine=affine),
    "sep_conv_5x5_exp3" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 5, stride, 2, affine=affine, expansion=6),
    "sep_conv_5x5_exp6" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 5, stride, 2, affine=affine, expansion=6),
    "sep_conv_7x7" : lambda C, C_out, stride, affine: SepConv(
        C, C_out, 7, stride, 3, affine=affine),
    "dil_conv_3x3" : lambda C, C_out, stride, affine: DilConv(
        C, C_out, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5" : lambda C, C_out, stride, affine: DilConv(
        C, C_out, 5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7" : conv_7x1_1x7,

    "bn_conv_relu_1x1": lambda C, C_out, stride, affine: BNConvReLU(C, C_out,
                                                                    1, stride, 0, affine=affine),
    "nor_conv_1x1" : lambda C, C_out, stride, affine: ReLUConvBN(C, C_out,
                                                                     1, stride, 0, affine=affine),
    "nor_conv_3x3" : lambda C, C_out, stride, affine: ReLUConvBN(C, C_out,
                                                                     3, stride, 1, affine=affine),
    "relu_conv_bn_1x1" : lambda C, C_out, stride, affine: ReLUConvBN(C, C_out,
                                                                     1, stride, 0, affine=affine),
    "relu_conv_bn_3x3" : lambda C, C_out, stride, affine: ReLUConvBN(C, C_out,
                                                                     3, stride, 1, affine=affine),
    "relu_conv_bn_5x5" : lambda C, C_out, stride, affine: ReLUConvBN(C, C_out,
                                                                     5, stride, 2, affine=affine),
    "conv_bn_relu_1x1" : lambda C, C_out, stride, affine: ConvBNReLU(C, C_out,
                                                                     1, stride, 0, affine=affine),
    "conv_bn_relu_3x3" : lambda C, C_out, stride, affine: ConvBNReLU(C, C_out,
                                                                     3, stride, 1, affine=affine),
    "conv_bn_3x3" : lambda C, C_out, stride, affine: ConvBNReLU(
        C, C_out, 3, stride, 1, affine=affine, relu=False),
    "conv_bn_relu_5x5" : lambda C, C_out, stride, affine: ConvBNReLU(C, C_out,
                                                                     5, stride, 2, affine=affine),
    "conv_1x1" : lambda C, C_out, stride, affine: nn.Conv2d(C, C_out, 1, stride, 0),
    "inspect_block" : lambda C, C_out, stride, affine: inspectBlock(C, C_out, stride,
                                                                    affine=affine),
    "conv_3x3" : lambda C, C_out, stride, affine: nn.Conv2d(C, C_out, 3, stride, 1),
    "bn_relu" : lambda C, C_out, stride, affine: BNReLU(C, C_out, affine),
    "imagenet_stem0": lambda C, C_out, stride, affine: nn.Sequential(
        nn.Conv2d(3, C_out // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_out // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_out // 2, C_out, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_out)
    ),
    "imagenet_stem0_7x7": lambda C, C_out, stride, affine: nn.Sequential(
        nn.Conv2d(C, C_out, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    ),
    "imagenet_stem1": lambda C, C_out, stride, affine: nn.Sequential(
        nn.Conv2d(3, C_out // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_out // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_out // 2, C_out, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_out, C_out, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU(inplace=True),
    ),
    "tnas": lambda C, C_out, stride, affine: nn.Sequential(
        nn.Conv2d(C, C_out // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(C_out // 2),
        nn.MaxPool2d(2, 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(C_out // 2, C_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(C_out),
        nn.MaxPool2d(2, 2)
    ),

    "NB201ResidualBlock": lambda C, C_out, stride, affine: NB201ResidualBlock(C, C_out, stride,
                                                                   affine=affine),
    "se_module": lambda C, C_out, stride, affine, reduction=4: SEModule(C, reduction),

    # activations
    "tanh": lambda C=None, C_out=None, stride=None, affine=None: nn.Tanh(),
    "relu": lambda C=None, C_out=None, stride=None, affine=None: nn.ReLU(inplace=True),
    "sigmoid": lambda C=None, C_out=None, stride=None, affine=None: nn.Sigmoid(),
    "identity": lambda C=None, C_out=None, stride=None, affine=None: Identity(),
    "h_swish": lambda C=None, C_out=None, stride=None, affine=None: Hswish(inplace=True),
    "h_sigmoid": lambda C=None, C_out=None, stride=None, affine=None: Hsigmoid(inplace=True),
    "relu6": lambda C=None, C_out=None, stride=None, affine=None: nn.ReLU6(inplace=True),
    None: lambda C=None, C_out=None, stride=None, affine=None: nn.Sequential(),
}

def register_primitive(name, func, override=False):
    assert callable(func), "A primitive must be callable"
    assert not (name in PRIMITVE_FACTORY and not override),\
        "some func already registered as {};"\
        " to override, use `override=True` keyword arguments.".format(name)
    PRIMITVE_FACTORY[name] = func

def get_op(name):
    assert name in PRIMITVE_FACTORY, \
        "{} not registered, use `register_primitive` to register primitive op".format(name)
    return PRIMITVE_FACTORY[name]

class BNReLU(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(BNReLU, self).__init__()
        assert C_in == C_out
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        return F.relu(self.bn(x))

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, kernel_size=1):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        group_dim = C_out // stride

        padding = int((kernel_size - 1) / 2)
        self.convs = [nn.Conv2d(C_in, group_dim, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)\
                      for _ in range(stride)]
        self.convs = nn.ModuleList(self.convs)

        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

        # just specificy one conv module here, as only C_in, kernel_size, group is used
        # for inject prob calculation every output position, this will work even though
        # not so meaningful conceptually
        object.__setattr__(self, "last_conv_module", self.convs[-1])

    def forward(self, x):
        x = self.relu(x)
        mod = x.size(2) % self.stride
        if mod != 0:
            pad = self.stride - mod
            x = F.pad(x, (pad, 0, pad, 0), "constant", 0)
        out = torch.cat([conv(x[:, :, i:, i:]) for i, conv in enumerate(self.convs)], dim=1)
        out = self.bn(out)
        return out

class DoubleConnect(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, relu=True):
        super(DoubleConnect, self).__init__()
        self.path1 = ReLUConvBN(C_in, C_out, kernel_size, stride=stride, padding=padding, affine=affine)
        self.path2 = ReLUConvBN(C_in, C_out, kernel_size, stride=stride, padding=padding, affine=affine)

    def forward(self, x):
        return self.path1(x) + self.path2(x)


class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, relu=True):
        super(ConvBNReLU, self).__init__()
        if relu:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )
        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)


class BNConvReLU(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, relu=True):
        super(BNConvReLU, self).__init__()
        if relu:
            self.op = nn.Sequential(
                nn.BatchNorm2d(C_in, affine=affine),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.ReLU(inplace=False)
            )
        else:
            self.op = nn.Sequential(
                nn.BatchNorm2d(C_in, affine=affine),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)

class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, expansion=1):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in*expansion, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in*expansion, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in*expansion, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

    def forward_one_step(self, context=None, inputs=None):
        return self.op.forward_one_step(context, inputs)

def forward_one_step(self, context=None, inputs=None):
    #pylint: disable=protected-access,too-many-branches
    assert not context is None

    if not hasattr(self, "_conv_mod_inds"):
        self._conv_mod_inds = []
        mods = list(self._modules.values())
        mod_num = len(mods)
        for i, mod in enumerate(mods):
            if isinstance(mod, nn.Conv2d):
                if i < mod_num - 1 and isinstance(mods[i+1], nn.BatchNorm2d):
                    self._conv_mod_inds.append(i+1)
                else:
                    self._conv_mod_inds.append(i)
        self._num_convs = len(self._conv_mod_inds)

    if not self._num_convs:
        return stub_forward_one_step(self, context, inputs)

    _, op_ind = context.next_op_index
    if inputs is None:
        inputs = context.current_op[-1]
    modules_num = len(list(self._modules.values()))
    if op_ind < self._num_convs:
        for mod_ind in range(self._conv_mod_inds[op_ind-1]+1 if op_ind > 0 else 0,
                             self._conv_mod_inds[op_ind]+1):
            # running from the last point(exclusive) to the #op_ind"s point (inclusive)
            inputs = self._modules[str(mod_ind)](inputs)
        if op_ind == self._num_convs - 1 and self._conv_mod_inds[-1] + 1 == modules_num:
            # if the last calculated module is already the last module in the Sequence container
            context.previous_op.append(inputs)
            context.current_op = []
        else:
            context.current_op.append(inputs)
        last_mod = self._modules[str(self._conv_mod_inds[op_ind])]
        context.last_conv_module = last_mod if isinstance(last_mod, nn.Conv2d) \
            else self._modules[str(self._conv_mod_inds[op_ind]-1)]
    elif op_ind == self._num_convs:
        for mod_ind in range(self._conv_mod_inds[-1]+1, modules_num):
            inputs = self._modules[str(mod_ind)](inputs)
        context.previous_op.append(inputs)
        context.current_op = []
        context.flag_inject(False)
    else:
        assert "ERROR: wrong op index! should not reach here!"
    return inputs, context

def stub_forward_one_step(self, context=None, inputs=None):
    assert not inputs is None and not context is None
    state = self.forward(inputs)
    context.previous_op.append(state)
    if isinstance(self, nn.Conv2d):
        context.last_conv_module = self
    return state, context

nn.Sequential.forward_one_step = forward_one_step
nn.Module.forward_one_step = stub_forward_one_step

def get_last_conv_module(self):
    if hasattr(self, "last_conv_module"):
        return self.last_conv_module

    # in some cases, can auto induce the last conv module
    if isinstance(self, nn.Conv2d):
        return self
    if isinstance(self, nn.Sequential):
        for mod in reversed(self._modules.values()):
            if isinstance(mod, nn.Conv2d):
                return mod
        return None
    if not self._modules:
        return None
    if len(self._modules) == 1:
        only_sub_mod = list(self._modules.values())[0]
        return get_last_conv_module(only_sub_mod)
    raise Exception("Cannot auto induce the last conv module of mod {}, "
                    "Specificy `last_conv_module` attribute!`".format(self))

nn.Module.get_last_conv_module = get_last_conv_module

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class inspectBlock(torch.nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super(inspectBlock, self).__init__()
        self.op1 = ReLUConvBN(C_in, C_out, kernel_size=3, stride=stride,
                              padding=1, affine=affine)
        self.op2 = ReLUConvBN(C_in, C_out, kernel_size=1, stride=stride,
                              padding=0, affine=affine)
        self.op3 = SepConv(C_in, C_out, kernel_size=3, stride=stride,
                           padding=1, affine=affine)

    def forward(self, x):
        rand_ = np.random.random()
        if rand_ < 1./3.:
            return self.op1(x)
        if rand_ < 2./3.:
            return self.op2(x)
        return self.op3(x)


class NB201ResidualBlock(nn.Module):

    def __init__(self, inplanes, planes, stride, affine=True):
        super(NB201ResidualBlock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, affine)
        self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, affine)
        else:
            self.downsample = None
        self.in_dim  = inplanes
        self.out_dim = planes
        self.stride  = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs, genotype=None, **kwargs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock

    def sub_named_members(self, genotype, prefix="", member="parameters", check_visited=False):
        _func = getattr(self, "named_" + member)
        for n, v in _func(prefix):
            yield n, v

# ---- added for rnn ----
# from https://github.com/carpedm20/ENAS-pytorch
class EmbeddingDropout(torch.nn.Embedding):
    """Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.
    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).
    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).
    """
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 dropout=0.1,
                 scale=None):
        """Embedding constructor.
        Args:
            dropout: Dropout probability.
            scale: Used to scale parameters of embedding weight matrix that are
                not dropped out. Note that this is _in addition_ to the
                `1/(1 - dropout)` scaling.
        See `torch.nn.Embedding` for remaining arguments.
        """
        torch.nn.Embedding.__init__(self,
                                    num_embeddings=num_embeddings,
                                    embedding_dim=embedding_dim,
                                    max_norm=max_norm,
                                    norm_type=norm_type,
                                    scale_grad_by_freq=scale_grad_by_freq,
                                    sparse=sparse)
        self.dropout = dropout
        assert 1.0 > dropout >= 0.0, "Dropout must be >= 0.0 and < 1.0"
        self.scale = scale

    def forward(self, inputs):
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * mask
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale

        return F.embedding(inputs,
                           masked_weight,
                           max_norm=self.max_norm,
                           norm_type=self.norm_type,
                           scale_grad_by_freq=self.scale_grad_by_freq,
                           sparse=self.sparse)

class LockedDropout(nn.Module):
    """
    Variational dropout: same dropout mask at each time step. Gal and Ghahramani (2015).

    Ref: https://github.com/salesforce/awd-lstm-lm/
    """

    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        # batch_size, num_hidden
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.div_(1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class ResFactorizedReduceBlock(nn.Module):
    def __init__(self, C, C_out, stride, affine):
        super(ResFactorizedReduceBlock, self).__init__()
        kernel_size = 1
        padding = int((kernel_size - 1) / 2)
        self.op_1 = ReLUConvBN(
            C, C_out, kernel_size, stride,
            padding, affine=affine) if stride == 1 \
            else FactorizedReduce(C, C_out, stride=stride, affine=affine)
        self.op_2 = ReLUConvBN(C_out, C_out,
                               kernel_size, 1, padding, affine=affine)
        self.skip_op = Identity() if stride == 1 and C == C_out else \
                       ConvBNReLU(C, C_out, 1, stride, 0, affine=affine)

    def forward(self, inputs):
        inner = self.op_1(inputs)
        out = self.op_2(inner)
        out_skip = self.skip_op(inputs)
        return out + out_skip

    def forward_one_step(self, context=None, inputs=None):
        raise NotImplementedError()

class ChannelConcat(nn.Module):
    @property
    def is_elementwise(self):
        return False

    def forward(self, states):
        return torch.cat(states, dim=1)

class ElementwiseAdd(nn.Module):
    @property
    def is_elementwise(self):
        return True

    def forward(self, states):
        return sum(states)

class ElementwiseMean(nn.Module):
    @property
    def is_elementwise(self):
        return True

    def forward(self, states):
        return sum(states) / len(states)

class ElementwiseMul(nn.Module):
    @property
    def is_elementwise(self):
        return True

    def forward(self, states):
        res = states[0]
        for state in states[1:]:
            res *= state
        return res

CONCAT_OPS = {
    "concat": ChannelConcat,
    "sum": ElementwiseAdd,
    "mean": ElementwiseMean,
    "mul": ElementwiseMul,
}

def get_concat_op(type_):
    return CONCAT_OPS[type_]()


# ---- added for OFA ----
class FlexibleLayer(object):
    def __init__(self):
        self.reset_mask()

    def set_mask(self, *args, **kwargs):
        raise NotImplementedError()

    def reset_mask(self):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class FlexiblePointLinear(nn.Conv2d, FlexibleLayer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(FlexiblePointLinear, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=bias)
        FlexibleLayer.__init__(self)
        self._bias = bias

    def _select_params(self, in_mask=None, out_mask=None):
        if in_mask is None and out_mask is None:
            return self.weight, self.bias
        if out_mask is None:
            return self.weight[:, in_mask, :, :].contiguous(), self.bias

        bias = None if not self._bias else self.bias[out_mask].contiguous()
        if in_mask is None:
            return self.weight[out_mask, :, :, :].contiguous(), bias
        return self.weight[:, in_mask, :, :][out_mask, :, :, :].contiguous(), bias


    def set_mask(self, in_mask, out_mask):
        self.in_mask = in_mask
        self.out_mask = out_mask
    
    def reset_mask(self):
        self.in_mask = None
        self.out_mask = None

    def forward_mask(self, inputs, in_mask=None, out_mask=None):
        filters, bias = self._select_params(in_mask, out_mask)
        padding = max(self.kernel_size) // 2
        return F.conv2d(inputs, filters, bias=bias, padding=padding, stride=self.stride, dilation=self.dilation)

    def forward(self, inputs):
        return self.forward_mask(inputs, self.in_mask, self.out_mask)

    def finalize(self):
        """
        The method should be called only after set_mask is called, or there will be no effect.
        """
        weight, bias = self._select_params(self.in_mask, self.out_mask)
        C_out, C_in, H, W = weight.shape
        assert H == W
        padding = H // 2
        final_conv = nn.Conv2d(C_in, C_out, H, stride=self.stride, padding=padding, bias=self._bias)
        final_conv.weight.data.copy_(weight)
        if self._bias:
            final_conv.bias.data.copy_(bias)
        return final_conv

class FlexibleDepthWiseConv(nn.Conv2d, FlexibleLayer):
    def __init__(self, in_channels, kernel_sizes, stride=1, dilation=1, do_kernel_transform=True, bias=False):
        assert isinstance(kernel_sizes, (list, tuple))
        kernel_sizes = sorted(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.max_kernel_size = kernel_sizes[-1]
        self.do_kernel_transform = do_kernel_transform
        super(FlexibleDepthWiseConv, self).__init__(in_channels, in_channels, self.max_kernel_size, stride, dilation=dilation, groups=in_channels, bias=bias)
        if self.do_kernel_transform:
            for smaller, larger in reversed(list(zip(self.kernel_sizes[:-1], self.kernel_sizes[1:]))):
                if self.max_kernel_size >= larger:
                    kernel_transform_matrix = nn.Linear(
                        smaller * smaller, smaller * smaller, bias=False)
                    torch.nn.init.eye_(kernel_transform_matrix.weight.data)
                    setattr(self, "linear_{}to{}".format(larger, smaller), kernel_transform_matrix)

        FlexibleLayer.__init__(self)
        self._bias = bias

    def _select_channels(self, mask):
        return self.weight[mask, :, :, :].contiguous(), self.bias[mask].contiguous() if self.bias else None

    def _transform_kernel(self, origin_filter, kernel_size):
        expect(kernel_size in self.kernel_sizes, "The kernel_size must be one of {}, got {} instead".format(self.kernel_sizes, kernel_size), ValueError)
        if origin_filter.shape[-1] == kernel_size:
            return origin_filter
        if not self.do_kernel_transform:
            return get_sub_kernel(origin_filter, kernel_size)
        cur_filter = origin_filter
        expect(cur_filter.shape[-1] > kernel_size, "The kernel size must be less than origin kernel size {}, got {} instead.".format(origin_filter.shape[-1], kernel_size), ValueError)
        for smaller, larger in reversed(list(zip(self.kernel_sizes[:-1], self.kernel_sizes[1:]))):
            if cur_filter.shape[-1] < larger:
                continue
            if kernel_size >= larger:
                break
            sub_filter = get_sub_kernel(origin_filter, smaller).view(cur_filter.shape[0], cur_filter.shape[1], -1)
            sub_filter = sub_filter.view(-1, sub_filter.shape[-1])
            sub_filter = getattr(self, "linear_{}to{}".format(larger, smaller))(sub_filter)
            sub_filter = sub_filter.view(origin_filter.shape[0], origin_filter.shape[1], smaller ** 2)
            sub_filter = sub_filter.view(origin_filter.shape[0], origin_filter.shape[1], smaller, smaller)
            cur_filter = sub_filter
        return cur_filter

    def _select_params(self, mask, kernel_size):
        filters = self.weight
        bias = self.bias
        if mask is not None:
            filters, bias = self._select_channels(mask)
        if kernel_size:
            filters = self._transform_kernel(filters, kernel_size)
        return filters, bias

    def set_mask(self, mask, kernel_size):
        self.mask = mask
        self._kernel_size = kernel_size

    def reset_mask(self):
        self.mask = None
        self._kernel_size = self.max_kernel_size

    def forward_mask(self, inputs, mask=None, kernel_size=None):
        filters, bias = self._select_params(mask, kernel_size)
        padding = filters.shape[-1] // 2
        groups = filters.shape[0]
        return F.conv2d(inputs, filters, bias=bias, padding=padding, stride=self.stride, dilation=self.dilation, groups=groups)

    def forward(self, inputs):
        return self.forward_mask(inputs, self.mask, self._kernel_size)

    def finalize(self):
        """
        The method should be called only after set_mask is called, or there will be no effect.
        """
        weight, bias = self._select_params(self.mask, self._kernel_size)
        C, _, _, _ = weight.shape
        padding = self._kernel_size // 2
        final_conv = nn.Conv2d(C, C, self._kernel_size, stride=self.stride, padding=padding, groups=C, bias=self._bias)
        final_conv.weight.data.copy_(weight)
        if self._bias:
            final_conv.weight.data.copy_(bias)
        return final_conv


class FlexibleBatchNorm2d(nn.Module, FlexibleLayer):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(FlexibleBatchNorm2d, self).__init__()
        self.flex_bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        FlexibleLayer.__init__(self)

    def _select_params(self, mask):
        if mask is not None:
            running_mean = self.flex_bn.running_mean[mask].contiguous()
            running_var = self.flex_bn.running_var[mask].contiguous()
            weight = self.flex_bn.weight[mask].contiguous()
            bias = self.flex_bn.bias[mask].contiguous()
            return running_mean, running_var, weight, bias
        else:
            return self.flex_bn.running_mean, self.flex_bn.running_var, self.flex_bn.weight, self.flex_bn.bias

    def set_mask(self, mask):
        self.mask = mask

    def reset_mask(self):
        self.mask = None
    
    def forward_mask(self, inputs, mask=None):
        if mask is None or inputs.shape[1] == self.flex_bn.num_features:
            return self.flex_bn.forward(inputs)
        
        """ _BatchNorm official code"""
        if self.flex_bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.flex_bn.momentum

        if self.flex_bn.training and self.flex_bn.track_running_stats:
            if self.flex_bn.num_batches_tracked is not None:
                self.flex_bn.num_batches_tracked += 1
                if self.flex_bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.flex_bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.flex_bn.momentum
        running_mean, running_var, weight, bias = self._select_params(mask)
        return F.batch_norm(
            inputs, running_mean, running_var, weight, bias,
            self.flex_bn.training or not self.flex_bn.track_running_stats,
            exponential_average_factor, self.flex_bn.eps)

    def forward(self, inputs):
        return self.forward_mask(inputs, self.mask)

    def finalize(self):
        """
        The method should be called only after set_mask is called, or there will be no effect.
        """
        running_mean, running_var, weight, bias = self._select_params(self.mask)
        feature_dim = weight.shape[0]
        final_bn = nn.BatchNorm2d(feature_dim, self.flex_bn.eps, self.flex_bn.momentum, self.flex_bn.affine)
        for var in ["running_mean", "running_var", "weight", "bias"]:
            getattr(final_bn, var).data.copy_(eval(var))
        return final_bn


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4, reduction_layer=None, expand_layer=None):
        super(SEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        mid_channel = make_divisible(channel // reduction, 8)

        self.se = nn.Sequential(OrderedDict([
            ("reduction", reduction_layer or nn.Conv2d(self.channel, mid_channel, 1, 1, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("expand", expand_layer or nn.Conv2d(mid_channel, self.channel, 1, 1, 0)),
            ("activation", get_op("h_sigmoid")())
        ]))

    def forward(self, inputs):
        out = inputs.mean(3, keepdim=True).mean(2, keepdim=True)
        out = self.se(out)
        return inputs * out


class FlexibleSEModule(SEModule, FlexibleLayer):
    def __init__(self, channel, reduction=4):
        mid_channel = make_divisible(channel // reduction, 8)
        reduction_layer = FlexiblePointLinear(channel, mid_channel, 1, 1, 0, bias=True)
        expand_layer = FlexiblePointLinear(mid_channel, channel, 1, 1, 0, bias=True)
        super(FlexibleSEModule, self).__init__(channel, reduction, reduction_layer, expand_layer)
        FlexibleLayer.__init__(self)

    def reset_mask(self):
        self.se.reduction.reset_mask()
        self.se.expand.reset_mask()

    def set_mask(self, mask):
        if mask is None:
            return
        channel = mask.sum().item()
        mid_channel = make_divisible(channel // self.reduction, 8)
        exp_mask = _get_channel_mask(self.se.expand.weight.data, mid_channel)
        self.se.reduction.set_mask(mask, exp_mask)
        self.se.expand.set_mask(exp_mask, mask)
        
    def forward(self, inputs):
        out = inputs.mean(3, keepdim=True).mean(2, keepdim=True)
        out = self.se(out)
        return inputs * out

    def finalize(self):
        reduction_layer = self.se.reduction.finalize()
        expand_layer = self.se.expand.finalize()
        return SEModule(self.channel, self.reduction, reduction_layer, expand_layer)


# ---- added for Detection ----
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
        
def SeparableConv(in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    activation="relu", 
                    norm=True,
                    final_activation=None):
    """
    A simple separable conv.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(in_channels),
        get_op(activation)(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=not norm),
        nn.BatchNorm2d(out_channels) if norm else nn.Sequential(),
        get_op(final_activation)() if final_activation else nn.Sequential()
    )

def ConvModule(in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=1,
               activation="relu",
               norm=True,
               final_activation=None):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not
            norm),
        nn.BatchNorm2d(out_channels) if norm else nn.Sequential(),
        get_op(final_activation)() if final_activation else nn.Sequential()
    )

class FlexibleSepConv(nn.Module, FlexibleLayer):

    def __init__(self, in_channels, out_channels, kernel_sizes=[3], stride=1, activation=None,
            norm=True, final_activation=None):
        super(FlexibleSepConv, self).__init__()
        FlexibleLayer.__init__(self)

        self.kernel_sizes = kernel_sizes
        self.final_activation = final_activation

        #if max(kernel_sizes) > 1:
        self.depthwise_conv = FlexibleDepthWiseConv(in_channels, kernel_sizes,
                                                      stride=stride, bias=not norm)
        #else:
        #    self.depthwise_conv  = nn.Sequential()
        self.pointwise_conv = FlexiblePointLinear(in_channels, out_channels, bias=not norm)

        self.norm = norm
        if self.norm:
            self.bn0 = [FlexibleBatchNorm2d(in_channels)]
            if activation is not None:
                self.bn0 += [get_op(activation)()]
            self.bn0 = nn.Sequential(*self.bn0)

            self.bn1 = FlexibleBatchNorm2d(out_channels)

        if self.final_activation is not None:
            self.act = get_op(final_activation)()


        self.reset_mask()

    def forward(self, x):
        x = self.depthwise_conv(x)
        if self.norm:
            x = self.bn0(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn1(x)

        if self.final_activation:
            x = self.act(x)
        return x

    def set_mask(self, mask):
        pass

    def reset_mask(self):
        pass

    def finalize(self):
        return self


class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

