"""
NN operations.
"""
#pylint: disable=arguments-differ,useless-super-delegation,invalid-name

import torch
from torch import nn
import torch.nn.functional as F

def avg_pool_3x3(C, C_out, stride, affine):
    assert C == C_out
    return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

def max_pool_3x3(C, C_out, stride, affine):
    assert C == C_out
    return nn.MaxPool2d(3, stride=stride, padding=1)

def conv_7x1_1x7(C, C_out, stride, affine):
    assert C == C_out
    return nn.Sequential(
        # C_out is ignored
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    )

PRIMITVE_FACTORY = {
    "none" : lambda C, C_out, stride, affine: Zero(stride),
    "avg_pool_3x3" : avg_pool_3x3,
    "max_pool_3x3" : max_pool_3x3,
    "skip_connect" : lambda C, C_out, stride, affine: Identity() if stride == 1 \
      else FactorizedReduce(C, C_out, stride=stride, affine=affine),
    "sep_conv_3x3" : lambda C, C_out, stride, affine: SepConv(C, C_out,
                                                              3, stride, 1, affine=affine),
    "sep_conv_5x5" : lambda C, C_out, stride, affine: SepConv(C, C_out,
                                                              5, stride, 2, affine=affine),
    "sep_conv_7x7" : lambda C, C_out, stride, affine: SepConv(C, C_out,
                                                              7, stride, 3, affine=affine),
    "dil_conv_3x3" : lambda C, C_out, stride, affine: DilConv(C, C_out,
                                                              3, stride, 2, 2, affine=affine),
    "dil_conv_5x5" : lambda C, C_out, stride, affine: DilConv(C, C_out,
                                                              5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7" : conv_7x1_1x7,

    # activations
    "tanh": lambda **kwargs: nn.Tanh(),
    "relu": lambda **kwargs: nn.ReLU(),
    "sigmoid": lambda **kwargs: nn.Sigmoid(),
    "identity": lambda **kwargs: Identity()
}

def register_primitive(name, func, override=False):
    assert callable(func), "A primtive must be callable"
    assert not (name in PRIMITVE_FACTORY and not override),\
        "some func already registered as {};"\
        " to override, use `override=True` keyword arguments.".format(name)
    PRIMITVE_FACTORY[name] = func

def get_op(name):
    assert name in PRIMITVE_FACTORY, \
        "{} not registered, use `register_primitive` to register primitive op".format(name)
    return PRIMITVE_FACTORY[name]

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        group_dim = C_out // stride

        self.convs = [nn.Conv2d(C_in, group_dim, kernel_size=1,
                                stride=stride, padding=0, bias=False)\
                      for _ in range(stride)]
        self.convs = nn.ModuleList(self.convs)

        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        mod = x.size(2) % self.stride
        if mod != 0:
            pad = self.stride - mod
            x = F.pad(x, (pad, 0, pad, 0), "constant", 0)
        out = torch.cat([conv(x[:, :, i:, i:]) for i, conv in enumerate(self.convs)], dim=1)
        out = self.bn(out)
        return out

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


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


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
