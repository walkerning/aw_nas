# register the ops for layer2 ss
import torch
import torch.nn as nn
from aw_nas.ops import register_primitive

register_primitive(
    "none_for_layer2",
    lambda C, C_out, stride, affine: ZeroLayer2(C, C_out, stride),
)


class ZeroLayer2(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(ZeroLayer2, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = self.C_out
        shape[2] = shape[2] // self.stride
        shape[3] = shape[3] // self.stride
        return torch.zeros(size=shape, dtype=x.dtype, device=x.device)


from aw_nas.btcs.layer2 import (
    search_space,
    controller,
    final_model,
    bi_final_model,
    weights_manager,
    diff_weights_manager,
)
