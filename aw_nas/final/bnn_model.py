# from __future__ import print_function

from torch import nn

from aw_nas import ops
from aw_nas.final.cnn_model import CNNGenotypeModel

class BNNGenotypeModel(CNNGenotypeModel):
    NAME = "bnn_final_model"

    def __init__(self, *args, **kwargs):
        super(BNNGenotypeModel, self).__init__(*args, **kwargs)
        self.bi_flops = 0
        self._bi_flops_calculated = 0 # for report bi_flops only

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += 2* inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    outputs.size(2) * outputs.size(3) / module.groups
            elif isinstance(module, ops.BinaryConv2d):
                # 1-bit conv
                self.bi_flops += 2* inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size * module.kernel_size * \
                                    outputs.size(2) * outputs.size(3) / (module.groups)
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    def forward(self, *args, **kwargs):
        res = super(BNNGenotypeModel, self).forward(*args, **kwargs)

        if not self._bi_flops_calculated and self.bi_flops > 0:
            self.logger.info("BiOPS: binary ops num = %d M", self.bi_flops/1.e6)
            self._bi_flops_calculated = True
        return res
