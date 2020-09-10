"""
Layer 2 final model.
"""

class Layer2FinalModel(FinalModel):
    NAME = "layer2_final_model"

    def __init__(
        self,
        search_space,
        device,
        genotypes,
        num_classes=10,
        init_channels=36,
        stem_multiplier=1,
        dropout_rate=0.0,
        dropout_path_rate=0.0,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        schedule_cfg=None,
    ):
        pass

    @classmethod
    def supported_data_types(cls):
        return ["image"]

