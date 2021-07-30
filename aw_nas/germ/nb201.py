"""
Run `python scripts/generate_germ_search_cfg.py aw_nas/germ/nb201.py aw_nas/germ/nb201.yaml`
to generate the search space cfg and the partial search cfg
"""
# pylint: disable=arguments-differ

import torch
from torch import nn

from aw_nas import ops, germ


class GermNB201Net(germ.GermSuperNet):
    NAME = "nb201"

    def __init__(
        self,
        search_space,
        op_list=[
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        ],
        num_classes=10,
        num_layers=17,
        init_channels=16,
        stem_multiplier=1,
        dropout_rate=0.1,
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
    ):
        super(GermNB201Net, self).__init__(search_space)

        # search space configs
        self.op_list = op_list
        self.num_op = len(op_list)
        self._vertices = 4

        # task related
        self.num_classes = num_classes

        # supernet
        self.num_layers = num_layers
        # init channel number of the first cell layers,
        # x2 after every reduce cell
        self.init_channels = init_channels
        # channels of stem conv / init_channels
        self.stem_multiplier = stem_multiplier
        self.use_stem = use_stem
        self.dropout_rate = dropout_rate

        # choices
        self.op_choices_dict = germ.DecisionDict()
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                key = "f_{}_t_{}".format(from_, to_)
                self.op_choices_dict[key] = germ.Choices(
                    choices=list(range(self.num_op)), size=1
                )

        ## initialize sub modules
        if not self.use_stem:
            c_stem = 3
        elif isinstance(self.use_stem, (list, tuple)):
            self.stems = []
            c_stem = self.stem_multiplier * self.init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stems.append(
                    ops.get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )
            self.stem = nn.Sequential(self.stems)
        else:
            c_stem = self.stem_multiplier * self.init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )

        with self.begin_searchable() as ctx:
            self.cells = nn.ModuleList()
            num_channels = self.init_channels
            strides = [
                2 if self._is_reduce(i_layer) else 1
                for i_layer in range(self.num_layers)
            ]

            for i_layer, stride in enumerate(strides):
                _num_channels = num_channels if i_layer != 0 else c_stem
                if stride > 1:
                    num_channels *= stride
                # A patch: Can specificy input/output channels by hand in configuration,
                # instead of relying on the default
                # "whenever stride/2, channelx2 and mapping with preprocess operations" assumption
                _num_out_channels = num_channels
                if stride == 1:
                    cell = GermNB201Cell(
                        ctx,
                        layer_index=i_layer,
                        op_list=self.op_list,
                        op_choices=self.op_choices_dict,
                        num_channels=_num_channels,
                        num_out_channels=_num_out_channels,
                        stride=stride,
                    )
                else:
                    cell = ops.get_op("NB201ResidualBlock")(
                        _num_channels, _num_out_channels, stride=2, affine=True
                    )
                self.cells.append(cell)

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(num_channels), nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(num_channels, self.num_classes)

    def _is_reduce(self, layer_idx):
        return layer_idx in [
            (self.num_layers + 1) // 3 - 1,
            (self.num_layers + 1) * 2 // 3 - 1,
        ]

    # ---- APIs ----
    def forward(self, inputs, **kwargs):  # pylint: disable=arguments-differ
        if not self.use_stem:
            states = inputs
        else:
            stemed = self.stem(inputs)
            states = stemed
        for cell in self.cells:
            states = cell(states, **kwargs)
        out = self.lastact(states)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    # A totally out-place realization, not necessary now
    # def finalize_rollout(self, rollout):
    #     modules = []
    #     # stem
    #     if self.use_stem:
    #         modules.append(("stem", self.stem))

    #     # searchable cells
    #     for i_cell, cell in enumerate(self.cells):
    #         if isinstance(cell, germ.SearchableBlock):
    #             module = cell.finalize_rollout(rollout)
    #         else:
    #             module = cell
    #         modules.append(("cell_{}".format(i_cell), module))

    #     # final modules
    #     modules += [("lastact", self.lastact), ("global_pooling", self.global_pooling),
    #                 ("dropout", self.dropout), ("flatten", nn.Flatten()),
    #                 ("classifier", self.classifier)]
    #     modules = nn.Sequential(OrderedDict(modules))
    #     return modules


class GermNB201Cell(germ.SearchableBlock):
    def __init__(
        self,
        ctx,
        layer_index,
        op_list,
        num_channels,
        num_out_channels,
        stride,
        op_choices=None,
    ):
        super(GermNB201Cell, self).__init__(ctx)
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.num_out_channels = num_out_channels
        self.layer_index = layer_index
        self.op_choices = (
            op_choices  # None indicates independent cell arch for each layer
        )

        self._vertices = 4
        self.op_list = op_list

        self.edges = nn.ModuleDict()
        for from_ in range(self._vertices):
            for to_ in range(from_ + 1, self._vertices):
                key = "f_{}_t_{}".format(from_, to_)
                edge = germ.GermMixedOp(
                    ctx,
                    op_list=self.op_list,
                    op_choice=self.op_choices[key]
                    if self.op_choices is not None
                    else None,
                    C=self.num_channels,
                    C_out=self.num_out_channels,
                    stride=self.stride,
                    affine=False,
                )
                self.edges[key] = edge

    def forward(self, inputs):
        states_ = [inputs]
        for to_ in range(1, self._vertices):
            state_ = torch.zeros(inputs.shape).to(inputs.device)
            for from_ in range(to_):
                key = "f_{}_t_{}".format(from_, to_)
                out = self.edges[key](states_[from_])
                state_ = state_ + out
            states_.append(state_)
        return states_[-1]

    def forward_rollout(self, rollout, inputs):
        raise Exception("Should not be called")
