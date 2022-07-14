from functools import partial
from collections import abc as collection_abcs

import numpy as np
import torch
from torch import nn

from pycls.models.anynet import (
    get_stem_fun,
    gap2d,
    activation,
    BasicTransform,
    AnyHead,
)

from aw_nas import germ
from aw_nas import utils
from aw_nas.utils import getLogger
from aw_nas.utils.parallel_utils import parallelize

def _round_to_int(float_):
    return int(round(float_))

def _init_weights(m):
    """
    Performs ResNet-style weight initialization.
    Add a check for BN without affine transformation (for search process)
    """
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


## ---- BEGIN ResNet germ block ----
class GermBasicTransform(germ.SearchableBlock):
    """Germ Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, ctx, w_in, w_out, stride, _params):
        super(GermBasicTransform, self).__init__(ctx)
        self.layer_a = germ.SearchableConvBNBlock(
            ctx,
            w_in,
            w_out,
            3,
            stride=stride,
            force_use_ordinal_channel_handler=_params.get(
                "force_use_ordinal_channel_handler", False
            ),
            norm_cfg={"affine": _params["bn_affine"]},
        )
        self.act = activation()
        self.layer_b = germ.SearchableConvBNBlock(
            ctx,
            w_out,
            w_out,
            3,
            force_use_ordinal_channel_handler=_params.get(
                "force_use_ordinal_channel_handler", False
            ),
            norm_cfg={"affine": _params["bn_affine"]},
        )
        self.layer_b.bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class GermResBasicBlock(germ.SearchableBlock):
    """Germ Residual basic block: x + f(x), func = basic transform."""

    def __init__(self, ctx, w_in, w_out, stride, params):
        super(GermResBasicBlock, self).__init__(ctx)
        self.proj_bn = None
        if (w_in != w_out) or (stride != 1):
            # the first block in each stage
            self.proj_bn = germ.SearchableConvBNBlock(
                ctx,
                w_in,
                w_out,
                kernel_size=1,
                stride=stride,
                force_use_ordinal_channel_handler=params.get(
                    "force_use_ordinal_channel_handler", False
                ),
                norm_cfg={"affine": params["bn_affine"]},
            )
        self.func = GermBasicTransform(ctx, w_in, w_out, stride, params)
        self.act = activation()

    def forward(self, x):
        x_p = self.proj_bn(x) if self.proj_bn else x
        return self.act(x_p + self.func(x))


## ---- END ResNet germ block ----


## ---- BEGIN ResNext germ block ----
class GermSE(germ.SearchableBlock):
    """Germ Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, ctx, w_in, w_se, force_use_ordinal_channel_handler=False):
        super(GermSE, self).__init__(ctx)
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.Sequential(
            germ.SearchableConv(
                ctx,
                w_in,
                w_se,
                1,
                bias=True,
                force_use_ordinal_channel_handler=force_use_ordinal_channel_handler,
            ),
            activation(),
            germ.SearchableConv(
                ctx,
                w_se,
                w_in,
                1,
                bias=True,
                force_use_ordinal_channel_handler=force_use_ordinal_channel_handler,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class GermBottleneckTransform(germ.SearchableBlock):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, ctx, w_in, w_out, stride, params):
        super(GermBottleneckTransform, self).__init__(ctx)
        self.bot_mul = params["bot_mul"]
        self.se_r = params["se_r"]
        w_b = self.w_b = (w_out * self.bot_mul).apply(_round_to_int)

        if self.se_r:
            w_se = w_in * self.se_r
            if isinstance(w_se, germ.BaseDecision):
                w_se = w_se.apply(_round_to_int)
            else:
                w_se = int(round(w_se))
            self.w_se = w_se
        self.a_convbn = germ.SearchableConvBNBlock(
            ctx,
            w_in,
            w_b,
            1,
            force_use_ordinal_channel_handler=params.get(
                "force_use_ordinal_channel_handler", False
            ),
            norm_cfg={"affine": params["bn_affine"]},
        )

        self.a_act = activation()
        num_group = params["num_group"]
        self.b_convbn = germ.SearchableConvBNBlock(
            ctx,
            w_b,
            w_b,
            3,
            stride=stride,
            groups=num_group,
            force_use_ordinal_channel_handler=params.get(
                "force_use_ordinal_channel_handler", False
            ),
            norm_cfg={"affine": params["bn_affine"]},
        )
        self.b_af = activation()
        if self.se_r:
            self.se_block = GermSE(
                ctx,
                w_b,
                w_se,
                force_use_ordinal_channel_handler=params.get(
                    "force_use_ordinal_channel_handler", False
                ),
            )
        else:
            self.se_block = None
        self.c_convbn = germ.SearchableConvBNBlock(
            ctx,
            w_b,
            w_out,
            1,
            force_use_ordinal_channel_handler=params.get(
                "force_use_ordinal_channel_handler", False
            ),
            norm_cfg={"affine": params["bn_affine"]},
        )
        self.c_convbn.bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class GermResBottleneckBlock(germ.SearchableBlock):
    """Germ Residual bottleneck block: x + func(x), func = bottleneck transform."""

    def __init__(self, ctx, w_in, w_out, stride, params):
        super(GermResBottleneckBlock, self).__init__(ctx)
        self.proj_bn = None
        if (w_in != w_out) or (stride != 1):
            self.proj_bn = germ.SearchableConvBNBlock(
                ctx,
                w_in,
                w_out,
                kernel_size=1,
                stride=stride,
                force_use_ordinal_channel_handler=params.get(
                    "force_use_ordinal_channel_handler", False
                ),
                norm_cfg={"affine": params["bn_affine"]},
            )
        self.func = GermBottleneckTransform(ctx, w_in, w_out, stride, params)
        self.act = activation()

    def forward(self, x):
        x_p = self.proj_bn(x) if self.proj_bn else x
        return self.act(x_p + self.func(x))


## ---- END ResNext germ block ----


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        # "vanilla_block": GermVanillaBlock,
        "res_basic_block": GermResBasicBlock,
        "res_bottleneck_block": GermResBottleneckBlock,
        # "res_bottleneck_linear_block": GermResBottleneckLinearBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class GermAnyStage(germ.SearchableBlock):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, ctx, w_in, w_out, depth, stride, block_fun, params):
        super(GermAnyStage, self).__init__(ctx)
        max_depth = max(depth.choices) if isinstance(depth, germ.Choices) else depth
        self.depth = depth
        self.blocks = []
        for i in range(max_depth):
            block = block_fun(ctx, w_in, w_out, stride, params)
            self.blocks.append(block)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        _depth = self._get_decision(self.depth, self.ctx.rollout)
        for block in self.blocks[:_depth]:
            x = block(x)
        return x


class GermAnyHead(germ.SearchableBlock):
    """
    Germ AnyNet head: optional conv, AvgPool, 1x1.
    Only `w_in` can be a Germ.Choice.
    """

    def __init__(self, ctx, w_in, head_width, num_classes, _params):
        super(GermAnyHead, self).__init__(ctx)
        self.head_width = head_width
        if head_width > 0:
            self.conv_bn = germ.SearchableConvBNBlock(
                ctx,
                w_in,
                head_width,
                1,
                force_use_ordinal_channel_handler=_params.get(
                    "force_use_ordinal_channel_handler", False
                ),
                norm_cfg={"affine": _params["bn_affine"]},
            )
            self.act = activation()
            w_in = head_width
        self.avg_pool = gap2d(w_in)
        # use a conv1x1 as fc layer
        self.fc_layer = germ.SearchableConv(
            ctx,
            w_in,
            num_classes,
            kernel_size=1,
            bias=True,
            force_use_ordinal_channel_handler=_params.get(
                "force_use_ordinal_channel_handler", False
            ),
        )

    def forward(self, x):
        x = self.act(self.conv_bn(x)) if self.head_width > 0 else x
        x = self.avg_pool(x)
        x = self.fc_layer(x)
        x = x.view(x.size(0), -1)
        return x


class GermAnyNet(germ.GermSuperNet):
    """Germ AnyNet model."""

    @staticmethod
    def prepare_decisions(stages_choices):
        decision_list = []
        for stage_choices in stages_choices:
            if isinstance(stage_choices, collection_abcs.Iterable):
                stage_choices = germ.Choices(stage_choices)
            decision_list.append(stage_choices)
        return decision_list

    def __init__(
        self,
        search_space,
        block_type,
        depths,
        widths,
        strides,
        bot_muls,
        num_groups,
        stem_type,
        stem_w,
        num_classes,
        use_se=False,
        se_r=0.25,
        head_w=0,
        bn_affine=False,
        force_use_ordinal_channel_handler=False,
        stage_cls=GermAnyStage,
    ):
        super(GermAnyNet, self).__init__(search_space)

        self.use_se = use_se
        self.se_r = se_r
        self.bn_affine = bn_affine
        self.block_type = block_type

        assert all(
            stage_configs is None or len(stage_configs) == 3
            for stage_configs in [depths, widths, strides, bot_muls, num_groups]
        )
        self.depths = germ.DecisionList(self.prepare_decisions(depths))
        self.widths = germ.DecisionList(self.prepare_decisions(widths))
        self.strides = strides  # not searchable
        if bot_muls is not None:
            self.bot_muls = germ.DecisionList(self.prepare_decisions(bot_muls))
        else:
            # do not search bottleneck width ratio
            self.bot_muls = [None] * 3
        if num_groups is not None:
            self.num_groups = germ.DecisionList(self.prepare_decisions(num_groups))
        else:
            # do not search number of groups
            self.num_groups = [None] * 3

        # construct stem
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        prev_w = stem_w

        # construct searchable part
        self.stages = nn.ModuleList()
        block_fun = get_block_fun(block_type)
        keys = ["depth", "width", "stride", "bot_mul", "num_group"]
        with self.begin_searchable() as ctx:
            for i, (depth, width, stride, bot_mul, num_group) in enumerate(
                zip(
                    *[
                        self.depths,
                        self.widths,
                        self.strides,
                        self.bot_muls,
                        self.num_groups,
                    ]
                )
            ):
                self.logger.info(
                    "Constructing stage {}:\n\t{}".format(
                        i,
                        "\n\t".join(
                            [
                                "{:8} choices: {}".format(key, _choices)
                                for key, _choices in zip(
                                    keys, [depth, width, stride, bot_mul, num_group]
                                )
                            ]
                        ),
                    )
                )
                params = {
                    "bot_mul": bot_mul,
                    "num_group": num_group,
                    "se_r": se_r if use_se else 0,
                    "bn_affine": bn_affine,
                    "force_use_ordinal_channel_handler": force_use_ordinal_channel_handler,
                }
                stage = stage_cls(
                    ctx, w_in=prev_w, w_out=width, depth=depth,
                    stride=stride, block_fun=block_fun, params=params
                )
                self.stages.append(stage)
                prev_w = width

        # construct head
        self.head = GermAnyHead(ctx, prev_w, head_w, num_classes, params)

        # initialize weights
        self.apply(_init_weights)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x


class GermResNet(GermAnyNet):
    NAME = "nds_resnet"

    def __init__(
        self,
        search_space,
        num_classes=10,
        stem_type="res_stem_cifar",
        force_ordinal=False,
        depths=[[1, 2, 3, 4, 6, 8, 12, 16, 24]] * 3,
        widths=[[16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]] * 3,
    ):
        super(GermResNet, self).__init__(
            search_space,
            block_type="res_basic_block",
            depths=depths,
            widths=widths,
            strides=[1, 2, 2],
            bot_muls=None,
            num_groups=None,
            stem_type=stem_type,
            stem_w=16,
            num_classes=num_classes,
            use_se=False,
            se_r=0.25,
            head_w=0,
            bn_affine=False,
            force_use_ordinal_channel_handler=force_ordinal,
        )


class GermResNextA(GermAnyNet):
    NAME = "nds_resnexta"

    def __init__(
        self,
        search_space,
        num_classes=10,
        stem_type="res_stem_cifar",
        group_search=True,
        force_ordinal=False,
    ):
        super(GermResNextA, self).__init__(
            search_space,
            block_type="res_bottleneck_block",
            depths=[[1, 2, 4, 8, 16]] * 3,
            widths=[[16, 32, 64, 128, 256]] * 3,
            strides=[1, 2, 2],
            bot_muls=[[0.25, 0.5, 1]] * 3,
            num_groups=[[1, 2, 4]] * 3 if group_search else [1, 1, 1],
            stem_type=stem_type,
            stem_w=16,
            num_classes=num_classes,
            use_se=False,
            se_r=0.25,
            head_w=0,
            bn_affine=False,
            force_use_ordinal_channel_handler=force_ordinal,
        )

class GermCloseNet(GermAnyNet):
    def __init__(
        self,
        search_space,
        block_type,
        depths,
        widths,
        strides,
        bot_muls,
        num_groups,
        stem_type,
        stem_w,
        num_classes,
        use_se=False,
        se_r=0.25,
        head_w=0,
        bn_affine=False,
        force_use_ordinal_channel_handler=False,
        num_curriculums=1,
    ):
        super(GermCloseNet, self).__init__(
            search_space,
            block_type,
            depths,
            widths,
            strides,
            bot_muls,
            num_groups,
            stem_type,
            stem_w,
            num_classes,
            use_se,
            se_r,
            head_w,
            bn_affine,
            force_use_ordinal_channel_handler,
            stage_cls=partial(GermCloseStage, num_curriculums=num_curriculums),
        )
        self.num_curriculums = num_curriculums
    
    def forward(self, x, gate, is_training):
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x, gate[i], is_training)
        x = self.head(x)
        return x

    def new_curriculum(self, from_, to_):
        for stage in self.stages:
            stage.new_curriculum(from_, to_)
        self.logger.info("Successfully add a new curriculum of level {}".format(to_ + 1))


class GermCloseStage(germ.SearchableBlock):
    def __init__(self, ctx, w_in, w_out, depth, stride, block_fun, params, num_curriculums):
        super(GermCloseStage, self).__init__(ctx)
        self.depth = depth
        self.num_curriculums = num_curriculums
        self.max_depth = max(depth.choices) if isinstance(depth, germ.Choices) else depth
        self.curriculums = []
        self.preprocess_block = block_fun(ctx, w_in, w_out, stride, params)
        for i in range(self.num_curriculums):
            block = block_fun(ctx, w_out, w_out, 1, params)
            self.curriculums.append(block)
            self.add_module("c{}".format(i + 1), block)
    
    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def new_curriculum(self, from_, to_):
        params_1 = self.curriculums[from_].state_dict()
        params_2 = self.curriculums[to_].state_dict()
        assert len(params_1) == len(params_2)

        for par1, par2 in zip(params_1, params_2):
            params_2[par2] += params_1[par1]

        self.curriculums[to_].load_state_dict(params_2)
        self.logger.info("Successfully add a new global op weight from {} to {}".format(from_, to_))
    
    def forward(self, x, gate, is_training):
        _depth = self._get_decision(self.depth, self.ctx.rollout)
        x = self.preprocess_block(x)
        for i in range(_depth - 1):
            x = self.curriculums[gate[i]](x)
        return x

class GermCloseCandidateNet(germ.GermCandidateNet):
    def __init__(self, weights_manager, rollout, eval_no_grad=True, for_eval=False):
        super(GermCloseCandidateNet, self).__init__(weights_manager, rollout, eval_no_grad)
        self.for_eval = for_eval
        self.gate = rollout.gate

    def forward(self, inputs):
        return self.super_net.forward_rollout(
            self.rollout,
            inputs,
            self.gate,
            is_training=not self.for_eval
        )

class GermCloseWeightsManager(germ.GermWeightsManager):
    NAME = "germ_close"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="germ",
        germ_supernet_type=None,
        germ_supernet_cfg=None,
        germ_def_file=None,
        candidate_eval_no_grad=True,
        max_grad_norm=None,
        schedule_cfg=None,
       
        num_curriculums=1,
        cl_schedule=[1],
        cl_criteria="max",
        eval_arch=False,
    ):
        germ_supernet_cfg["num_curriculums"] = num_curriculums
        super(GermCloseWeightsManager, self).__init__(
            search_space,
            device,
            rollout_type,
            germ_supernet_type,
            germ_supernet_cfg,
            germ_def_file,
            candidate_eval_no_grad,
            max_grad_norm,
            schedule_cfg
        )
    
        self.num_curriculums = num_curriculums
        self.curriculum_level = 1
        self.cl_schedule = cl_schedule
        
        self.num_stages = self.super_net.num_stages
        self.num_positions = self.super_net.num_positions
        self.gate = [[0 for i in range(self.num_positions)] for j in range(self.num_stages)]
        self.groups = {0: (0, self.num_positions)}

        self.cl_criteria = cl_criteria
        self.eval_arch = eval_arch

    def assemble_candidate(self, rollout, for_eval=False):
        rollout.gate = self.gate
        return GermCloseCandidateNet(
            self,
            rollout,
            self.candidate_eval_no_grad,
            for_eval=for_eval
        )

    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger
    
    def split_curriculum(self, key):
        l = self.groups[key][0]
        r = self.groups[key][1]
        m = (l + r) // 2
        self.groups[key] = (l, m)
        for s in range(self.num_stages):
            for i in range(m, r):
                self.gate[s][i] = len(self.groups)
        if not self.eval_arch:
            self.super_net.new_curriculum(key, len(self.groups))
        self.groups[len(self.groups)] = (m, r)

    def new_curriculum(self, pre, now, tp):
        assert tp in ["max", "pre"]
        for _ in range(pre, now):
            if tp == "max":
                lens = {key: value[1] - value[0] for key, value in self.groups.items()}
                key = max(lens, key=lens.get)
                self.split_curriculum(key)
            elif tp == "pre":
                ind = 0
                pos = self.num_curriculums + 1
                for key, value in self.groups.items():
                    if value[1] - value[0] > 1 and value[0] < pos:
                        ind = key
                        pos = value[0]
                self.split_curriculum(ind)

    def on_epoch_start(self, epoch):
        super(GermCloseWeightsManager, self).on_epoch_start(epoch)
        pre = self.curriculum_level
        self.curriculum_level = np.where(np.array(self.cl_schedule) <= epoch)[0][-1] + 1
        self.new_curriculum(pre, self.curriculum_level, self.cl_criteria)
        self.to(self.device)


class GermCloseResNet(GermCloseNet):
    NAME = "nds_close_resnet"

    def __init__(
        self,
        search_space,
        num_classes=10,
        stem_type="res_stem_cifar",
        force_ordinal=False,
        depths=[[1, 2, 3, 4, 6, 8, 12, 16, 24]] * 3,
        widths=[[16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]] * 3,
        num_curriculums=1,
    ):
        super(GermCloseResNet, self).__init__(
            search_space,
            block_type="res_basic_block",
            depths=depths,
            widths=widths,
            strides=[1, 2, 2],
            bot_muls=None,
            num_groups=None,
            stem_type=stem_type,
            stem_w=16,
            num_classes=num_classes,
            use_se=False,
            se_r=0.25,
            head_w=0,
            bn_affine=False,
            force_use_ordinal_channel_handler=force_ordinal,
            num_curriculums=num_curriculums,
        )
        depths=[[1, 2, 3, 4, 6, 8, 12, 16, 24]] * 3
        widths=[[16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]] * 3
        
        self.num_parameters = len(depths) + len(widths)
        self.num_stages = len(depths)
        self.num_positions = self.stages[0].max_depth - 1


class GermCloseResNextA(GermCloseNet):
    NAME = "nds_close_resnexta"

    def __init__(
        self,
        search_space,
        num_classes=10,
        stem_type="res_stem_cifar",
        group_search=True,
        force_ordinal=False,
        num_curriculums=1,
    ):
        super(GermCloseResNextA, self).__init__(
            search_space,
            block_type="res_bottleneck_block",
            depths=[[1, 2, 4, 8, 16]] * 3,
            widths=[[16, 32, 64, 128, 256]] * 3,
            strides=[1, 2, 2],
            bot_muls=[[0.25, 0.5, 1]] * 3,
            num_groups=[[1, 2, 4]] * 3 if group_search else [1, 1, 1],
            stem_type=stem_type,
            stem_w=16,
            num_classes=num_classes,
            use_se=False,
            se_r=0.25,
            head_w=0,
            bn_affine=False,
            force_use_ordinal_channel_handler=force_ordinal,
            num_curriculums=num_curriculums,
        )
        depths = [[1, 2, 4, 8, 16]] * 3
        widths = [[16, 32, 64, 128, 256]] * 3
        bot_muls = [[0.25, 0.5, 1]] * 3
        num_groups = [[1, 2, 4]] * 3 if group_search else [1, 1, 1]

        self.num_parameters = len(depths) + len(widths) + len(bot_muls) + len(num_groups)
        self.num_stages = len(depths)
        self.num_positions = self.stages[0].max_depth - 1
