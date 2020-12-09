import torch
from torch import nn
import torch.nn.functional as F

from aw_nas import ops, utils
from aw_nas.utils import DistributedDataParallel
from aw_nas.btcs.layer2.search_space import *
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet

try:
    from torch.nn import SyncBatchNorm
    convert_sync_bn = SyncBatchNorm.convert_sync_batchnorm
except:
    def convert_sync_bn(m):
        return m


class Layer2DiffCandidateNet(CandidateNet):
    def __init__(
        self, supernet, rollout, eval_no_grad, gpus=tuple(), multiprocess=False
    ):
        super().__init__(eval_no_grad)

        self.supernet = supernet  # type: Layer2MacroSupernet
        self.rollout = rollout  # type: Layer2Rollout

        # for flops_calculation
        self.total_flops = 0
        self._flops_calculated = False

        self.gpus = gpus

        self.multiprocess = multiprocess

    def begin_virtual(self):
        raise NotImplementedError()

    def forward(self, inputs):
        # re-calculate the flops at each forward
        if not self._flops_calculated:
            if self.multiprocess:
                output = self.supernet.parallel_model.forward(
                    inputs, self.rollout, calc_flops=True
                )
            else:
                output = self.supernet.forward(inputs, self.rollout, calc_flops=True)
            self.total_flops = self.supernet.total_flops
            self._flops_calculated = True
            self.supernet.total_flops = 0
            for cell in self.supernet.cells:
                cell.total_flops = 0
        else:
            if self.multiprocess:
                output = self.supernet.parallel_model.forward(inputs, self.rollout)
            else:
                output = self.supernet.forward(inputs, self.rollout)
        return output

    def _forward_with_params(self, *args, **kwargs):
        raise NotImplementedError()

    def get_device(self):
        return self.supernet.device

    def named_parameters(self, *args, **kwargs):  # pylint: disable=arguments-differ
        return self.supernet.named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):  # pylint: disable=arguments-differ
        return self.supernet.named_buffers(*args, **kwargs)

    def eval_data(
        self, data, criterions, mode="eval", **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Override eval_data, to enable gradient.

        Returns:
           results (list of results return by criterions)
        """
        self._set_mode(mode)

        outputs = self.forward_data(data[0])
        # kwargs is detach_arch: False, since here the forward has no arg detach-arch, so not using the kwargs

        return utils.flatten_list([c(data[0], outputs, data[1]) for c in criterions])


class Layer2MacroDiffSupernet(BaseWeightsManager, nn.Module):
    NAME = "layer2_diff_supernet"

    def __init__(
        self,
        search_space,  # type: Layer2SearchSpace
        device,
        rollout_type="layer2",
        init_channels=16,
        # classifier
        num_classes=10,
        dropout_rate=0.0,
        # stem
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        stem_multiplier=1,
        max_grad_norm=5.0,
        # candidate
        candidate_eval_no_grad=True,
        # micro-cell cfg
        micro_cell_cfg={},
        # schedule
        schedule_cfg=None,
        gpus=tuple(),
        multiprocess=False,
    ):
        super(Layer2MacroDiffSupernet, self).__init__(
            search_space, device, rollout_type, schedule_cfg
        )
        nn.Module.__init__(self)

        self.macro_search_space = (
            search_space.macro_search_space
        )  # type: StagewiseMacroSearchSpace
        self.micro_search_space = (
            search_space.micro_search_space
        )  # type: DenseMicroSearchSpace

        self.num_cell_groups = self.macro_search_space.num_cell_groups
        self.cell_layout = self.macro_search_space.cell_layout
        self.reduce_cell_groups = self.macro_search_space.reduce_cell_groups

        self.max_grad_norm = max_grad_norm

        self.candidate_eval_no_grad = candidate_eval_no_grad

        self.micro_cell_cfg = micro_cell_cfg
        if "postprocess" in micro_cell_cfg.keys():
            self.cell_use_postprocess = micro_cell_cfg["postprocess"]
        else:
            self.cell_use_postprocess = (
                False  # defualt use preprocess if not specified in cfg
            )

        self.gpus = gpus
        self.multiprocess = multiprocess

        # make stem
        self.use_stem = use_stem
        if not self.use_stem:
            c_stem = 3
        elif isinstance(self.use_stem, (list, tuple)):
            self.stem = []
            c_stem = stem_multiplier * init_channels
            for i, stem_type in enumerate(self.use_stem):
                c_in = 3 if i == 0 else c_stem
                self.stem.append(
                    ops.get_op(stem_type)(
                        c_in, c_stem, stride=stem_stride, affine=stem_affine
                    )
                )

            self.stem = nn.Sequential(*self.stem)
        else:
            c_stem = stem_multiplier * init_channels
            self.stem = ops.get_op(self.use_stem)(
                3, c_stem, stride=stem_stride, affine=stem_affine
            )

        # make cells
        self.cells = nn.ModuleList()
        num_channels = init_channels
        prev_num_channels = c_stem

        """when use preprocess, extra stem is applied after normal stem to make 1st cell input 4c"""
        """currently no matter what stem is used, fp-conv 1x1 is inserted to align the width, maybe modify the stem?(maybe more flops)"""
        self.extra_stem = ops.get_op("nor_conv_1x1")(
            prev_num_channels,
            num_channels * self.micro_search_space.num_steps,
            1,
            affine=True,
        )

        for i, cg in enumerate(self.cell_layout):

            use_next_stage_width = np.array(self.macro_search_space.stages_end[:-1]) - 1
            # next_stage_begin_idx = np.array(self.macro_search_space,stages_begin[1:])+1
            use_next_stage_width = (
                i + 2 if i in use_next_stage_width else None
            )  # [i] the last cell; [i+1] the reduction cell; [i+2] the 1st cell next stage

            stride = 2 if cg in self.reduce_cell_groups else 1
            num_channels *= stride

            self.cells.append(
                Layer2MicroDiffCell(
                    prev_num_channels,
                    num_channels,
                    stride,
                    affine=True,
                    primitives=self.micro_search_space.primitives,
                    num_steps=self.micro_search_space.num_steps,
                    num_init_nodes=self.micro_search_space.num_init_nodes,
                    output_op=self.micro_search_space.concat_op,
                    width_choice=self.macro_search_space.width_choice,
                    cell_idx=i,
                    use_next_stage_width=use_next_stage_width,
                    **self.micro_cell_cfg,
                )
            )

            prev_num_channels = num_channels

        # make pooling and classifier
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        if self.cell_use_postprocess:
            self.classifier = nn.Linear(prev_num_channels, num_classes)
        else:
            self.classifier = nn.Linear(
                prev_num_channels * self.micro_search_space.num_steps, num_classes
            )

        self.total_flops = 0.0

        self.to(self.device)
        self._parallelize()

    def forward(
        self,
        inputs,
        rollout,  # type: Layer2Rollout
        calc_flops=False,
    ):
        macro_rollout = rollout.macro  # type: StagewiseMacroRollout
        micro_rollout = rollout.micro  # type: DenseMicroRollout

        overall_adj = self.macro_search_space.parse_overall_adj(macro_rollout.genotype)

        # all cell outputs + input/output states
        states = [None] * (len(self.cells) + 2)  # type: list[torch.Tensor]
        if self.use_stem:
            states[0] = self.stem(inputs)
            if calc_flops:  # donot calc stem inputs
                pass
                # self.get_op_flops(self.stem, inputs, states[0])
        else:
            states[0] = inputs

        if not self.cell_use_postprocess:
            states[0] = self.extra_stem(states[0])

        assert len(states) == len(overall_adj)

        for i_stage in range(len(macro_rollout.arch)):
            for to, froms_with_weight in enumerate(macro_rollout.arch[i_stage]):
                if to == 0:
                    continue
                froms = np.nonzero(
                    froms_with_weight
                ).detach()  # no detach will cause autograd failure, this bug occurs on some pytorch versions. e.g. 1.2.0
                if len(froms) == 0:
                    continue  # no inputs to this cell
                if any(states[i] is None for i in froms):
                    raise RuntimeError(
                        "Invalid compute graph. Cell output used before computed"
                    )

                # all inputs to a cell are added
                prev_idx = (
                    sum(self.search_space.macro_search_space.stage_node_nums[:i_stage])
                    - i_stage
                )
                to = prev_idx + to
                cell_idx = to - 1
                cell_input = sum(
                    states[prev_idx + i] * from_weight
                    for i, from_weight in enumerate(
                        froms_with_weight[: (to - prev_idx)]
                    )
                )

                if cell_idx < len(self.cells):
                    cg_id = self.cell_layout[cell_idx]
                    cell_arch = micro_rollout.arch[cg_id]
                    width_arch = macro_rollout.width_arch
                    width_logits = macro_rollout.width_arch

                    if calc_flops:
                        cell_logits = micro_rollout.logits_arch[cg_id]
                        states[to] = self.cells[cell_idx].forward(
                            cell_input,
                            cell_arch,
                            width_arch,
                            cell_logits,
                            width_logits,
                            calc_flops=True,
                        )
                        alphas_softmax = F.softmax(
                            macro_rollout.logits[0].split(
                                [
                                    i - 1
                                    for i in macro_rollout.search_space.stage_node_nums
                                ]
                            )[i_stage],
                            dim=0,
                        )
                        self.total_flops += (
                            self.cells[cell_idx].get_flops()
                            * (alphas_softmax[cell_idx - prev_idx + 1 :]).sum()
                        )
                    else:
                        states[to] = self.cells[cell_idx].forward(
                            cell_input,
                            cell_arch,
                            width_arch,
                            cell_logits=None,
                            width_logits=None,
                            calc_flops=False,
                        )
                else:
                    states[to] = cell_input  # the final output state

        assert states[-1] is not None

        out = self.pooling(states[-1]).squeeze()
        out = self.dropout(out)
        out_ = self.classifier(out)
        if calc_flops:  # donot calc final fc flops
            pass
            # self.get_op_flops(self.classifier, out, out_)

        return out_

    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self).to(self.device)
            object.__setattr__(
                self,
                "parallel_model",
                DistributedDataParallel(net, self.gpus, find_unused_parameters=True),
            )
        else:
            object.__setattr__(self, "parallel_model", self)

    def assemble_candidate(self, rollout):
        return Layer2DiffCandidateNet(
            self,
            rollout,
            self.candidate_eval_no_grad,
            gpus=self.gpus,
            multiprocess=self.multiprocess,
        )

    def set_device(self, device):
        self.device = device
        self.to(device)

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def save(self, path):
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    # calc the flops
    def get_op_flops(self, module, inputs, outputs, alpha=1.0):

        # preprocess for nested op
        if isinstance(module, ops.BinaryConv2d):
            module = module.conv
        elif isinstance(module, ops.ConvBNReLU):
            module = module.op[0]
        else:
            pass

        # count the op flops
        if isinstance(module, nn.Conv2d):
            cur_flops = (
                2
                * inputs.size(1)
                * outputs.size(1)
                * module.kernel_size[0]
                * module.kernel_size[1]
                * outputs.size(2)
                * outputs.size(3)
                / module.groups
            )
        elif isinstance(module, ops.XNORConv2d):
            # 1-bit conv
            cur_flops = (
                2
                * inputs.size(1)
                * outputs.size(1)
                * module.kernel_size
                * module.kernel_size
                * outputs.size(2)
                * outputs.size(3)
                / (module.groups)
            )
        elif isinstance(module, nn.Linear):
            cur_flops = 2 * inputs.size(1) * outputs.size(1)
        else:
            cur_flops = 0

        self.total_flops += alpha * cur_flops

    def get_flops(self):
        return self.total_flops

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["layer2", "layer2-differentiable"]


class Layer2MicroDiffCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        affine,
        primitives,
        num_steps,
        num_init_nodes,
        output_op="concat",
        width_choice=[1.0],
        postprocess=False,
        process_op="conv_1x1",
        cell_shortcut=False,
        cell_shortcut_op="skip_connect",
        partial_channel_proportion=None,
        cell_idx=None,
        use_next_stage_width=None,
    ):
        super(Layer2MicroDiffCell, self).__init__()

        self.out_channels = out_channels
        self.stride = stride

        self.primitives = primitives
        self.num_init_nodes = num_init_nodes
        self.num_steps = num_steps
        self.num_nodes = num_steps + num_init_nodes
        self.output_op = output_op

        self.cell_idx = cell_idx
        self.use_next_stage_width = use_next_stage_width

        self.postprocess = postprocess

        self.width_choice = width_choice
        assert 1.0 in self.width_choice, "Must have a width choice with 100% channels"

        self.register_buffer(
            "channel_masks", torch.zeros(len(self.width_choice), self.out_channels)
        )
        for i, w in enumerate(self.width_choice):
            self.channel_masks[i][: int(w * self.out_channels)] = 1.0

        self.partial_channel_proportion = partial_channel_proportion
        assert (
            partial_channel_proportion is None
        )  # currently dont support partial channel

        # it's easier to calc edge indices with a longer ModuleList and some None
        self.edges = nn.ModuleList()
        for j in range(self.num_nodes):
            for i in range(self.num_nodes):
                if j > i:
                    if i < self.num_init_nodes:
                        self.edges.append(
                            Layer2MicroDiffEdge(
                                primitives, in_channels, out_channels, stride, affine
                            )
                        )
                    else:
                        self.edges.append(
                            Layer2MicroDiffEdge(
                                primitives, out_channels, out_channels, 1, affine
                            )
                        )
                else:
                    self.edges.append(None)

        if cell_shortcut and cell_shortcut_op != "none":
            if not self.postprocess:
                self.shortcut = ops.get_op(cell_shortcut_op)(
                    in_channels * self.num_steps,
                    out_channels * self.num_steps,
                    stride,
                    affine,
                )
            else:
                self.shortcut = ops.get_op(cell_shortcut_op)(
                    in_channels,
                    out_channels,
                    stride,
                    affine,
                )
        else:
            self.shortcut = None

        if self.output_op == "concat":
            """
            no matter post/preprocess [4c,c]
            however with reduction cell, out_channel = 2*in_c,
            so when using preprocess, should use in_channels
            """
            if not self.postprocess:
                self.process = ops.get_op(process_op)(
                    in_channels * num_steps, in_channels, stride=1, affine=False
                )
            else:
                self.process = ops.get_op(process_op)(
                    out_channels * num_steps, out_channels, stride=1, affine=False
                )

        self.total_flops = 0.0

    def forward(
        self,
        inputs,
        cell_arch,
        width_arch=None,
        cell_logits=None,
        width_logits=None,
        calc_flops=False,
    ):
        # cell_arch shape: [#nodes, #nodes, #ops]
        # width_arch shape: [#cell_group, #width_choice], optional
        # cell_logits: same as cell_arch, but from before Softmax/Gumbel

        if self.use_next_stage_width is not None:
            width_arch_n = (
                width_arch[self.use_next_stage_width]
                if width_arch is not None
                else None
            )
            width_logits_n = (
                width_logits[self.use_next_stage_width]
                if width_logits is not None
                else None
            )
            channel_mask_n = (self.channel_masks * width_arch_n.view(-1, 1)).sum(dim=0)
            channel_mask_n = channel_mask_n.view(1, -1, 1, 1)

        width_arch = width_arch[self.cell_idx] if width_arch is not None else None
        width_logits = width_logits[self.cell_idx] if width_logits is not None else None

        # for the 1st cell, since extra_stem is applied, the input is 4C ch
        if not self.postprocess:
            inputs_processed = self.process(inputs)
        else:
            inputs_processed = inputs

        n, _, h, w = inputs_processed.shape
        node_outputs = [inputs_processed] * self.num_init_nodes

        if width_arch is not None:
            channel_mask = (self.channel_masks * width_arch.view(-1, 1)).sum(dim=0)
            channel_mask = channel_mask.view(1, -1, 1, 1)

        for to in range(self.num_init_nodes, self.num_nodes):

            froms = np.arange(0, to)  # ugly but compatible fix
            edge_indices = froms + (to * self.num_nodes)
            if any(self.edges[i] is None for i in edge_indices):
                raise RuntimeError(
                    "Invalid compute graph in cell. Cannot compute an edge where j <= i"
                )

            # outputs `to` this node `from` all used edges

            if calc_flops:
                edge_outputs = []
                for edge_i, from_i in zip(edge_indices, froms):
                    edge_output = self.edges[edge_i](
                        node_outputs[from_i], cell_arch[to, from_i]
                    )
                    # for op_weight_idx, op_weight in enumerate(F.softmax(cell_arch[to, from_i],dim=0)):
                    for op_weight_idx, op_weight in enumerate(
                        F.softmax(cell_logits[to, from_i], dim=0)
                    ):
                        # with respeact to different op, is hard to feed in
                        # need more logic in the get_op_flops func
                        if width_logits is not None:
                            for width, width_weight in zip(
                                self.width_choice, F.softmax(width_logits, dim=0)
                            ):
                                self.get_op_flops(
                                    module=self.edges[edge_i].p_ops[op_weight_idx],
                                    inputs=node_outputs[from_i],
                                    outputs=edge_output,
                                    width=width,
                                    alpha=op_weight * width_weight,
                                )
                        else:
                            self.get_op_flops(
                                module=self.edges[edge_i].p_ops[op_weight_idx],
                                inputs=node_outputs[from_i],
                                outputs=edge_output,
                                width=1.0,
                                alpha=op_weight,
                            )
                    edge_outputs.append(edge_output)
            else:
                edge_outputs = [
                    self.edges[edge_i](node_outputs[from_i], cell_arch[to, from_i])
                    for edge_i, from_i in zip(edge_indices, froms)
                ]
            if len(edge_outputs) != 0:
                node_outputs.append(sum(edge_outputs))
                if width_arch is not None:
                    if self.use_next_stage_width:
                        node_outputs[-1] *= channel_mask_n
                    else:
                        node_outputs[-1] *= channel_mask
            elif self.output_op == "concat":
                # append fake outputs if required by concat
                node_outputs.append(
                    torch.zeros(
                        n,
                        self.out_channels,
                        h // self.stride,
                        w // self.stride,
                        device=inputs.device,
                    )
                )

        node_outputs = node_outputs[self.num_init_nodes :]
        if len(node_outputs) == 0:
            # no node outputs (including fake outputs) in this cell
            out = 0
        elif self.output_op == "concat":
            if self.postprocess:
                out = self.process(torch.cat(node_outputs, dim=1))
            else:
                out = torch.cat(node_outputs, dim=1)
        elif self.output_op == "add":
            out = sum(node_outputs)
        else:
            raise ValueError("Unknown cell output op `{}`".format(self.output_op))

        # whether to use cell-wise shortcut for use-next-stage-width cell
        if self.shortcut is not None:
            NO_SHORTCUT = False
            if NO_SHORTCUT:
                if self.use_next_stage_width:
                    pass
                else:
                    out += self.shortcut(inputs)
            else:
                out += self.shortcut(inputs)

        # Applying final mask to cell output
        """
        if use-next-stage width apply the channel_mask_n instead of channel_mask
        when using the preprocess, since the mask is of [c] channels, the out is of shape [4c]
        should apply to each node outputs
        """
        # when using preprocess should duplicate the channel_mask to appky on each of the node's output

        if self.use_next_stage_width:
            if not self.postprocess:
                channel_mask_n = channel_mask_n.repeat(1, self.num_steps, 1, 1)
            out *= channel_mask_n
        else:
            if width_arch is not None:
                if not self.postprocess:
                    channel_mask = channel_mask.repeat(1, self.num_steps, 1, 1)
                out *= channel_mask

        return out

    def get_op_flops(self, module, inputs, outputs, width=1.0, alpha=1.0):

        # preprocess for nested op
        if isinstance(module, ops.BinaryConvBNReLU):
            if module.stride == 1:
                module = module.conv
            else:
                if module.reduction_op_type == "factorized":
                    assert len(module.convs) == 2
                    module = module.convs[0]
                    alpha = alpha * 2  # since there are 2 convs
                else:
                    module = module.conv
        elif isinstance(module, ops.ConvBNReLU):
            module = module.op[0]
        else:
            pass

        # count the op flops
        if isinstance(module, nn.Conv2d):
            cur_flops = (
                2
                * inputs.size(1)
                * outputs.size(1)
                * module.kernel_size[0]
                * module.kernel_size[1]
                * outputs.size(2)
                * outputs.size(3)
                / module.groups
            )
            cur_flops *= width * width
        elif isinstance(module, ops.BinaryConv2d):
            # 1-bit conv
            # since inputs is not tuple here, so no need for inputs[0]
            cur_flops = (
                2
                * inputs.size(1)
                * outputs.size(1)
                * module.kernel_size
                * module.kernel_size
                * outputs.size(2)
                * outputs.size(3)
                / module.groups
            )
            cur_flops *= width * width
        elif isinstance(module, nn.Linear):
            cur_flops = 2 * inputs.size(1) * outputs.size(1)
            cur_flops *= width * width
        else:
            cur_flops = 0

        self.total_flops += alpha * cur_flops

    def get_flops(self):
        return self.total_flops


class Layer2MicroDiffEdge(nn.Module):
    def __init__(
        self,
        primitives,
        in_channels,
        out_channels,
        stride,
        affine,
        partial_channel_proportion=None,
    ):
        super(Layer2MicroDiffEdge, self).__init__()
        # assert "none" not in primitives, "Edge should not have `none` primitive"

        self.primitives = primitives
        self.stride = stride
        self.partial_channel_proportion = partial_channel_proportion

        if self.partial_channel_proportion is not None:
            expect(
                in_channels % self.partial_channel_proportion == 0,
                "partial_channel_proportion must be divisible by #channels",
                ConfigException,
            )
            expect(
                out_channels % self.partial_channel_proportion == 0,
                "partial_channel_proportion must be divisible by #channels",
                ConfigException,
            )
            in_channels = in_channels // self.partial_channel_proportion
            out_channels = out_channels // self.partial_channel_proportion

        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = ops.get_op(primitive)(in_channels, out_channels, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(out_channels, affine=False))

            self.p_ops.append(op)

    def forward(self, x, weights, detach_arch=True):
        if weights.ndimension() == 2:
            # weights: (batch_size, num_op)
            if not weights.shape[0] == x.shape[0]:
                # every `x.shape[0] % weights.shape[0]` data use the same sampled arch weights
                assert x.shape[0] % weights.shape[0] == 0
                weights = weights.repeat(x.shape[0] // weights.shape[0], 1)
            return sum(
                [
                    weights[:, i].reshape(-1, 1, 1, 1) * op(x)
                    for i, op in enumerate(self.p_ops)
                ]
            )

        out_act: torch.Tensor = 0.0
        # weights: (num_op)
        if self.partial_channel_proportion is None:
            for w, op in zip(weights, self.p_ops):
                act = op(x)
                out_act += w * act
        else:
            op_channels = x.shape[1] // self.partial_channel_proportion
            x_1 = x[:, :op_channels, :, :]  # these channels goes through op
            x_2 = x[:, op_channels:, :, :]  # these channels skips op

            # apply pooling if the ops have stride=2
            if self.stride == 2:
                x_2 = F.max_pool2d(x_2, 2, 2)

            for w, op in zip(weights, self.p_ops):
                # if detach_arch and w.item() == 0:
                #     continue  # not really sure about this
                act = op(x_1)

                # if w.item() == 0:
                #     act.detach_()  # not really sure about this either
                out_act += w * act

            out_act = torch.cat((out_act, x_2), dim=1)

            # PC-DARTS implements a deterministic channel_shuffle() (not what they said in the paper)
            # ref: https://github.com/yuhuixu1993/PC-DARTS/blob/b74702f86c70e330ce0db35762cfade9df026bb7/model_search.py#L9
            out_act = self._channel_shuffle(out_act, self.partial_channel_proportion)

            # this is the random channel shuffle for now
            # channel_perm = torch.randperm(out_act.shape[1])
            # out_act = out_act[:, channel_perm, :, :]

        return out_act

    def get_flops(self):
        return self.total_flops

    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int):
        """channel shuffle for PC-DARTS"""
        n, c, h, w = x.shape

        x = x.view(n, groups, -1, h, w).transpose(1, 2).contiguous()

        x = x.view(n, c, h, w).contiguous()

        return x
