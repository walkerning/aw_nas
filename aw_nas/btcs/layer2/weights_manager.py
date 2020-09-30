import torch
from torch import nn

from aw_nas import ops
from aw_nas.btcs.layer2.search_space import *
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet


class Layer2CandidateNet(CandidateNet):
    def __init__(self, supernet, rollout, eval_no_grad):
        super().__init__(eval_no_grad)

        self.supernet = supernet  # type: Layer2MacroSupernet
        self.rollout = rollout  # type: Layer2Rollout

    def begin_virtual(self):
        raise NotImplementedError()

    def forward(self, inputs):
        return self.supernet.forward(inputs, self.rollout)

    def _forward_with_params(self, *args, **kwargs):
        raise NotImplementedError()

    def get_device(self):
        return self.supernet.device


class Layer2MacroSupernet(BaseWeightsManager, nn.Module):
    NAME = "layer2_supernet"

    def __init__(
        self,
        search_space,  # type: Layer2SearchSpace
        device,
        rollout_type="layer2",
        init_channels=16,
        # classifier
        num_classes=10,
        dropout_rate=0.0,
        max_grad_norm=None,
        # stem
        use_stem="conv_bn_3x3",
        stem_stride=1,
        stem_affine=True,
        stem_multiplier=1,
        # candidate
        candidate_eval_no_grad=True,
        # schedule
        schedule_cfg=None,
    ):
        super().__init__(search_space, device, rollout_type, schedule_cfg)
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

        for i, cg in enumerate(self.cell_layout):
            stride = 2 if cg in self.reduce_cell_groups else 1
            num_channels *= stride

            self.cells.append(
                Layer2MicroCell(
                    prev_num_channels,
                    num_channels,
                    stride,
                    affine=True,
                    primitives=self.micro_search_space.primitives,
                    num_steps=self.micro_search_space.num_steps,
                    num_init_nodes=self.micro_search_space.num_init_nodes,
                    output_op=self.micro_search_space.concat_op,
                    postprocess_op="conv_1x1",
                    cell_shortcut=True,
                    cell_shortcut_op="skip_connect",
                )
            )

            prev_num_channels = num_channels

        # make pooling and classifier
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        self.classifier = nn.Linear(prev_num_channels, num_classes)

        self.to(self.device)

    def forward(
        self,
        inputs,
        rollout,  # type: Layer2Rollout
    ):
        macro_rollout = rollout.macro  # type: StagewiseMacroRollout
        micro_rollout = rollout.micro  # type: DenseMicroRollout

        overall_adj = self.macro_search_space.parse_overall_adj(macro_rollout)

        # all cell outputs + input/output states
        states = [None] * (len(self.cells) + 2)  # type: list[torch.Tensor]
        if self.use_stem:
            states[0] = self.stem(inputs)
        else:
            states[0] = inputs

        assert len(states) == len(overall_adj)

        for to, froms in enumerate(overall_adj):
            froms = np.nonzero(froms)[0]
            if len(froms) == 0:
                continue  # no inputs to this cell
            if any(states[i] is None for i in froms):
                raise RuntimeError(
                    "Invalid compute graph. Cell output used before computed"
                )

            # all inputs to a cell are added
            cell_idx = to - 1
            cell_input = sum(states[i] for i in froms)

            if cell_idx < len(self.cells):
                cell_arch = micro_rollout.arch[self.cell_layout[cell_idx]]
                states[to] = self.cells[cell_idx].forward(cell_input, cell_arch)
            else:
                states[to] = cell_input  # the final output state

        assert states[-1] is not None

        out = self.pooling(states[-1]).squeeze()
        out = self.dropout(out)
        out = self.classifier(out)

        return out

    def assemble_candidate(self, rollout):
        return Layer2CandidateNet(self, rollout, self.candidate_eval_no_grad)

    def set_device(self, device):
        self.device = device
        self.to(device)

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

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["layer2"]


class Layer2MicroCell(nn.Module):
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
        postprocess_op="conv_1x1",
        cell_shortcut=False,
        cell_shortcut_op="skip_connect",
    ):
        super().__init__()

        self.out_channels = out_channels
        self.stride = stride

        self.primitives = primitives
        self.num_init_nodes = num_init_nodes
        self.num_nodes = num_steps + num_init_nodes
        self.output_op = output_op

        # it's easier to calc edge indices with a longer ModuleList and some None
        self.edges = nn.ModuleList()
        for j in range(self.num_nodes):
            for i in range(self.num_nodes):
                if j > i:
                    if i < self.num_init_nodes:
                        self.edges.append(
                            Layer2MicroEdge(
                                primitives, in_channels, out_channels, stride, affine
                            )
                        )
                    else:
                        self.edges.append(
                            Layer2MicroEdge(
                                primitives, out_channels, out_channels, 1, affine
                            )
                        )
                else:
                    self.edges.append(None)

        if cell_shortcut and cell_shortcut_op != "none":
            self.shortcut = ops.get_op(cell_shortcut_op)(
                in_channels, out_channels, stride, affine
            )
        else:
            self.shortcut = None

        if self.output_op == "concat":
            self.postprocess = ops.get_op(postprocess_op)(
                out_channels * num_steps, out_channels, stride=1, affine=False
            )

    def forward(self, inputs, cell_arch):
        # cell_arch shape: [#nodes, #nodes, #ops]
        n, _, h, w = inputs.shape
        node_outputs = [inputs] * self.num_init_nodes

        for to in range(self.num_init_nodes, self.num_nodes):
            froms = np.nonzero(cell_arch[to].sum(axis=1))[0]

            edge_indices = froms + (to * self.num_nodes)
            if any(self.edges[i] is None for i in edge_indices):
                raise RuntimeError(
                    "Invalid compute graph in cell. Cannot compute an edge where j <= i"
                )

            # outputs `to` this node `from` all used edges
            edge_outputs = [
                self.edges[edge_i](node_outputs[from_i], cell_arch[to, from_i])
                for edge_i, from_i in zip(edge_indices, froms)
            ]

            if len(edge_outputs) != 0:
                node_outputs.append(sum(edge_outputs))
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
            out = self.postprocess(torch.cat(node_outputs, dim=1))
        elif self.output_op == "add":
            out = sum(node_outputs)
        else:
            raise ValueError("Unknown cell output op `{}`".format(self.output_op))

        if self.shortcut is not None:
            out += self.shortcut(inputs)

        return out


class Layer2MicroEdge(nn.Module):
    def __init__(self, primitives, in_channels, out_channels, stride, affine):
        super().__init__()

        assert "none" not in primitives, "Edge should not have `none` primitive"

        self.ops = nn.ModuleList(
            ops.get_op(prim)(in_channels, out_channels, stride, affine)
            for prim in primitives
        )

    def forward(self, inputs, edge_arch):
        outputs = []
        for op, use_op in zip(self.ops, edge_arch):
            if use_op != 0:
                outputs.append(op(inputs))

        if len(outputs) != 0:
            return sum(outputs)
        else:
            raise RuntimeError(
                "Edge module does not handle the case where no op is "
                "used. It should be handled in Cell and Edge.forward "
                "should not be called"
            )
