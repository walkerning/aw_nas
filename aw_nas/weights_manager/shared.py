import numpy as np
import torch
from torch import nn

from aw_nas import ops
from aw_nas.weights_manager.base import BaseWeightsManager

class SharedNet(BaseWeightsManager, nn.Module):
    def __init__(self, search_space, device,
                 cell_cls, op_cls,
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 cell_group_kwargs=None):
        super(SharedNet, self).__init__(search_space, device)
        nn.Module.__init__(self)

        self.num_classes = num_classes
        # init channel number of the first cell layers,
        # x2 after every reduce cell
        self.init_channels = init_channels
        # channels of stem conv / init_channels
        self.stem_multiplier = stem_multiplier

        # training
        self.max_grad_norm = max_grad_norm
        self.dropout_rate = dropout_rate

        # search space configs
        self._num_init = self.search_space.num_init_nodes
        self._cell_layout = self.search_space.cell_layout
        self._reduce_cgs = self.search_space.reduce_cell_groups
        self._num_layers = self.search_space.num_layers

        ## initialize sub modules
        c_stem = self.stem_multiplier * self.init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_stem)
        )

        self.cells = nn.ModuleList()
        num_channels = self.init_channels
        prev_num_channels = [c_stem] * self._num_init
        strides = [2 if self._is_reduce(i_layer) else 1 for i_layer in range(self._num_layers)]

        for i_layer, stride in enumerate(strides):
            if stride > 1:
                num_channels *= stride
            if cell_group_kwargs is not None:
                # support passing in different kwargs when instantializing
                # cell class for different cell groups
                kwargs = cell_group_kwargs[self._cell_layout[i_layer]]
            else:
                kwargs = {}
            cell = cell_cls(op_cls,
                            self.search_space,
                            layer_index=i_layer,
                            num_channels=num_channels,
                            prev_num_channels=tuple(prev_num_channels),
                            stride=stride,
                            prev_strides=[1] * self._num_init + strides[:i_layer],
                            **kwargs)
            prev_num_channel = cell.num_out_channel()
            prev_num_channels.append(prev_num_channel)
            prev_num_channels = prev_num_channels[1:]
            self.cells.append(cell)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self.dropout_rate and self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = ops.Identity()
        self.classifier = nn.Linear(prev_num_channel,
                                    self.num_classes)

        self.to(self.device)

    def _is_reduce(self, layer_idx):
        return self._cell_layout[layer_idx] in self._reduce_cgs

    def forward(self, inputs, genotypes, **kwargs): #pylint: disable=arguments-differ
        stemed = self.stem(inputs)
        states = [stemed] * self._num_init

        for cg_idx, cell in zip(self._cell_layout, self.cells):
            genotype = genotypes[cg_idx]
            states.append(cell(states, genotype, **kwargs))
            states = states[1:]

        out = self.global_pooling(states[-1])
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def step_current_gradients(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # clip the gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def save(self, path):
        torch.save({"state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def supported_data_types(cls):
        return ["image"]


class SharedCell(nn.Module):
    def __init__(self, op_cls, search_space, layer_index, num_channels,
                 prev_num_channels, stride, prev_strides):
        super(SharedCell, self).__init__()
        self.search_space = search_space
        self.stride = stride
        self.is_reduce = stride != 1
        self.num_channels = num_channels
        self.layer_index = layer_index

        self._steps = self.search_space.num_steps
        self._num_init = self.search_space.num_init_nodes
        self._primitives = self.search_space.shared_primitives

        self.preprocess_ops = nn.ModuleList()
        prev_strides = list(np.cumprod(list(reversed(prev_strides))))
        prev_strides.insert(0, 1)
        prev_strides = reversed(prev_strides[:len(prev_num_channels)])
        for prev_c, prev_s in zip(prev_num_channels, prev_strides):
            if prev_s > 1:
                # need skip connection, and is not the connection from the input image
                preprocess = ops.FactorizedReduce(C_in=prev_c,
                                                  C_out=num_channels,
                                                  stride=prev_s,
                                                  affine=False)
            else: # prev_c == _steps * num_channels or inputs
                preprocess = ops.ReLUConvBN(C_in=prev_c,
                                            C_out=num_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            affine=False)
            self.preprocess_ops.append(preprocess)
        assert len(self.preprocess_ops) == self._num_init

        self.edges = nn.ModuleList()
        for i in range(self._steps):
            for j in range(i + self._num_init):
                op = op_cls(self.num_channels, stride=self.stride if j < self._num_init else 1,
                            primitives=self._primitives)
                self.edges.append(op)


class SharedOp(nn.Module):
    """
    The operation on an edge, consisting of multiple primitives.
    """

    def __init__(self, C, stride, primitives):
        super(SharedOp, self).__init__()
        self.primitives = primitives
        self.stride = stride
        self.p_ops = nn.ModuleList()
        for primitive in self.primitives:
            op = ops.get_op(primitive)(C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C,
                                                      affine=False))
            self.p_ops.append(op)
