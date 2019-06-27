import os
import math
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from aw_nas import AwnasPlugin
from aw_nas.objective import BaseObjective
from aw_nas.weights_manager.diff_super_net import DiffSuperNet
from aw_nas.utils.torch_utils import accuracy
from aw_nas.ops import register_primitive

def weights_init(m, deepth=0, max_depth=2):
    if deepth > max_depth:
        return
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        return
    elif isinstance(m, torch.nn.ReLU):
        return
    elif isinstance(m, torch.nn.Module):
        deepth += 1
        for m_ in m.modules():
            weights_init(m_, deepth)
    else:
        raise ValueError("%s is unk" % m.__class__.__name__)

class WeightInitDiffSuperNet(DiffSuperNet):
    NAME = "fb_diff_supernet"
    def __init__(self, *args, **kwargs):
        super(WeightInitDiffSuperNet, self).__init__(*args, **kwargs)
        self.apply(weights_init)

class FBNetBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, expansion, bn=True):
        super(FBNetBlock, self).__init__()
        # assert not bn, "not support bn for now"
        bias_flag = not bn
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        else:
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        inner_dim = int(C_in * expansion)
        self.opa = nn.Sequential(
            nn.Conv2d(C_in, inner_dim, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(inner_dim, inner_dim, kernel_size, stride=stride,
                      padding=padding, bias=bias_flag),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(inner_dim, C_out, 1, stride=1, padding=0, bias=bias_flag),
            nn.BatchNorm2d(C_out),
        )
        self.opb = nn.Sequential(
          nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, bias=bias_flag),
          nn.BatchNorm2d(C_in),
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias_flag),
          nn.BatchNorm2d(C_out),
          )
        self.relus = nn.ReLU(inplace=False)
  
    def forward(self, x):
        a = self.opa(x)
        b = self.opb(x)
        return self.relus(a + b)

def block_0(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 3, stride, 1)

def block_1(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 5, stride, 1)

def block_2(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 7, stride, 1)

def block_3(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 3, stride, 1./2)

def block_4(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 5, stride, 1./2)

def block_5(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 7, stride, 1./2)

def block_6(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 3, stride, 1./4)

def block_7(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 5, stride, 1./4)

def block_8(C_in, C_out, stride, affine):
    return FBNetBlock(C_in, C_out, 7, stride, 1./4)

register_primitive("block_0", block_0)
register_primitive("block_1", block_1)
register_primitive("block_2", block_2)
register_primitive("block_3", block_3)
register_primitive("block_4", block_4)
register_primitive("block_5", block_5)
register_primitive("block_6", block_6)
register_primitive("block_7", block_7)
register_primitive("block_8", block_8)

class CosineDecayLR(_LRScheduler):
    def __init__(self, optimizer, T_max, alpha=1e-4,
                 t_mul=2, lr_mul=0.9,
                 last_epoch=-1,
                 warmup_step=300,
                 logger=None):
        self.T_max = T_max
        self.alpha = alpha
        self.t_mul = t_mul
        self.lr_mul = lr_mul
        self.warmup_step = warmup_step
        self.logger = logger
        self.last_restart_step = 0
        self.flag = True
        super(CosineDecayLR, self).__init__(optimizer, last_epoch)

        self.min_lrs = [b_lr * alpha for b_lr in self.base_lrs]
        self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                         for (b, m) in zip(self.base_lrs, self.min_lrs)]

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart_step
        assert T_cur >= 0
        if T_cur <= self.warmup_step and (not self.flag):
            base_lrs = [min_lr + rise_lr * T_cur
                        for (base_lr, min_lr, rise_lr) in 
                        zip(self.base_lrs, self.min_lrs, self.rise_lrs)]
            if T_cur == self.warmup_step:
                self.last_restart_step = self.last_epoch
                self.flag = True
        else:
            base_lrs = [self.alpha + (base_lr - self.alpha) *
                        (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                        for base_lr in self.base_lrs]
        if T_cur == self.T_max:
            self.last_restart_step = self.last_epoch
            self.min_lrs = [b_lr * self.alpha for b_lr in self.base_lrs]
            self.base_lrs = [b_lr * self.lr_mul for b_lr in self.base_lrs]
            self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                             for (b, m) in zip(self.base_lrs, self.min_lrs)]
            self.T_max = int(self.T_max * self.t_mul)
            self.flag = False
        return base_lrs

torch.optim.lr_scheduler.CosineDecayLR = CosineDecayLR

class LatencyObjective(BaseObjective):
    NAME = "latency"

    def __init__(self, search_space, alpha=0.2, beta=0.6, lamb=None, latency_file="speed.txt"):
        super(LatencyObjective, self).__init__(search_space)
        assert os.path.exists(latency_file)
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb # TODO: find this coeff when using discrete rollout
        with open(latency_file, "r") as f:
            lat_lines = f.readlines()
            self.latency_lut = []
            for lat_line in lat_lines:
                lat_line = lat_line.rstrip()
                self.latency_lut.append([float(x) for x in lat_line.split()])
        self._min_lat = sum([min(lat) for lat in self.latency_lut])
        self._max_lat = sum([max(lat) for lat in self.latency_lut])
        self.logger.info("Min possible latency: %.3f; Max possible latency: %.3f",
                         self._min_lat, self._max_lat)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return ["acc", "mean_latency"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        acc = float(accuracy(outputs, targets)[0]) / 100
        total_latency = 0.
        ss = self.search_space
        if cand_net.super_net.rollout_type == "discrete":
            for i_layer, geno in enumerate(cand_net.genotypes):
                prim = geno[0][0]
                prims = ss.cell_shared_primitives[ss.cell_layout[i_layer]]
                total_latency += float(self.latency_lut[i_layer][prims.index(prim)])
        else:
            for i_layer, arch in enumerate(cand_net.arch):
                latency = (arch[0] * \
                           torch.Tensor(self.latency_lut[i_layer]).to(arch.device)).sum().item()
                if arch[0].ndimension() == 2:
                    latency /= arch[0].shape[0]
                total_latency += latency
        return [acc, total_latency]

    def get_reward(self, inputs, outputs, targets, cand_net):
        acc = float(accuracy(outputs, targets)[0]) / 100
        if self.lamb is not None:
            latency_penalty = 0.
            ss = self.search_space
            for i_layer, geno in enumerate(cand_net.genotypes):
                prim = geno[0][0]
                prims = ss.cell_shared_primitives[ss.cell_layout[i_layer]]
                latency_penalty += float(self.latency_lut[i_layer][prims.index(prim)])
            # return acc  + float(self.lamb) / (latency_penalty - self._min_lat + 1.)
            return acc  + float(self.lamb) * (1. / latency_penalty - 1. / self._max_lat)
        return acc

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if add_controller_regularization:
            # differentiable rollout
            latency_loss = 0.
            for i_layer, arch in enumerate(cand_net.arch):
                latency = (arch[0] * \
                           torch.Tensor(self.latency_lut[i_layer]).to(arch.device)).sum()
                if arch[0].ndimension() == 2:
                    latency = latency / arch[0].shape[0]
                latency_loss += latency
            loss = loss + self.alpha * (latency_loss).pow(self.beta)
        return loss

class FBNetPlugin(AwnasPlugin):
    NAME = "fbnet"
    objective_list = [LatencyObjective]
    weights_manager_list = [WeightInitDiffSuperNet]
