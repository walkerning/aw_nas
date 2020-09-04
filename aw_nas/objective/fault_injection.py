# -*- coding: utf-8 -*-
"""
Fault injection objective.
* Clean accuracy and fault-injected accuracy weighted for reward (for discrete controller search)
* Clean loss and fault-injected loss weighted for loss
  (for differentiable controller search or fault-injection training).
"""

import threading
from collections import defaultdict
import six

import numpy as np
import torch
from scipy.stats import binom
from torch import nn

from aw_nas import utils
from aw_nas.utils.torch_utils import accuracy
from aw_nas.objective.base import BaseObjective
from aw_nas.utils.exception import expect, ConfigException

def _get_average_meter_defaultdict():
    return defaultdict(utils.AverageMeter)

class FaultInjector(object):
    def __init__(self, gaussian_std=1., mode="fixed", tile_size=None, max_value_mode=True,
                 macwise_inputdep_rate=1.0):
        self.tile_size = tile_size
        self.random_inject = 0.001
        self.gaussian_std = gaussian_std
        self.mode = mode
        self.max_value_mode = max_value_mode
        self.m_i_rate = macwise_inputdep_rate
        self.fault_bit_list = np.array([2**x for x in range(8)] + [-2**x for x in range(8)],
                                       dtype=np.float32)

    def set_random_inject(self, value):
        self.random_inject = value

    def set_gaussian_std(self, value):
        self.gaussian_std = value

    def inject_gaussian(self, out):
        gaussian = torch.randn(out.shape, dtype=out.dtype, device=out.device) * self.gaussian_std
        out = out + gaussian
        return out

    def inject_saltandpepper(self, out):
        random_tensor = out.new(out.size()).random_(0, 2*int(1./self.random_inject))
        salt_ind = (random_tensor == 0)
        pepper_ind = (random_tensor == 1)
        max_ = torch.max(torch.abs(out)).cpu().data
        out[salt_ind] = 0
        out[pepper_ind] = max_
        return out

    def inject_bitflip(self, out):
        tile_size = self.tile_size
        if tile_size is None:
            tile_size = tuple(out.size())
            repeat = None
        else:
            repeat = np.ceil(np.array(out.size())  /  np.array(tile_size)).astype(np.int)
        random_tensor = out.new(torch.Size(tile_size)).random_(0, int(1. / self.random_inject))
        if repeat is not None:
            random_tensor = random_tensor.repeat(*repeat)
        # using bitflip, must have been quantized!
        # FIXME: nics_fix_pytorch should keep this the same as out.device, isn't it...
        scale = out.data_cfg["scale"].to(out.device)
        bitwidth = out.data_cfg["bitwidth"].to(out.device)
        step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]).to(out.device),
                                                 requires_grad=False),
                         (scale.float() - bitwidth.float()))
        fault_ind = (random_tensor < 1)
        fault_mask = np.round(np.exp(np.random.randint(0, 8, size=fault_ind.sum().cpu().data)\
                                     * np.log(2))).astype(np.int32)
        random_tensor.zero_()
        random_tensor[fault_ind] = torch.tensor(fault_mask).to(out.device).float()
        random_tensor = random_tensor.to(torch.int32)

        # FIXME: negative correct? no... if assume random bit-flip in complement representation
        # can realize flip sign-bit by -256(Q=8, complement = 1000 000)
        # but only support Q=8,16,32,64
        # sure we can implement this flip ourself by bias operation,
        # but i'don't think this would be more reliable than the bias model, actually...
        ori_type = out.dtype
        out = (out.div_(step).to(torch.int32) ^ random_tensor).to(ori_type).mul_(step)
        return out

    def inject_fixed(self, out, n_mac=1):
        if not hasattr(out, "data_cfg"):
            # skip connections that are added together should share the same
            # fixed quantization, but now, this is not supported by nics_fix_pt.
            # so, estimate the quantization config here
            scale = torch.ceil(torch.log(
                torch.max(torch.max(torch.abs(out)),
                          torch.tensor(1e-5).float().to(out.device))) / np.log(2.))
            bitwidth = torch.tensor([8]).to(int).to(out.device) # default 8
            max_ = float(torch.max(torch.abs(out)).cpu().data.numpy())
        else:
            scale = out.data_cfg["scale"].to(out.device)
            bitwidth = out.data_cfg["bitwidth"].to(out.device)
            max_ = float((2**scale.float()).cpu().data.numpy())
        step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]).to(out.device),
                                                 requires_grad=False),
                         (scale.float() - (bitwidth.float() - 1)))

        # ---- handle tile ----
        # Currently, only the fault position is tiled,
        # the correlation between the biases at tiling positions is not considered
        # However, easy to modify
        tile_size = self.tile_size
        if tile_size is None:
            tile_size = list(out.shape)
            repeat = None
        else:
            repeat = np.ceil(np.array(out.size())[1:]  /  np.array(tile_size)).astype(np.int)
            tile_size = [out.shape[0]] + list(tile_size)
        random_inject = self.random_inject
        bitwidth_data = int(bitwidth.cpu().data.numpy())

        # ---- handle n_mac ----
        # if n_mac != 1:
        n_addi_affect_bits = int(np.floor(np.log2(n_mac)))
        # if n_mac is very large, define the fraction length as F_w = (Q_w-1) - S_w
        # n_addi_affect_bits <= F_w + F_i - F_o; S_w + S_i - S_o >= 0
        # so n_addi_affect_bits <= (Q_w-1) + (Q_i-1) - (Q_o-1)
        # currently, we assume Q_w == Q_i == Q_o == bitwidth_data
        n_addi_affect_bits = min(bitwidth_data - 1, n_addi_affect_bits)
        random_inject = random_inject * \
                        (float(n_addi_affect_bits + bitwidth_data) / bitwidth_data)

        # ---- generate fault position mask ----
        random_tensor = out.new(torch.Size(tile_size)).random_(0, int(1. / random_inject))
        if repeat is not None:
            random_tensor = random_tensor.repeat(1, *repeat)\
                            [:, :out.shape[1], :out.shape[2], :out.shape[3]]
        fault_ind = (random_tensor < 1)
        random_tensor.zero_()

        # ---- generate bias ----
        if self.max_value_mode:
            fault_bias = step * 128.
            random_tensor[fault_ind] = fault_bias
        # elif n_mac != 1:
        else:
            _n_err = bitwidth_data + n_addi_affect_bits
            fault_bit_list = np.array([2**x for x in range(-n_addi_affect_bits, bitwidth_data)] + \
                                      [-2**x for x in range(-n_addi_affect_bits, bitwidth_data)],
                                      dtype=np.float32)
            size_ = fault_ind.sum().cpu().numpy()
            n_bias = n_mac if self.m_i_rate == 1.0 else\
                     torch.tensor(binom.rvs(int(n_mac), self.m_i_rate, size=size_).astype(np.float32)).to(out.device)
            random_tensor[fault_ind] = step * \
                                       (n_bias * torch.tensor(fault_bit_list[
                                           np.random.randint(
                                               0, 2 * _n_err,
                                               size=size_)])\
                                        .to(out.device)).floor()
        # else:
        #     random_tensor[fault_ind] = step * \
        #                                (torch.tensor(
        #                                    self.fault_bit_list[np.random.randint(
        #                                        0, 16, size=fault_ind.sum().cpu().data)]
        #                                ).to(out.device).floor())
        out = out + random_tensor
        # TODO: tile + cin
        # clip
        out.clamp_(min=-max_, max=max_)

        # # for masked bp
        # normal_mask = torch.ones_like(out)
        # normal_mask[fault_ind] = 0
        # masked = normal_mask * out
        # out = (out - masked).detach() + masked
        return out

    def inject(self, out, **kwargs):
        return eval("self.inject_" + self.mode)(out, **kwargs) #pylint: disable=eval-used

class FaultInjectionObjective(BaseObjective):
    NAME = "fault_injection"
    SCHEDULABLE_ATTRS = ["fault_reward_coeff", "fault_loss_coeff", "latency_reward_coeff", "inject_prob", "gaussian_std"]

    def __init__(self, search_space,
                 fault_modes="gaussian", gaussian_std=1., inject_prob=0.001, max_value_mode=True,
                 inject_macwise_inputdep_rate=1.0,
                 inject_n_cin=None,
                 inject_tile_size=None, # c_o, h_o, w_o
                 inject_propto_flops=False,
                 activation_fixed_bitwidth=None,
                 # loss
                 fault_loss_coeff=0.,
                 as_controller_regularization=False,
                 as_evaluator_regularization=False,
                 # reward
                 fault_reward_coeff=0.2,
                 latency_reward_coeff=0.,
                 calc_latency=True,
                 schedule_cfg=None):
        super(FaultInjectionObjective, self).__init__(search_space, schedule_cfg)
        assert 0. <= fault_reward_coeff <= 1.
        self.injector = FaultInjector(gaussian_std, fault_modes, inject_tile_size,
                                      max_value_mode=max_value_mode,
                                      macwise_inputdep_rate=inject_macwise_inputdep_rate)
        self.inject_n_cin = inject_n_cin
        self.injector.set_random_inject(inject_prob)
        self.fault_loss_coeff = fault_loss_coeff
        self.as_controller_regularization = as_controller_regularization
        self.as_evaluator_regularization = as_evaluator_regularization
        if self.fault_loss_coeff > 0:
            expect(self.as_controller_regularization or self.as_evaluator_regularization,
                   "When `fault_loss_coeff` > 0, you should either use this fault-injected loss"
                   " as controller regularization or as evaluator regularization, or both. "
                   "By setting `as_controller_regularization` and `as_evaluator_regularization`.",
                   ConfigException)
        self.fault_reward_coeff = fault_reward_coeff
        self.latency_reward_coeff = latency_reward_coeff
        self.calc_latency = calc_latency
        if not self.calc_latency:
            expect(latency_reward_coeff == 0,
                   "`latency_reward_coeff` must equal 0 when latency is not calculated",
                   ConfigException)
        self.inject_propto_flops = inject_propto_flops
        #if self.inject_propto_flops:
        #    expect(fault_modes == "fixed",
        #           "When `inject_propto_flops` is True, must use the bit-flip fault mode `fixed`",
        #           ConfigException)
        self.inject_prob_avg_meters = defaultdict(utils.AverageMeter)
        self.cls_inject_prob_avg_meters = defaultdict(lambda: defaultdict(utils.AverageMeter))

        self.activation_fixed_bitwidth = activation_fixed_bitwidth
        self._init_thread_local()

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(cls):
        return ["acc_clean", "acc_fault", "flops"]

    def get_reward(self, inputs, outputs, targets, cand_net):
        perfs = self.get_perfs(inputs, outputs, targets, cand_net)
        if not self.calc_latency:
            return perfs[0] * (1 - self.fault_reward_coeff) + perfs[1] * self.fault_reward_coeff
        return perfs[0] * (1 - self.fault_reward_coeff) + \
            perfs[1] * self.fault_reward_coeff + perfs[2] * self.latency_reward_coeff

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        outputs_f = cand_net.forward_one_step_callback(inputs, callback=self.inject)
        if hasattr(cand_net, "super_net"):
            cand_net.super_net.reset_flops()
        if self.calc_latency:
            cand_net.forward(inputs)
            if isinstance(cand_net, nn.DataParallel):
                flops = cand_net.module.total_flops
            else:
                flops = cand_net.super_net.total_flops if hasattr(cand_net, "super_net") else \
                        cand_net.total_flops
            if hasattr(cand_net, "super_net"):
                cand_net.super_net._flops_calculated = True
            return float(accuracy(outputs, targets)[0]) / 100, \
                float(accuracy(outputs_f, targets)[0]) / 100, \
                1 / max(flops * 1e-6 - 180, 20)
        return float(accuracy(outputs, targets)[0]) / 100, \
            float(accuracy(outputs_f, targets)[0]) / 100, \

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            inputs: data inputs
            outputs: logits
            targets: labels
        """
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if self.fault_loss_coeff > 0 and \
           ((add_controller_regularization and self.as_controller_regularization) or \
            (add_evaluator_regularization and self.as_evaluator_regularization)):
            # only forward and random inject once, this might not be of high variance
            # for differentiable controller training?
            outputs_f = cand_net.forward_one_step_callback(inputs, callback=self.inject)
            ce_loss_f = nn.CrossEntropyLoss()(outputs_f, targets)
            loss = (1 - self.fault_loss_coeff) * loss + self.fault_loss_coeff * ce_loss_f
        return loss

    def inject(self, state, context):
        # This method can be call concurrently when using `DataParallel.forward_one_step_callback`
        # Add a lock to protect the critic section
        if self.activation_fixed_bitwidth:
            # quantize the activation
            # NOTE: the quantization of the weights is done in nfp patch,
            # see `examples/fixed_point_patch.py`
            # import this manually before creating operations, or soft-link this script under
            # plugin dir to enable quantization for the weights
            state = context.last_state = self.thread_local.fix(state)
        if context.is_last_concat_op or not context.is_last_inject:
            return
        assert state is context.last_state

        with self.thread_lock:
            if self.inject_propto_flops:
                mod = context.last_conv_module
                backup_inject_prob = self.inject_prob
                if mod is None:
                    return # last op is not conv op
                if self.inject_n_cin is not None:
                    if mod.groups != 1:
                        # FIXME: currently, assume depthwise, (other group-conv not supported)
                        # each OFM value is calculated without adder tree, just MAC
                        n_mac = mod.kernel_size[0] * mod.kernel_size[1]
                    else:
                        inject_prob = 1 - (1 - backup_inject_prob) ** self.inject_n_cin
                        n_mac = np.ceil(float(mod.in_channels / self.inject_n_cin)) * \
                                mod.kernel_size[0] * mod.kernel_size[1]
                        self.inject_prob = inject_prob
                else:
                    mul_per_loc = mod.in_channels / mod.groups * \
                                  mod.kernel_size[0] * mod.kernel_size[1]
                    inject_prob = 1 - (1 - backup_inject_prob) ** mul_per_loc
                    n_mac = 1
                    self.inject_prob = inject_prob
                self.inject_prob_avg_meters[context.index].update(self.inject_prob)
                if mod.groups > 1:
                    # sep conv
                    cls_name = "conv_{}x{}".format(mod.kernel_size[0], mod.kernel_size[1])
                else:
                    # normal conv
                    cls_name = "conv_Cx{}x{}".format(mod.kernel_size[0], mod.kernel_size[1])
                self.cls_inject_prob_avg_meters[cls_name][context.index].update(self.inject_prob)
            context.last_state = self.injector.inject(state, n_mac=n_mac)
            if self.inject_propto_flops:
                self.inject_prob = backup_inject_prob

    def on_epoch_end(self, epoch):
        super(FaultInjectionObjective, self).on_epoch_end(epoch)
        if self.inject_prob_avg_meters:
            # in final trianing, if the base inject prob do not vary, the inject prob of the same
            # position/feature map should always be the same.
            stats = [(ind, meter.avg) for ind, meter in six.iteritems(self.inject_prob_avg_meters)]
            num_pos = len(stats) # number of inject position
            stats = sorted(stats, key=lambda stat: stat[1])
            mean_prob = np.mean([stat[1] for stat in stats])
            geomean_prob = np.prod([stat[1] for stat in stats])**(1.0/num_pos)
            self.logger.info("[NOTE: not meaningful in search, as every pass the same index "
                             "corresponds to different op] Num feature map injected: %3d; "
                             "Inject prob range: [%.4f (%s), %.4f (%s)]; "
                             "Mean: %.4f ; Geometric mean: %.4f",
                             num_pos, stats[0][1], stats[0][0], stats[-1][1], stats[-1][0],
                             mean_prob, geomean_prob)
            self.inject_prob_avg_meters = defaultdict(utils.AverageMeter) # reset

            # mean according to operation types
            for cls_name, avg_meters in sorted(self.cls_inject_prob_avg_meters.items(),
                                               key=lambda item: item[0]):
                stats = [(ind, meter.avg) for ind, meter in six.iteritems(avg_meters)]
                num_pos = len(stats) # number of inject position
                stats = sorted(stats, key=lambda stat: stat[1])
                mean_prob = np.mean([stat[1] for stat in stats])
                geomean_prob = np.prod([stat[1] for stat in stats])**(1.0/num_pos)
                self.logger.info("Type: %s: Num feature map injected: %3d; "
                                 "Inject prob range: [%.4f (%s), %.4f (%s)]; "
                                 "Mean: %.4f ; Geometric mean: %.4f", cls_name,
                                 num_pos, stats[0][1], stats[0][0], stats[-1][1], stats[-1][0],
                                 mean_prob, geomean_prob)
            self.cls_inject_prob_avg_meters = defaultdict(_get_average_meter_defaultdict)

    @property
    def inject_tile_size(self):
        return self.injector.tile_size

    @inject_tile_size.setter
    def inject_tile_size(self, tile_size):
        self.injector.tile_size = tile_size

    @property
    def inject_prob(self):
        return self.injector.random_inject

    @inject_prob.setter
    def inject_prob(self, value):
        self.injector.set_random_inject(value)

    @property
    def gaussian_std(self):
        return self.injector.gaussian_std

    @gaussian_std.setter
    def gaussian_std(self, value):
        self.injector.set_gaussian_std(value)

    def __getstate__(self):
        state = super(FaultInjectionObjective, self).__getstate__()
        del state["thread_lock"]
        if "thread_local" in state:
            del state["thread_local"]
        return state

    def __setstate__(self, state):
        super(FaultInjectionObjective, self).__setstate__(state)
        self._init_thread_local()

    def _init_thread_local(self):
        if self.activation_fixed_bitwidth:
            import nics_fix_pt.nn_fix as nfp
            self.thread_local = utils.LazyThreadLocal(creator_map={
                "fix": lambda: nfp.Activation_fix(nf_fix_params={
                    "activation": {
                        # auto fix
                        "method": torch.autograd.Variable(torch.IntTensor(np.array([1])),
                                                          requires_grad=False),
                        # not meaningful
                        "scale": torch.autograd.Variable(torch.IntTensor(np.array([0])),
                                                         requires_grad=False),
                        "bitwidth": torch.autograd.Variable(torch.IntTensor(
                            np.array([self.activation_fixed_bitwidth])),
                                                            requires_grad=False)
                    }
                })
            })
        self.thread_lock = threading.Lock()
