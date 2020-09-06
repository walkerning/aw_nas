#pylint: disable=invalid-name
"""
Script for patching fixed point modules.
"""
import time
import six
import numpy as np
import torch
from torch import nn
import nics_fix_pt.nn_fix as nfp
from nics_fix_pt.fix_modules import register_fix_module
from aw_nas.weights_manager.super_net import *
from aw_nas.objective.fault_injection import *
from aw_nas.final.cnn_model import *

BITWIDTH = 8
FIX_METHOD = 1 # auto_fix

INJECT = True
#INJECT_PROB = 5e-2
#INJECT_PROB = 1e-1
SAF1_RATIO = 0.163
INJECT_STEP = 1
MASK_THRESH = 2**3

def _generate_default_fix_cfg(names, scale=0, bitwidth=8, method=0):
    return {n: {
        "method": torch.autograd.Variable(torch.IntTensor(np.array([method])), requires_grad=False),
        "scale": torch.autograd.Variable(torch.IntTensor(np.array([scale])), requires_grad=False),
        "bitwidth": torch.autograd.Variable(torch.IntTensor(np.array([bitwidth])), requires_grad=False)
    } for n in names}

class FixedConv(nfp.Conv2d_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedConv, self).__init__(*args, **kwargs)

def num_one(num):
    counter = 0
    while(num > 0):
        counter += num % 2
        num //= 2
    return counter

INJECT_PROB = 0.025
TABLE_SIZE = int(1 / INJECT_PROB * 100000)
num_ones = [num_one(x) for x in range(128)]

# position mask (1 indicate that error occurs at this position, 0 otherwise)
pos_mask = [0.1**num_ones[x]*0.9**(7 - num_ones[x]) for x in range(128)]
reg = sum(pos_mask) # should be ~ 1.0
pos_mask = [x/reg for x in pos_mask]
for i in range(1, len(pos_mask)):
    pos_mask[i] += pos_mask[i - 1]
# rescale 0~1 -> 0~TABLE_SIZE to enable quick array index access
pos_mask = [int(x * TABLE_SIZE) for x in pos_mask]
pos_mask_dict = []
ind = 0
for i in range(TABLE_SIZE):
    if i >= pos_mask[ind]:
        ind += 1
    pos_mask_dict.append(ind)
pos_mask_dict = torch.Tensor(pos_mask_dict).cuda().to(torch.int)
# 47.83% prob no error

# saf 0 or 1 (1 indicate saf1, 0 indicate saf0)
saf_mask = [0.16**num_ones[x]*0.84**(7 - num_ones[x]) for x in range(128)]
reg = sum(saf_mask) # should be ~ 1.0
saf_mask = [x/reg for x in saf_mask]
for i in range(1, len(saf_mask)):
    saf_mask[i] += saf_mask[i - 1]
saf_mask = [int(x * TABLE_SIZE) for x in saf_mask]
saf_mask_dict = []
ind = 0
for i in range(TABLE_SIZE):
    if i >= saf_mask[ind]:
        ind += 1
    saf_mask_dict.append(ind)
saf_mask_dict = torch.Tensor(saf_mask_dict).cuda().to(torch.int)

class RRAMConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(RRAMConv, self).__init__(*args, **kwargs)
        self.step = 0
        self.random_tensor_pos = None
        self.random_tensor_neg = None
        self.inject_prob = 0

    def set_saf_ratio(self, ratio):
        self.inject_prob = ratio

    def forward(self, x):
        if self.inject_prob < 1e-8:
            return super(RRAMConv, self).forward(x)
        param = self.weight
        # param_size = param.shape[0]*param.shape[1]*param.shape[2]*param.shape[3]
        # max_ = torch.max(torch.abs(param)).cpu().data
        # scale = torch.ceil(torch.log(
        #        torch.max(max_, torch.tensor(1e-5).float())) / np.log(2.)).to(param.device)
        scale = torch.ceil(torch.log(
            torch.max(torch.max(torch.abs(param)),
                      torch.tensor(1e-5).float().to(param.device))) / np.log(2.))
        step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]).to(param.device),
                                                 requires_grad=False), (scale.float() - 7.))
        # random_tensor_pos = torch.multinomial(pos_mask, param_size, True)\
        #                          .reshape(param.shape).to(torch.int).to(param.device)
        # random_tensor_neg = torch.multinomial(pos_mask, param_size, True)\
        #                          .reshape(param.shape).to(torch.int).to(param.device)
        # saf_pos = torch.multinomial(saf_mask, param_size, True)\
        #                .reshape(param.shape).to(torch.int).to(param.device)
        # saf_neg = torch.multinomial(saf_mask, param_size, True)\
        #                .reshape(param.shape).to(torch.int).to(param.device)
        random_tensor_pos = pos_mask_dict[param.new(param.size()).random_(0, TABLE_SIZE)\
                                          .to(torch.long)]\
                            .reshape(param.shape).to(torch.int).to(param.device)
        random_tensor_neg = pos_mask_dict[param.new(param.size()).random_(0, TABLE_SIZE)\
                                          .to(torch.long)]\
                            .reshape(param.shape).to(torch.int).to(param.device)
        saf_pos = saf_mask_dict[param.new(param.size()).random_(0, TABLE_SIZE).to(torch.long)]\
                  .reshape(param.shape).to(torch.int).to(param.device)
        saf_neg = saf_mask_dict[param.new(param.size()).random_(0, TABLE_SIZE).to(torch.long)]\
                  .reshape(param.shape).to(torch.int).to(param.device)
        param_int = (param / step).to(torch.int)
        param_pos = torch.zeros(param_int.shape).to(param.device).to(torch.int)
        param_neg = torch.zeros(param_int.shape).to(param.device).to(torch.int)
        param_pos[param_int > 0] = param_int[param_int > 0]
        param_neg[param_int < 0] = -param_int[param_int < 0]
        param_pos = saf_pos & random_tensor_pos | param_pos & (~random_tensor_pos)
        param_neg = saf_neg & random_tensor_neg | param_neg & (~random_tensor_neg)
        new_param_int = param_pos - param_neg
        new_param = new_param_int.to(torch.float) * step

        # for masked bp
        normal_mask = torch.ones_like(param)
        normal_mask[torch.abs(new_param_int - param_int) > MASK_THRESH] = 0
        masked = normal_mask * param
        param = (param - masked).detach() + masked

        object.__setattr__(self, "weight", new_param)
        out = super(RRAMConv, self).forward(x)
        return out

register_fix_module(RRAMConv)

class RRAMFixedConv(nfp.RRAMConv_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(RRAMFixedConv, self).__init__(*args, **kwargs)

class FixedLinear(nfp.Linear_fix):
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias"], method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedLinear, self).__init__(*args, **kwargs)

class FixedBatchNorm2d(nfp.BatchNorm2d_fix):
    """
    FIXME:
      When training using multi gpu, the update for the buffers(running_mean/running_var)
      is done using data on only the main gpu; This will lead to discrepancy between
      one-gpu/multi-gpu training. Maybe can consider using `encoding.nn.BatchNorm2d`.

      Moreover, as the fixed-point configuration is shared between threads, and each gpu
      separately computes using its own data split, the unprotected concurrent read/write
      to this shared config lead to undefined behaviour. Maybe this config should be
      thread-local too. (Maybe this should be a fix in the nics_fix_pytorch lib).

      An temporary fix is not to quantize the `running_mean` and `running_var` here.
    """
    def __init__(self, *args, **kwargs):
        kwargs["nf_fix_params"] = _generate_default_fix_cfg(
            ["weight", "bias", "running_mean", "running_var"],
            method=FIX_METHOD, bitwidth=BITWIDTH)
        super(FixedBatchNorm2d, self).__init__(*args, **kwargs)

if INJECT:
    nn.Conv2d = RRAMFixedConv
else:
    nn.Conv2d = FixedConv
#nn.BatchNorm2d = FixedBatchNorm2d
nn.Linear = FixedLinear

class CNNGenotypeModelPatch(CNNGenotypeModel):
    NAME = "cnn_patch_final_model"

    def set_saf_ratio(self, ratio):
        for idx, _module in self.named_modules():
            if isinstance(_module, nn.Conv2d):
                _module.set_saf_ratio(ratio)

class SubCandidateNetPatch(SubCandidateNet):
    def set_saf_ratio(self, ratio):
        for idx, _module in self.named_modules():
            if isinstance(_module, nn.Conv2d):
                _module.set_saf_ratio(ratio)


class SuperNetPatch(SuperNet):
    NAME = "supernet_patch"

    def __init__(self, search_space, device, rollout_type="discrete",
                 gpus=tuple(),
                 num_classes=10, init_channels=16, stem_multiplier=3,
                 max_grad_norm=5.0, dropout_rate=0.1,
                 use_stem="conv_bn_3x3", stem_stride=1, stem_affine=True,
                 cell_use_preprocess=True, preprocess_op_type=None, cell_group_kwargs=None,
                 candidate_member_mask=True, candidate_cache_named_members=False,
                 candidate_virtual_parameter_only=False, candidate_eval_no_grad=True):
        super(SuperNetPatch, self).__init__(search_space, device, rollout_type,
                                       gpus=gpus,
                                       num_classes=num_classes, init_channels=init_channels,
                                       stem_multiplier=stem_multiplier,
                                       max_grad_norm=max_grad_norm, dropout_rate=dropout_rate,
                                       use_stem=use_stem, stem_stride=stem_stride,
                                       stem_affine=stem_affine,
                                       cell_use_preprocess=cell_use_preprocess,
                                       preprocess_op_type=preprocess_op_type,
                                       cell_group_kwargs=cell_group_kwargs,
                                       candidate_member_mask=candidate_member_mask, candidate_cache_named_members=candidate_cache_named_members,
                                       candidate_virtual_parameter_only=candidate_virtual_parameter_only, candidate_eval_no_grad=candidate_eval_no_grad)

    def assemble_candidate(self, rollout):
        return SubCandidateNetPatch(self, rollout,
                               gpus=self.gpus,
                               member_mask=self.candidate_member_mask,
                               cache_named_members=self.candidate_cache_named_members,
                               virtual_parameter_only=self.candidate_virtual_parameter_only,
                               eval_no_grad=self.candidate_eval_no_grad)


class RRAMSafObjective(BaseObjective):
    NAME = "saf_injection"
    SCHEDULABLE_ATTRS = ["fault_reward_coeff", "fault_loss_coeff", "latency_reward_coeff", "inject_prob"]

    def __init__(self, search_space,
                 inject_prob=0.1,
                 activation_fixed_bitwidth=None,
                 # loss
                 fault_loss_coeff=0.,
                 as_controller_regularization=False,
                 as_evaluator_regularization=False,
                 # reward
                 fault_reward_coeff=0.2,
                 latency_reward_coeff=0.2,
                 schedule_cfg=None):
        super(RRAMSafObjective, self).__init__(search_space, schedule_cfg)
        assert 0. <= fault_reward_coeff <= 1.
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
        self.inject_prob = inject_prob
        self.activation_fixed_bitwidth = activation_fixed_bitwidth
        self._init_thread_local()

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(cls):
        return ["acc_clean", "acc_fault", "flops"]

    def get_reward(self, inputs, outputs, targets, cand_net):
        perfs = self.get_perfs(inputs, outputs, targets, cand_net)
        return perfs[0] * (1 - self.fault_reward_coeff) + perfs[1] * self.fault_reward_coeff + perfs[2] * self.latency_reward_coeff

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        if hasattr(cand_net, "super_net"):
            cand_net.super_net.reset_flops()
        cand_net.set_saf_ratio(self.inject_prob)
        outputs_f = cand_net.forward(inputs)
        cand_net.set_saf_ratio(0)
        flops = cand_net.super_net.total_flops if hasattr(cand_net, "super_net") else cand_net.total_flops
        if hasattr(cand_net, "super_net"):
            cand_net.super_net._flops_calculated = True
        return float(accuracy(outputs, targets)[0]) / 100, \
            float(accuracy(outputs_f, targets)[0]) / 100, \
            1 / max(flops * 1e-6 - 180, 20)

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
            cand_net.set_saf_ratio(self.inject_prob)
            outputs_f = cand_net.forward(inputs)
            cand_net.set_saf_ratio(0)
            ce_loss_f = nn.CrossEntropyLoss()(outputs_f, targets)
            loss = (1 - self.fault_loss_coeff) * loss + self.fault_loss_coeff * ce_loss_f
        return loss

    def on_epoch_end(self, epoch):
        super(RRAMSafObjective, self).on_epoch_end(epoch)

    def __getstate__(self):
        state = super(RRAMSafObjective, self).__getstate__()
        del state["thread_lock"]
        if "thread_local" in state:
            del state["thread_local"]
        return state

    def __setstate__(self, state):
        super(RRAMSafObjective, self).__setstate__(state)
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
