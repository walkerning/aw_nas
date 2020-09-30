"""
2-layer controller.
"""

from aw_nas import utils, assert_rollout_type
from aw_nas.utils import DistributedDataParallel
from aw_nas.controller.base import BaseController
from aw_nas.btcs.layer2.search_space import (
    Layer2Rollout,
    Layer2DiffRollout,
    DenseMicroRollout,
    DenseMicroDiffRollout,
    StagewiseMacroRollout,
    StagewiseMacroDiffRollout,
    SinkConnectMacroDiffRollout,
)

from collections import OrderedDict

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    # from torch.nn.SyncBatchNorm import convert_sync_batch_norm as convert_sync_bn
    from torch.nn import SyncBatchNorm

    convert_sync_bn = SyncBatchNorm.convert_sync_batchnorm
except ImportError:
    convert_sync_bn = lambda m: m


class Layer2Optimizer(optim.Optimizer):
    def __init__(self, params, **opt_cfg):
        super(Layer2Optimizer, self).__init__([torch.tensor([])], defaults={})
        macro_opt_type = opt_cfg["macro"].pop("type")
        micro_opt_type = opt_cfg["micro"].pop("type")
        # currently width alphas & macro-alpha share the same optimizer
        self.macro_optimizer = getattr(optim, macro_opt_type)(
            nn.ParameterList(params[0:2]), **opt_cfg["macro"]
        )  # after adding width-alphas, as 2nd
        self.micro_optimizer = getattr(optim, micro_opt_type)(
            nn.ParameterList(params[2:]), **opt_cfg["micro"]
        )

    def step(self):
        self.macro_optimizer.step()
        self.micro_optimizer.step()


torch.optim.layer2 = Layer2Optimizer  # add patch the torch optim


class Layer2DiffController(BaseController, nn.Module):
    NAME = "layer2-differentiable"

    def __init__(
        self,
        search_space,
        rollout_type,
        mode="eval",
        device="cuda",
        macro_controller_type="random_sample",
        macro_controller_cfg={},
        micro_controller_type="random_sample",
        micro_controller_cfg={},
        inspect_hessian_every=-1,
        save_alphas_every=-1,
        multiprocess=False,
        schedule_cfg=None,
    ):
        super(Layer2DiffController, self).__init__(
            search_space, rollout_type, schedule_cfg=schedule_cfg
        )

        nn.Module.__init__(self)

        self.search_space = search_space
        self.rollout_type = rollout_type
        self.device = device
        self.to(self.device)

        self.inspect_hessian_every = inspect_hessian_every
        self.inspect_hessian = False

        self.save_alphas_every = save_alphas_every
        self.save_alphas = False
        self.saved_dict = {
            "macro": [],
            "micro": [],
            "width": [],
        }

        self.multiprocess = multiprocess

        # the macro/micro controllers
        if macro_controller_type == "macro-stagewise-diff":
            self.macro_controller = MacroStagewiseDiffController(
                self.search_space.macro_search_space,
                macro_controller_type,
                device=self.device,
                multiprocess=self.multiprocess,
                **macro_controller_cfg,
            )
        elif macro_controller_type == "macro-sink-connect-diff":
            self.macro_controller = MacroSinkConnectDiffController(
                self.search_space.macro_search_space,
                macro_controller_type,
                device=self.device,
                multiprocess=self.multiprocess,
                **macro_controller_cfg,
            )
        else:
            raise NotImplementedError()

        if micro_controller_type == "micro-dense-diff":
            self.micro_controller = MicroDenseDiffController(
                self.search_space.micro_search_space,
                micro_controller_type,
                device=self.device,
                multiprocess=self.multiprocess,
                **micro_controller_cfg,
            )
        else:
            raise NotImplementedError()

        object.__setattr__(self, "parallel_model", self)
        self._parallelize()

    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self).to(self.device)
            object.__setattr__(
                self,
                "parallel_model",
                DistributedDataParallel(
                    self, (self.device,), find_unused_parameters=True
                ),
            )

    def on_epoch_start(self, epoch):
        super(Layer2DiffController, self).on_epoch_start(epoch)
        if self.inspect_hessian_every >= 0 and epoch % self.inspect_hessian_every == 0:
            self.inspect_hessian = True
        if self.save_alphas_every >= 0 and epoch % self.save_alphas_every == 0:
            self.save_alphas = True

        # save alphas every epoch
        if self.save_alphas:
            self.saved_dict["macro"].append(
                [alpha.data.cpu() for alpha in self.macro_controller.cg_alphas]
            )
            self.saved_dict["micro"].append(
                [alpha.data.cpu() for alpha in self.micro_controller.cg_alphas]
            )
            self.saved_dict["width"].append(
                [
                    width_alpha.cpu()
                    for width_alpha in self.macro_controller.width_alphas
                ]
            )

        self.macro_controller.on_epoch_start(epoch)
        self.micro_controller.on_epoch_start(epoch)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_mode(self, mode):
        super(Layer2DiffController, self).set_mode(mode)

        if mode == "train":
            nn.Module.train(self)
        elif mode == "eval":
            nn.Module.eval(self)
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def parameters(self, recurse=False):
        # FIXME: normal nn.module.parameters() use recurse=True to acquire all params
        param_list = nn.ParameterList([])
        param_list.extend(self.macro_controller.parameters())
        param_list.extend(self.micro_controller.parameters())
        return param_list

    def _entropy_loss(self):
        return (
            self.macro_controller._entropy_loss()
            + self.micro_controller._entropy_loss()
        )

    def sample(self, n=1, batch_size=1):
        if self.multiprocess:
            return self.parallel_model.forward(n=n, batch_size=batch_size)
        else:
            return self.forward(n=n, batch_size=batch_size)

    def forward(self, n=1, batch_size=1):
        rollouts = []
        macro_rollouts = self.macro_controller.forward(n=n, batch_size=batch_size)
        micro_rollouts = self.micro_controller.forward(n=n, batch_size=batch_size)
        for i in range(n):
            rollouts.append(
                Layer2DiffRollout(
                    macro_rollouts[i], micro_rollouts[i], self.search_space
                )
            )

        return rollouts

    def gradient(self, loss, return_grads=True, zero_grads=True):
        if zero_grads:
            self.zero_grad()

        if self.inspect_hessian:
            for name, param in self.named_parameters():
                max_eig = utils.torch_utils.max_eig_of_hessian(loss, param)
                self.logger.info("Max eigenvalue of Hessian of %s: %f", name, max_eig)

        _loss = loss + self._entropy_loss()
        _loss.backward()
        if return_grads:
            return utils.get_numpy(_loss), [
                (k, v.grad.clone()) for k, v in self.named_parameters()
            ]
        return utils.get_numpy(_loss)

    def step_current_gradient(self, optimizer):
        self.macro_controller.step_current_gradient(optimizer.macro_optimizer)
        self.micro_controller.step_current_gradient(optimizer.micro_optimizer)

    def step_gradient(self, gradients, optimizer):
        self.macro_controller.step_gradient(gradients[0], optimizer.macro_optimizer)
        self.micro_controller.step_gradient(gradients[1], optimizer.micro_optimizer)

    def step(self, rollouts, optimizer, perf_name):
        macro_rollouts = [r.macro for r in rollouts]
        micro_rollouts = [r.micro for r in rollouts]
        macro_loss = self.macro_controller.step(
            macro_rollouts, optimizer.macro_optimizer, perf_name
        )
        micro_loss = self.micro_controller.step(
            micro_rollouts, optimizer.micro_optimizer, perf_name
        )
        return macro_loss, micro_loss

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        macro_rollouts = [r.macro for r in rollouts]
        micro_rollouts = [r.micro for r in rollouts]
        self.macro_controller.summary(
            macro_rollouts, log=log, log_prefix=log_prefix, step=None
        )
        self.micro_controller.summary(
            micro_rollouts, log=log, log_prefix=log_prefix, step=None
        )

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)
        """save alphas"""
        if self.save_alphas_every is not None:
            # os.path.dirname means the parent path of the `PATH`
            torch.save(
                self.saved_dict,
                os.path.join(os.path.dirname(os.path.dirname(path)), "alphas.pth"),
            )


    def load(self, path):
        """Load the parameters from disk."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    # since the layer2controller.parameters() is a list of [macro_parameters(), micro_parameters()], we need to override the zero_grad() since it used model.parameters()
    def zero_grad(self):
        for param in self.parameters():
            for p in param:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @classmethod
    def supported_rollout_types(cls):
        return ["layer2", "layer2-differentiable"]


class GetArchMacro(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        search_space,
        op_weights,
        device,
        i_stage,
    ):
        stage_conn = torch.zeros(
            (
                search_space.stage_node_nums[i_stage],
                search_space.stage_node_nums[i_stage],
            )
        ).to(device)
        stage_conn[search_space.idxes[i_stage]] = op_weights
        ctx.save_for_backward(
            torch.as_tensor(op_weights), torch.as_tensor(search_space.idxes[i_stage])
        )

        return stage_conn

    @staticmethod
    def backward(ctx, grad_output):
        op_weights, idxes = ctx.saved_tensors
        op_weights_grad = grad_output[idxes[0], idxes[1]]
        return None, op_weights_grad, None, None, None


class MacroStagewiseDiffController(BaseController, nn.Module):
    NAME = "macro-stagewise-diff"

    SCHEDULABLE_ATTRS = [
        "gumbel_temperature",
        "entropy_coeff",
        "force_uniform",
        "width_gumbel_temperature",
        "width_entropy_coeff",
    ]

    def __init__(
        self,
        search_space,
        rollout_type,
        mode="eval",
        device="cuda",
        use_prob=False,
        gumbel_hard=False,
        gumbel_temperature=1.0,
        use_sigmoid=False,
        use_edge_normalization=False,
        entropy_coeff=0.01,
        max_grad_norm=None,
        force_uniform=False,
        full_init=False,  # use all-one initialization and big flops reg
        progressive_pruning_th=None,
        multiprocess=False,
        per_stage_width=True,  # default use per stage width
        width_entropy_coeff=0.01,
        width_gumbel_temperature=1.0,
        schedule_cfg=None,
    ):
        super(MacroStagewiseDiffController, self).__init__(
            search_space, rollout_type, schedule_cfg=schedule_cfg
        )
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.use_prob = use_prob
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature
        self.use_sigmoid = use_sigmoid
        # use_prob / use_sigmoid should not the True at the same time
        # if both false use plain gumbel softmax
        assert not (use_prob and use_sigmoid)

        # edge normalization
        self.use_edge_normalization = use_edge_normalization

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        self.progressive_pruning_th = progressive_pruning_th
        self.width_choice = self.search_space.width_choice

        self.multiprocess = multiprocess

        self.per_stage_width = per_stage_width
        self.width_gumbel_temperature = width_gumbel_temperature
        self.width_entropy_coeff = width_entropy_coeff

        # generate parameters
        self.full_init = full_init
        if not self.full_init:
            init_value = 1.0e-3
        else:
            init_value = 1.0

        self.cg_alphas = nn.ParameterList(
            [
                nn.Parameter(
                    init_value * torch.randn(sum(self.search_space.num_possible_edges))
                )
            ]
        )
        # width choices [#cells , #width_choice]
        if self.width_choice is not None:
            if not self.per_stage_width:
                self.width_alphas = nn.ParameterList(
                    [
                        nn.Parameter(
                            init_value
                            * torch.randn(
                                len(self.search_space.cell_layout),
                                len(self.width_choice),
                            )
                        )
                    ]
                )
            else:
                self.width_alphas = nn.ParameterList(
                    [
                        nn.Parameter(
                            init_value
                            * torch.randn(
                                len(self.search_space.stage_node_nums),
                                len(self.width_choice),
                            )
                        )
                    ]
                )

        self.stage_num_alphas = (
            self.search_space.num_possible_edges
        )  # used for competible with sink-connecting ss

        if self.use_edge_normalization:
            raise NotImplementedError("MacroDiffController does not support edge-norm")
        else:
            self.cg_betas = None
        self.get_arch = GetArchMacro()

        self.to(self.device)

    def set_mode(self, mode):
        super(MacroStagewiseDiffController, self).set_mode(mode)
        if mode == "train":
            nn.Module.train(self)
        elif mode == "eval":
            nn.Module.eval(self)
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def set_device(self, device):
        self.device = device
        self.to(device)

    def progressive_pruning(self):
        for alpha in self.cg_alphas:
            # inpalce replace alphas that smaller than the pruning threshold, no grad
            alpha.data = alpha * (alpha.gt(self.progressive_pruning_th).float())

    def forward(self, n=1, batch_size=1):
        return self.sample(n=n, batch_size=batch_size)

    def sample(self, n=1, batch_size=1):
        if self.progressive_pruning_th is not None:
            self.progressive_pruning()

        width_arch, width_logits = self.sample_width(n=n, batch_size=batch_size)

        rollouts = []
        for i_sample in range(n):
            # op_weights.shape: [num_edges, [batch_size,] num_ops]
            # edge_norms.shape: [num_edges] do not have batch_size.
            op_weights_list = []
            edge_norms_list = []
            sampled_list = []
            logits_list = []

            for alphas in self.cg_alphas:
                if (
                    self.progressive_pruning_th is not None
                    and self.progressive_pruning_th > 0
                ):
                    alphas = alphas.clamp(self.progressive_pruning_th, 1.0e4)
                else:
                    pass
                if self.force_uniform:  # cg_alpha parameters will not be in the graph
                    # NOTE: `force_uniform` config does not affects edge_norms (betas),
                    # if one wants a force_uniform search, keep `use_edge_normalization=False`
                    alphas = torch.zeros_like(alphas)

                if batch_size > 1:
                    expanded_alpha = (
                        alphas.reshape([alphas.shape[0], 1, alphas.shape[1]])
                        .repeat([1, batch_size, 1])
                        .reshape([-1, alphas.shape[-1]])
                    )
                else:
                    expanded_alpha = alphas
                if self.use_prob:
                    sampled = F.softmax(
                        expanded_alpha / self.gumbel_temperature, dim=-1
                    )
                elif self.use_sigmoid:
                    sampled = utils.relaxed_bernoulli_sample(
                        expanded_alpha, self.gumbel_temperature
                    )
                else:
                    # gumbel sampling
                    sampled, _ = utils.gumbel_softmax(
                        expanded_alpha, self.gumbel_temperature, hard=False
                    )

                if self.gumbel_hard:
                    op_weights = utils.straight_through(sampled)
                else:
                    op_weights = sampled

                if batch_size > 1:
                    sampled = sampled.reshape([-1, batch_size, op_weights.shape[-1]])
                    op_weights = op_weights.reshape(
                        [-1, batch_size, op_weights.shape[-1]]
                    )

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampled))
                # logits_list.append(utils.get_numpy(alphas))
                logits_list.append(alphas)

                stage_conns = []
                split_op_weights = torch.split(op_weights, self.stage_num_alphas)

                for i_stage in range(self.search_space.stage_num):
                    stage_conn = self.get_arch.apply(
                        self.search_space,
                        split_op_weights[i_stage],
                        self.device,
                        i_stage,
                    )
                    stage_conns.append(stage_conn)

            rollouts.append(
                StagewiseMacroDiffRollout(
                    arch=stage_conns,
                    sampled=sampled_list,
                    logits=logits_list,
                    width_arch=width_arch[i_sample],
                    width_logits=width_logits[i_sample],
                    search_space=self.search_space,
                )
            )

        return rollouts

    def sample_width(self, n=1, batch_size=1):
        assert batch_size == 1, "sample_width should not have batch size > 1"
        width_sampled_list = []
        width_logits_list = []
        width_op_weights_list = []

        for _ in range(n):
            # sample the width alphas
            for width_alphas in self.width_alphas:
                if self.force_uniform:  # cg_alpha parameters will not be in the graph
                    # NOTE: `force_uniform` config does not affects edge_norms (betas),
                    # if one wants a force_uniform search, keep `use_edge_normalization=False`
                    width_alphas = torch.zeros_like(width_alphas)

                if batch_size > 1:
                    expanded_width_alpha = (
                        width_alphas.reshape(
                            [width_alphas.shape[0], 1, width_alphas.shape[1]]
                        )
                        .repeat([1, batch_size, 1])
                        .reshape([-1, width_alphas.shape[-1]])
                    )
                else:
                    expanded_width_alpha = width_alphas

                if self.use_prob:
                    width_sampled = F.softmax(
                        expanded_width_alpha / self.width_gumbel_temperature, dim=-1
                    )
                elif self.use_sigmoid:
                    width_sampled = utils.relaxed_bernoulli_sample(
                        expanded_width_alpha, self.width_gumbel_temperature
                    )
                else:
                    # gumbel sampling
                    width_sampled, _ = utils.gumbel_softmax(
                        expanded_width_alpha, self.width_gumbel_temperature, hard=False
                    )

                if self.gumbel_hard:
                    width_op_weights = utils.straight_through(width_sampled)
                else:
                    width_op_weights = width_sampled

                if batch_size > 1:
                    width_sampled = width_sampled.reshape(
                        [-1, batch_size, width_op_weights.shape[-1]]
                    )
                    width_op_weights = width_op_weights.reshape(
                        [-1, batch_size, width_op_weights.shape[-1]]
                    )

                if not self.per_stage_width:
                    width_op_weights_full = width_op_weights
                    width_sampled_full = width_sampled
                    width_alphas_full = width_alphas
                else:
                    # the last stage has one more node
                    node_list = self.search_space.stage_node_nums.copy()
                    # let the 1st stage num_node -1
                    # to let all reduction cell uses the width-alphas of next stage
                    node_list[0] = node_list[0] - 1

                    width_op_weights_full = torch.cat(
                        [
                            width_op_weights[idx_stage].repeat(num_nodes - 1, 1)
                            for idx_stage, num_nodes in enumerate(node_list)
                        ]
                    )

                    width_sampled_full = torch.cat(
                        [
                            width_sampled[idx_stage].repeat(num_nodes - 1, 1)
                            for idx_stage, num_nodes in enumerate(node_list)
                        ]
                    )

                    width_alphas_full = torch.cat(
                        [
                            width_alphas[idx_stage].repeat(num_nodes - 1, 1)
                            for idx_stage, num_nodes in enumerate(node_list)
                        ]
                    )

            width_op_weights_list.append(width_op_weights_full)
            width_sampled_list.append(utils.get_numpy(width_sampled_full))
            # logits_list.append(utils.get_numpy(alphas))
            width_logits_list.append(width_alphas_full)

        return width_op_weights_list, width_logits_list

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        """Load the parameters from disk."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    def _entropy_loss(self):
        ent_loss = 0.0
        if self.entropy_coeff > 0:
            alphas = self.cg_alphas[0].split(
                [i - 1 for i in self.search_space.stage_node_nums]
            )
            probs = [F.softmax(alpha, dim=-1) for alpha in self.cg_alphas]
            ent_loss = (
                self.entropy_coeff
                * sum(-(torch.log(prob) * prob).sum() for prob in probs)
                + ent_loss
            )
        if self.width_entropy_coeff > 0:
            width_alphas = self.width_alphas
            probs = [F.softmax(alpha, dim=-1) for alpha in self.width_alphas]
            ent_loss = (
                self.width_entropy_coeff
                * sum(-(torch.log(prob) * prob).sum() for prob in probs)
                + ent_loss
            )

        return ent_loss

    def gradient(self, loss, return_grads=True, zero_grads=True):
        raise NotImplementedError(
            "the grad function is implemented in the layer2diffcontroller.gradient()"
        )

    def step_current_gradient(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step_gradient(self, gradients, optimizer):
        self.zero_grad()
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # clip the gradients
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def step(self, rollouts, optimizer, perf_name):  # very memory inefficient
        self.zero_grad()
        losses = [r.get_perf(perf_name) for r in rollouts]
        optimizer.step()
        [l.backward() for l in losses]
        return np.mean([l.detach().cpu().numpy() for l in losses])

    def __getstate__(self):
        state = super(MacroStagewiseDiffController, self).__getstate__().copy()
        del state["get_arch"]
        return state

    def __setstate__(self, state):
        super(MacroStagewiseDiffController, self).__setstate__(state)
        self.get_arch = GetArchMacro()

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        num = len(rollouts)
        logits_list = [
            [utils.get_numpy(logits) for logits in r.logits] for r in rollouts
        ]
        _ss = self.search_space
        if self.gumbel_hard:
            cg_logprobs = [0.0 for _ in range(_ss.num_cell_groups)]
        cg_entros = [0.0 for _ in range(_ss.num_cell_groups)]
        for rollout, logits in zip(rollouts, logits_list):
            for cg_idx, (vec, cg_logits) in enumerate(zip(rollout.arch, logits)):
                prob = utils.softmax(cg_logits)
                logprob = np.log(prob)
                if self.gumbel_hard:
                    inds = np.argmax(utils.get_numpy(vec.op_weights), axis=-1)
                    cg_logprobs[cg_idx] += np.sum(logprob[range(len(inds)), inds])
                cg_entros[cg_idx] += -(prob * logprob).sum()

        # mean across rollouts
        if self.gumbel_hard:
            cg_logprobs = [s / num for s in cg_logprobs]
            total_logprob = sum(cg_logprobs)
            cg_logprobs_str = ",".join(["{:.2f}".format(n) for n in cg_logprobs])

        cg_entros = [s / num for s in cg_entros]
        total_entro = sum(cg_entros)
        cg_entro_str = ",".join(["{:.2f}".format(n) for n in cg_entros])

        if log:
            # maybe log the summary
            self.logger.info(
                "%s%d rollouts: %s ENTROPY: %2f (%s)",
                log_prefix,
                num,
                "-LOG_PROB: %.2f (%s) ;" % (-total_logprob, cg_logprobs_str)
                if self.gumbel_hard
                else "",
                total_entro,
                cg_entro_str,
            )
            if step is not None and not self.writer.is_none():
                if self.gumbel_hard:
                    self.writer.add_scalar("log_prob", total_logprob, step)
                self.writer.add_scalar("entropy", total_entro, step)

        stats = [
            (n + " ENTRO", entro) for n, entro in zip(_ss.cell_group_names, cg_entros)
        ]
        if self.gumbel_hard:
            stats += [
                (n + " LOGPROB", logprob)
                for n, logprob in zip(_ss.cell_group_names, cg_logprobs)
            ]
        return OrderedDict(stats)

    @classmethod
    def supported_rollout_types(cls):
        return ["macro-stagewise", "macro-stagewise-diff", "macro-sink-connect-diff"]


class GetArchMacroSinkConnect(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        search_space,
        op_weights,
        device,
        i_stage,
    ):
        stage_conn = torch.zeros(
            (
                search_space.stage_node_nums[i_stage],
                search_space.stage_node_nums[i_stage],
            )
        ).to(device)
        stage_conn[np.arange(len(op_weights)) + 1, np.arange(len(op_weights))] = 1
        stage_conn[-1, : len(op_weights)] = op_weights
        ctx.save_for_backward(
            torch.as_tensor(op_weights), torch.as_tensor(search_space.idxes[i_stage])
        )
        return stage_conn

    @staticmethod
    def backward(ctx, grad_output):
        op_weights, idxes = ctx.saved_tensors
        op_weights_grad = grad_output[-1, : len(op_weights)]
        return None, op_weights_grad, None, None, None


class MacroSinkConnectDiffController(MacroStagewiseDiffController):
    NAME = "macro-sink-connect-diff"
    # The TF_NAS-like macro search space(sink-based connecting)
    # during each stage, before the reduction node, a `sinking point` aggregate the output of each node's output with softmax
    # noted that cg-alpha here should denote whether connected or not
    def __init__(self, *args, **kwargs):
        super(MacroSinkConnectDiffController, self).__init__(*args, **kwargs)

        if not self.full_init:
            self.cg_alphas = nn.ParameterList(
                [
                    nn.Parameter(
                        1e-3
                        * torch.randn(
                            sum([n - 1 for n in self.search_space.stage_node_nums])
                        )
                    )
                ]
            )
        else:
            self.cg_alphas = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.ones(
                            sum([n - 1 for n in self.search_space.stage_node_nums])
                        )
                    )
                ]
            )

        assert (
            self.use_sigmoid == False
        )  # sink-connecting should introduce competition in edges
        self.get_arch = GetArchMacroSinkConnect()
        self.stage_num_alphas = [n - 1 for n in self.search_space.stage_node_nums]
        self.to(self.device)  # move the newly generated cg_alphas to cuda

    # The only difference with MacroStageWiseDiffController's sample is that the arch is packed into `sink-connect-diff-rollout`
    def sample(self, n=1, batch_size=1):
        # if use progressive pruning
        if self.progressive_pruning_th is not None:
            self.progressive_pruning()

        width_arch, width_logits = self.sample_width(n=n, batch_size=batch_size)

        rollouts = []
        for i_sample in range(n):
            # op_weights.shape: [num_edges, [batch_size,] num_ops]
            # edge_norms.shape: [num_edges] do not have batch_size.
            op_weights_list = []
            edge_norms_list = []
            sampled_list = []
            logits_list = []

            for alphas in self.cg_alphas:
                splits = [i - 1 for i in self.search_space.stage_node_nums]
                op_weights = []
                sampleds = []
                for alpha in alphas.split(splits):
                    if (
                        self.force_uniform
                    ):  # cg_alpha parameters will not be in the graph
                        # NOTE: `force_uniform` config does not affects edge_norms (betas),
                        # if one wants a force_uniform search, keep `use_edge_normalization=False`
                        alpha = torch.zeros_like(alpha)

                    if batch_size > 1:
                        expanded_alpha = (
                            alpha.reshape([alpha.shape[0], 1, alpha.shape[1]])
                            .repeat([1, batch_size, 1])
                            .reshape([-1, alpha.shape[-1]])
                        )
                    else:
                        expanded_alpha = alpha

                    if self.use_prob:
                        sampled = F.softmax(
                            expanded_alpha / self.gumbel_temperature, dim=-1
                        )
                    elif self.use_sigmoid:
                        sampled = utils.relaxed_bernoulli_sample(
                            expanded_alpha, self.gumbel_temperature
                        )
                    else:
                        # gumbel sampling
                        sampled, _ = utils.gumbel_softmax(
                            expanded_alpha, self.gumbel_temperature, hard=False
                        )

                    if self.gumbel_hard:
                        op_weight = utils.straight_through(sampled)
                    else:
                        op_weight = sampled

                    if batch_size > 1:
                        sampled = sampled.reshape([-1, batch_size, op_weight.shape[-1]])
                        op_weight = op_weight.reshape(
                            [-1, batch_size, op_weight.shape[-1]]
                        )

                    op_weights.append(op_weight)
                    sampleds.append(sampled)

                op_weights = torch.cat(op_weights)
                sampleds = torch.cat(sampleds)

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampleds))
                logits_list.append(alphas)

                stage_conns = []
                split_op_weights = torch.split(op_weights, self.stage_num_alphas)

                for i_stage in range(self.search_space.stage_num):
                    stage_conn = self.get_arch.apply(
                        self.search_space,
                        split_op_weights[i_stage],
                        self.device,
                        i_stage,
                    )
                    stage_conns.append(stage_conn)

            rollouts.append(
                SinkConnectMacroDiffRollout(
                    arch=stage_conns,
                    sampled=sampled_list,
                    logits=logits_list,
                    width_arch=width_arch[i_sample],
                    width_logits=width_logits[i_sample],
                    search_space=self.search_space,
                )
            )

        return rollouts

    def __setstate__(self, state):
        super(MacroSinkConnectDiffController, self).__setstate__(state)
        self.get_arch = GetArchMacroSinkConnect()


class GetArchMicro(torch.autograd.Function):
    @staticmethod
    def forward(ctx, search_space, op_weights, device):
        empty_arch = torch.zeros(
            (
                search_space._num_nodes,
                search_space._num_nodes,
                search_space.num_op_choices,
            )
        ).to(device)
        empty_arch[search_space.idx] = op_weights
        ctx.save_for_backward(
            torch.as_tensor(op_weights), torch.as_tensor(search_space.idx)
        )
        return empty_arch

    @staticmethod
    def backward(ctx, grad_output):
        op_weights, idxes = ctx.saved_tensors
        op_weights_grad = grad_output[idxes[0], idxes[1]]
        return None, op_weights_grad, None


class MicroDenseDiffController(BaseController, nn.Module):
    NAME = "micro-dense-diff"

    SCHEDULABLE_ATTRS = ["gumbel_temperature", "entropy_coeff", "force_uniform"]

    def __init__(
        self,
        search_space,
        rollout_type,
        mode="eval",
        device="cuda",
        use_prob=False,
        gumbel_hard=False,
        gumbel_temperature=1.0,
        use_sigmoid=True,
        use_edge_normalization=False,
        entropy_coeff=0.01,
        max_grad_norm=None,
        force_uniform=False,
        full_init=False,
        progressive_pruning_th=None,
        multiprocess=False,
        schedule_cfg=None,
    ):
        super(MicroDenseDiffController, self).__init__(
            search_space, rollout_type, schedule_cfg=schedule_cfg
        )
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.use_prob = use_prob
        self.use_sigmoid = use_sigmoid
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature

        assert not (use_prob and use_sigmoid)
        # edge normalization
        self.use_edge_normalization = use_edge_normalization

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        self.full_init = full_init
        self.progressive_pruning_th = progressive_pruning_th

        self.multiprocess = multiprocess

        _num_init_nodes = self.search_space.num_init_nodes
        _num_edges_list = [
            sum(
                _num_init_nodes + i
                for i in range(self.search_space.get_num_steps(i_cg))
            )
            for i_cg in range(self.search_space.num_cell_groups)
        ]

        if not self.full_init:
            self.cg_alphas = nn.ParameterList(
                [
                    nn.Parameter(
                        1e-3
                        * torch.randn(
                            _num_edges,
                            len(self.search_space.cell_shared_primitives[i_cg]),
                        )
                    )  # shape: [num_edges, num_ops]
                    for i_cg, _num_edges in enumerate(_num_edges_list)
                ]
            )
        else:
            self.cg_alphas = nn.ParameterList(
                [
                    nn.Parameter(
                        1
                        * torch.ones(
                            _num_edges,
                            len(self.search_space.cell_shared_primitives[i_cg]),
                        )
                    )  # shape: [num_edges, num_ops]
                    for i_cg, _num_edges in enumerate(_num_edges_list)
                ]
            )

        if self.use_edge_normalization:
            raise NotImplementedError("MicroDenseController does not support edge-norm")
        else:
            self.cg_betas = None

        self.get_arch = GetArchMicro()

        self.to(self.device)

    def set_mode(self, mode):
        super(MicroDenseDiffController, self).set_mode(mode)
        if mode == "train":
            nn.Module.train(self)
        elif mode == "eval":
            nn.Module.eval(self)
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def set_device(self, device):
        self.device = device
        self.to(device)

    def progressive_pruning(self):
        for alpha in self.cg_alphas:
            # inpalce replace alphas that smaller than the pruning threshold, no grad
            alpha.data = alpha * (alpha.gt(self.progressive_pruning_th).float())

    def forward(self, n=1, batch_size=1):  # pylint: disable=arguments-differ
        return self.sample(n=n, batch_size=batch_size)

    def sample(self, n=1, batch_size=1):
        if self.progressive_pruning_th is not None:
            self.progressive_pruning()
        rollouts = []
        for _ in range(n):
            # op_weights.shape: [num_edges, [batch_size,] num_ops]
            # edge_norms.shape: [num_edges] do not have batch_size.
            op_weights_list = []
            edge_norms_list = []
            sampled_list = []
            logits_list = []

            for alphas in self.cg_alphas:
                if self.force_uniform:  # cg_alpha parameters will not be in the graph
                    # NOTE: `force_uniform` config does not affects edge_norms (betas),
                    # if one wants a force_uniform search, keep `use_edge_normalization=False`
                    alphas = torch.zeros_like(alphas)

                if batch_size > 1:
                    expanded_alpha = (
                        alphas.reshape([alphas.shape[0], 1, alphas.shape[1]])
                        .repeat([1, batch_size, 1])
                        .reshape([-1, alphas.shape[-1]])
                    )
                else:
                    expanded_alpha = alphas

                if self.use_prob:
                    # probability as sample
                    sampled = F.softmax(
                        expanded_alpha / self.gumbel_temperature, dim=-1
                    )
                elif self.use_sigmoid:
                    sampled = utils.relaxed_bernoulli_sample(
                        expanded_alpha, self.gumbel_temperature
                    )
                else:
                    # gumbel sampling
                    sampled, _ = utils.gumbel_softmax(
                        expanded_alpha, self.gumbel_temperature, hard=False
                    )

                if self.gumbel_hard:
                    op_weights = utils.straight_through(sampled)
                else:
                    op_weights = sampled

                if batch_size > 1:
                    sampled = sampled.reshape([-1, batch_size, op_weights.shape[-1]])
                    op_weights = op_weights.reshape(
                        [-1, batch_size, op_weights.shape[-1]]
                    )

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampled))
                # logits_list.append(utils.get_numpy(alphas))
                logits_list.append((alphas))

            if self.use_edge_normalization:
                raise NotImplementedError
            else:
                arch_list = []
                logits_arch_list = []
                for op_weights in op_weights_list:
                    arch = self.get_arch.apply(
                        self.search_space, op_weights, self.device
                    )
                    arch_list.append(arch)
                for logits in logits_list:
                    logits_arch = self.get_arch.apply(
                        self.search_space, logits, self.device
                    )
                    logits_arch_list.append(logits_arch)

            rollouts.append(
                DenseMicroDiffRollout(
                    arch_list,
                    sampled_list,
                    logits_list,
                    logits_arch_list,
                    search_space=self.search_space,
                )
            )
        return rollouts

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        """Load the parameters from disk."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    def _entropy_loss(self):
        if self.entropy_coeff > 0:
            probs = [F.softmax(alpha, dim=-1) for alpha in self.cg_alphas]
            return self.entropy_coeff * sum(
                -(torch.log(prob) * prob).sum() for prob in probs
            )
        return 0.0

    def gradient(self, loss, return_grads=True, zero_grads=True):
        raise NotImplementedError(
            "the grad function is implemented in the layer2diffcontroller.gradient()"
        )

    def step_current_gradient(self, optimizer):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step_gradient(self, gradients, optimizer):
        self.zero_grad()
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # clip the gradients
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def step(self, rollouts, optimizer, perf_name):  # very memory inefficient
        self.zero_grad()
        losses = [r.get_perf(perf_name) for r in rollouts]
        optimizer.step()
        [l.backward() for l in losses]
        return np.mean([l.detach().cpu().numpy() for l in losses])

    def __getstate__(self):
        state = super(MicroDenseDiffController, self).__getstate__().copy()
        del state["get_arch"]
        return state

    def __setstate__(self, state):
        super(MicroDenseDiffController, self).__setstate__(state)
        self.get_arch = GetArchMicro()

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        num = len(rollouts)
        logits_list = [
            [utils.get_numpy(logits) for logits in r.logits] for r in rollouts
        ]
        _ss = self.search_space
        if self.gumbel_hard:
            cg_logprobs = [0.0 for _ in range(_ss.num_cell_groups)]
        cg_entros = [0.0 for _ in range(_ss.num_cell_groups)]
        for rollout, logits in zip(rollouts, logits_list):
            for cg_idx, (vec, cg_logits) in enumerate(zip(rollout.arch, logits)):
                prob = utils.softmax(cg_logits)
                logprob = np.log(prob)
                if self.gumbel_hard:
                    inds = np.argmax(utils.get_numpy(vec), axis=-1)
                    cg_logprobs[cg_idx] += np.sum(logprob[range(len(inds)), inds])
                cg_entros[cg_idx] += -(prob * logprob).sum()

        # mean across rollouts
        if self.gumbel_hard:
            cg_logprobs = [s / num for s in cg_logprobs]
            total_logprob = sum(cg_logprobs)
            cg_logprobs_str = ",".join(["{:.2f}".format(n) for n in cg_logprobs])

        cg_entros = [s / num for s in cg_entros]
        total_entro = sum(cg_entros)
        cg_entro_str = ",".join(["{:.2f}".format(n) for n in cg_entros])

        if log:
            # maybe log the summary
            self.logger.info(
                "%s%d rollouts: %s ENTROPY: %2f (%s)",
                log_prefix,
                num,
                "-LOG_PROB: %.2f (%s) ;" % (-total_logprob, cg_logprobs_str)
                if self.gumbel_hard
                else "",
                total_entro,
                cg_entro_str,
            )
            if step is not None and not self.writer.is_none():
                if self.gumbel_hard:
                    self.writer.add_scalar("log_prob", total_logprob, step)
                self.writer.add_scalar("entropy", total_entro, step)

        stats = [
            (n + " ENTRO", entro) for n, entro in zip(_ss.cell_group_names, cg_entros)
        ]
        if self.gumbel_hard:
            stats += [
                (n + " LOGPROB", logprob)
                for n, logprob in zip(_ss.cell_group_names, cg_logprobs)
            ]
        return OrderedDict(stats)

    @classmethod
    def supported_rollout_types(cls):
        return ["micro-dense", "micro-dense-diff"]
