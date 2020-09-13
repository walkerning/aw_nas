"""
2-layer controller.
"""

# from awnas import utils
from aw_nas import utils, assert_rollout_type
from aw_nas.controller.base import BaseController
from aw_nas.btcs.layer2.search_space import Layer2Rollout, DenseMicroRollout, StagewiseMacroRollout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Layer2Optimizer(optim.Optimizer):
    # def __init__(self, params, opt_type=["SGD"]*2, lr=[0.1]*2, momentum=[0.9]*2, betas=[0.9, 0.999]*2, weight_decay=[5e-4]*2, **opt_cfgs):
    def __init__(self, params, **opt_cfg):
        super(Layer2Optimizer, self).__init__([torch.tensor([])], defaults={}) 
        macro_opt_type = opt_cfg["macro"].pop("type")
        micro_opt_type = opt_cfg["micro"].pop("type")
        self.macro_optimizer=getattr(optim, macro_opt_type)(params[0],**opt_cfg["macro"])
        self.micro_optimizer=getattr(optim, micro_opt_type)(params[1],**opt_cfg["micro"]) 

    def step(self):
        self.macro_optimizer.step()
        self.micro_optimizer.step()

torch.optim.Layer2 = Layer2Optimizer # add patch the torch optim

class Layer2Controller(BaseController, nn.Module):
    NAME = "layer2"

    def __init__(self, search_space, rollout_type, mode="eval",device="cuda",
                 macro_controller_type="random_sample",
                 macro_controller_cfg={},
                 micro_controller_type="random_sample",
                 micro_controller_cfg={},
                 schedule_cfg=None):
        super(Layer2Controller, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)

        nn.Module.__init__(self)

        self.search_space=search_space
        self.rollout_type=rollout_type
        self.device=device   # FIXME: how 2 feed device in
        self.to(self.device)

        # the macro/micro controllers
        if macro_controller_type == "macro-stagewise-diff":
            self.macro_controller = MacroStagewiseDiffController(
                                        self.search_space.macro_search_space,
                                        macro_controller_type,
                                        **macro_controller_cfg,
            )
        else:
            raise NotImplementedError()

        if micro_controller_type == "micro-dense-diff":
            self.micro_controller = MicroDenseDiffController(
                                        self.search_space.micro_search_space,
                                        micro_controller_type,
                                        **micro_controller_cfg,
            )
        else:
            raise NotImplementedError()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_mode(self, mode):
        super(MacroStagewiseDiffController, self).set_mode(mode)
        if mode == "train":
            nn.Module.train(self)
        elif mode == "eval":
            nn.Module.eval(self)
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def parameters(self):
        return [self.macro_controller.parameters(), self.micro_controller.parameters()]

    def forward(self, n=1):
        return self.sample(n=n)

    def sample(self, n=1, batch_size=1):
        rollouts = []
        macro_rollouts = self.macro_controller.sample(n=n,batch_size=batch_size)
        micro_rollouts = self.micro_controller.sample(n=n,batch_size=batch_size)
        for i in range(n):
            rollouts.append(Layer2Rollout(macro_rollouts[i],micro_rollouts[i], self.search_space))
        return rollouts

    def gradient(self, loss, return_grads=True, zero_grads=True):
        macro_loss = self.macro_controller.gradient(loss, return_grads, zero_grads)
        micro_loss = self.micro_controller.gradient(loss, return_grads, zero_grads)
        return macro_loss, micro_loss

    def step_current_gradient(self, optimizer):
        #FIXME: maybe  optimizer being a list of [macro_opt, micro_opt]
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def step_gradient(self, gradients, optimizer):
        #FIXME: maybe  optimizer being a list of [macro_grad, micro_grad]
        self.macro_controller.step_gradient(gradients[0], optimizer.macro_optimizer)
        self.micro_controller.step_gradient(gradients[1], optimizer.micro_optimizer)

    def step(self, rollouts, optimizer, perf_name):
        macro_rollouts = [r.macro_rollout for r in rollouts]
        micro_rollouts = [r.micro_rollout for r in rollouts]
        macro_loss = self.macro_controller.step(macro_rollouts, optimizer.macro_optimizer, perf_name)
        micro_loss = self.micro_controller.step(micro_rollouts, optimizer.micro_optimizer, perf_name)
        return macro_loss, micro_loss

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        macro_rollouts = [r.macro_rollout for r in rollouts]
        micro_rollouts = [r.micro_rollout for r in rollouts]
        self.macro_controller.summary(macro_rollout, log=log, log_prefix=log_prefix, step=None)
        self.micro_controller.summary(micro_rollout, log=log, log_prefix=log_prefix, step=None)

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)
        self.logger.info("Saved controller network to %s", path)

    def load(self, path):
        """Load the parameters from disk."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])
        self.logger.info("Loaded controller network from %s", path)

    @classmethod
    def supported_rollout_types(cls):
        return ["layer2"]

# TODO: macro controller
class GetArchMacro(torch.autograd.Function):
    @staticmethod
    def forward(ctx, search_space, op_weights, device, i_stage, interval):
        begin, end = interval
        stage_conn = torch.zeros((search_space.stage_node_nums[i_stage],
                               search_space.stage_node_nums[i_stage])).to(device)
        stage_conn[search_space.idxes[i_stage]] = \
            op_weights
        ctx.save_for_backward(torch.tensor(op_weights), torch.tensor(search_space.idxes[i_stage]), torch.tensor([begin, end]))

        return stage_conn

    @staticmethod
    def backward(ctx, grad_output):
        op_weights, idxes, interval = ctx.saved_tensors
        begin, end = interval
        op_weights_grad = grad_output[idxes[0], idxes[1]]
        return None, op_weights_grad, None, None, None


class MacroStagewiseDiffController(BaseController, nn.Module):
    NAME = "macro-stagewise-diff"

    SCHEDULABLE_ATTRS = [
        "gumbel_temperature",
        "entropy_coeff",
        "force_uniform"
    ]

    def __init__(self, search_space, rollout_type, mode="eval",device="cuda",
                 use_prob=False, gumbel_hard=False, gumbel_temperature=1.0,
                 use_edge_normalization=False,
                 entropy_coeff=0.01, max_grad_norm=None, force_uniform=False,
                 schedule_cfg=None):
        super(MacroStagewiseDiffController, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.use_prob = use_prob
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature

        # edge normalization
        self.use_edge_normalization = use_edge_normalization

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        # generate parameters
        self.cg_alphas = nn.ParameterList([nn.Parameter(1e-3*torch.randn(sum(self.search_space.num_possible_edges)))])

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

    def forward(self, n=1):  # pylint: disable=arguments-differ
        return self.sample(n=n)

    def sample(self, n=1, batch_size=1):
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
                    expanded_alpha = alphas.reshape([alphas.shape[0], 1, alphas.shape[1]]) \
                        .repeat([1, batch_size, 1]) \
                        .reshape([-1, alphas.shape[-1]])
                else:
                    expanded_alpha = alphas

                if self.use_prob:
                    # probability as sample
                    sampled = F.softmax(expanded_alpha / self.gumbel_temperature, dim=-1)
                else:
                    # gumbel sampling
                    sampled, _ = utils.gumbel_softmax(expanded_alpha, self.gumbel_temperature,
                                                      hard=False)

                if self.gumbel_hard:
                    op_weights = utils.straight_through(sampled)
                else:
                    op_weights = sampled

                if batch_size > 1:
                    sampled = sampled.reshape([-1, batch_size, op_weights.shape[-1]])
                    op_weights = op_weights.reshape([-1, batch_size, op_weights.shape[-1]])

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampled))
                logits_list.append(utils.get_numpy(alphas))

                # TODO: ck if grad ok
                stage_conns = []
                begin=0
                end=self.search_space.num_possible_edges[0]

                for i_stage in range(self.search_space.stage_num):
                    stage_conn = self.get_arch.apply(self.search_space, op_weights[begin:end], self.device, i_stage, [begin, end])
                    begin += self.search_space.num_possible_edges[i_stage]
                    end = end + self.search_space.num_possible_edges[i_stage+1] if i_stage+1 < self.search_space.stage_num else end # FIXME: Dirty!
                    stage_conns.append(stage_conn)

            rollouts.append(StagewiseMacroRollout(stage_conns, search_space=self.search_space))

        return rollouts


    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)
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
            return self.entropy_coeff * sum(-(torch.log(prob) * prob).sum() for prob in probs)
        return 0.

    def gradient(self, loss, return_grads=True, zero_grads=True):
        if zero_grads:
            self.zero_grad()
        _loss = loss + self._entropy_loss()
        _loss.backward()
        if return_grads:
            return utils.get_numpy(_loss), [(k, v.grad.clone()) for k, v in self.named_parameters()]
        return utils.get_numpy(_loss)

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

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        num = len(rollouts)
        logits_list = [[utils.get_numpy(logits) for logits in r.logits] for r in rollouts]
        _ss = self.search_space
        if self.gumbel_hard:
            cg_logprobs = [0. for _ in range(_ss.num_cell_groups)]
        cg_entros = [0. for _ in range(_ss.num_cell_groups)]
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
            self.logger.info("%s%d rollouts: %s ENTROPY: %2f (%s)",
                             log_prefix, num,
                             "-LOG_PROB: %.2f (%s) ;" % (-total_logprob, cg_logprobs_str) \
                                 if self.gumbel_hard else "",
                             total_entro, cg_entro_str)
            if step is not None and not self.writer.is_none():
                if self.gumbel_hard:
                    self.writer.add_scalar("log_prob", total_logprob, step)
                self.writer.add_scalar("entropy", total_entro, step)

        stats = [(n + " ENTRO", entro) for n, entro in zip(_ss.cell_group_names, cg_entros)]
        if self.gumbel_hard:
            stats += [(n + " LOGPROB", logprob) for n, logprob in \
                      zip(_ss.cell_group_names, cg_logprobs)]
        return OrderedDict(stats)

    @classmethod
    def supported_rollout_types(cls):
        return ["macro-stagewise","macro-stagewise-diff"]


# TODO: micro controller
class GetArchMicro(torch.autograd.Function):
    @staticmethod
    def forward(ctx, search_space, op_weights, device):
        empty_arch = torch.zeros((search_space._num_nodes, search_space._num_nodes, search_space.num_op_choices)).to(device)
        empty_arch[search_space.idx] = op_weights
        ctx.save_for_backward(torch.tensor(op_weights), torch.tensor(search_space.idx))
        return empty_arch

    @staticmethod
    def backward(ctx, grad_output):
        op_weights, idxes = ctx.saved_tensors
        op_weights_grad = grad_output[idxes[0], idxes[1]]
        return None, op_weights_grad, None


class MicroDenseDiffController(BaseController, nn.Module):
    NAME = "micro-dense-diff"

    SCHEDULABLE_ATTRS = [
        "gumbel_temperature",
        "entropy_coeff",
        "force_uniform"
    ]

    def __init__(self, search_space, rollout_type, mode="eval",device="cuda",
                 use_prob=False, gumbel_hard=False, gumbel_temperature=1.0,
                 use_edge_normalization=False,
                 entropy_coeff=0.01, max_grad_norm=None, force_uniform=False,
                 schedule_cfg=None):
        super(MicroDenseDiffController, self).__init__(search_space, rollout_type, schedule_cfg=schedule_cfg)
        nn.Module.__init__(self)

        self.device = device

        # sampling
        self.use_prob = use_prob
        self.gumbel_hard = gumbel_hard
        self.gumbel_temperature = gumbel_temperature

        # edge normalization
        self.use_edge_normalization = use_edge_normalization

        # training
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.force_uniform = force_uniform

        _num_init_nodes = self.search_space.num_init_nodes
        _num_edges_list = [
            sum(
                _num_init_nodes + i
                for i in range(self.search_space.get_num_steps(i_cg))
            )
            for i_cg in range(self.search_space.num_cell_groups)
        ]

        self.cg_alphas = nn.ParameterList([
            nn.Parameter(
                1e-3 * torch.randn(
                    _num_edges, len(self.search_space.cell_shared_primitives[i_cg])
                )
            )  # shape: [num_edges, num_ops]
            for i_cg, _num_edges in enumerate(_num_edges_list)
        ])

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

    def forward(self, n=1):  # pylint: disable=arguments-differ
        return self.sample(n=n)

    def sample(self, n=1, batch_size=1):
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
                    expanded_alpha = alphas.reshape([alphas.shape[0], 1, alphas.shape[1]]) \
                        .repeat([1, batch_size, 1]) \
                        .reshape([-1, alphas.shape[-1]])
                else:
                    expanded_alpha = alphas

                if self.use_prob:
                    # probability as sample
                    sampled = F.softmax(expanded_alpha / self.gumbel_temperature, dim=-1)
                else:
                    # gumbel sampling
                    sampled, _ = utils.gumbel_softmax(expanded_alpha, self.gumbel_temperature,
                                                      hard=False)

                if self.gumbel_hard:
                    op_weights = utils.straight_through(sampled)
                else:
                    op_weights = sampled

                if batch_size > 1:
                    sampled = sampled.reshape([-1, batch_size, op_weights.shape[-1]])
                    op_weights = op_weights.reshape([-1, batch_size, op_weights.shape[-1]])

                op_weights_list.append(op_weights)
                sampled_list.append(utils.get_numpy(sampled))
                logits_list.append(utils.get_numpy(alphas))

            if self.use_edge_normalization:
                for i_cg, betas in enumerate(self.cg_betas):
                    # eg: for 2 init_nodes and 3 steps, this is [2, 3, 4]
                    num_inputs_on_nodes = np.arange(self.search_space.get_num_steps(i_cg)) \
                                          + self.search_space.num_init_nodes
                    edge_norms = []
                    for i_node, num_inputs_on_node in enumerate(num_inputs_on_nodes):
                        # eg: for node_0, it has edge_{0, 1} as inputs, there for start=0, end=2
                        start = num_inputs_on_nodes[i_node - 1] if i_node > 0 else 0
                        end = start + num_inputs_on_node

                        edge_norms.append(F.softmax(betas[start:end], dim=0))

                    edge_norms_list.append(torch.cat(edge_norms))
                arch_list = [
                    DartsArch(op_weights=op_weights, edge_norms=edge_norms)
                    for op_weights, edge_norms in zip(op_weights_list, edge_norms_list)
                ]
            else:
                arch_list = []
                # TODO: maybe grad error?
                for op_weights in op_weights_list:
                    arch = self.get_arch.apply(self.search_space, op_weights, self.device)
                    arch_list.append(arch)

            rollouts.append(DenseMicroRollout(arch_list, search_space=self.search_space))
        return rollouts

    def save(self, path):
        """Save the parameters to disk."""
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)
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
            return self.entropy_coeff * sum(-(torch.log(prob) * prob).sum() for prob in probs)
        return 0.

    def gradient(self, loss, return_grads=True, zero_grads=True):
        if zero_grads:
            self.zero_grad()
        _loss = loss + self._entropy_loss()
        _loss.backward()
        if return_grads:
            return utils.get_numpy(_loss), [(k, v.grad.clone()) for k, v in self.named_parameters()]
        return utils.get_numpy(_loss)

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

    def summary(self, rollouts, log=False, log_prefix="", step=None):
        num = len(rollouts)
        logits_list = [[utils.get_numpy(logits) for logits in r.logits] for r in rollouts]
        _ss = self.search_space
        if self.gumbel_hard:
            cg_logprobs = [0. for _ in range(_ss.num_cell_groups)]
        cg_entros = [0. for _ in range(_ss.num_cell_groups)]
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
            self.logger.info("%s%d rollouts: %s ENTROPY: %2f (%s)",
                             log_prefix, num,
                             "-LOG_PROB: %.2f (%s) ;" % (-total_logprob, cg_logprobs_str) \
                                 if self.gumbel_hard else "",
                             total_entro, cg_entro_str)
            if step is not None and not self.writer.is_none():
                if self.gumbel_hard:
                    self.writer.add_scalar("log_prob", total_logprob, step)
                self.writer.add_scalar("entropy", total_entro, step)

        stats = [(n + " ENTRO", entro) for n, entro in zip(_ss.cell_group_names, cg_entros)]
        if self.gumbel_hard:
            stats += [(n + " LOGPROB", logprob) for n, logprob in \
                      zip(_ss.cell_group_names, cg_logprobs)]
        return OrderedDict(stats)

    @classmethod
    def supported_rollout_types(cls):
        return ["micro-dense", "micro-dense-diff"]
