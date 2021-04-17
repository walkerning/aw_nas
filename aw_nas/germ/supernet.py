import copy
import contextlib
from functools import partial
from collections import defaultdict

import yaml
import torch
from torch import nn

from aw_nas.base import Component
from aw_nas.utils.registry import RegistryMeta
from aw_nas.utils import expect
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.germ.germ import SearchableBlock, finalize_rollout


class SearchableContext(object):
    @contextlib.contextmanager
    def begin_searchable(self, supernet):
        yield self
        supernet.parse_searchable()

    def __getattr__(self, name):
        dict_ = RegistryMeta.all_classes("searchable_block")
        dict_cls_names = {v.__name__: v for v in dict_.values()}
        if name in dict_:
            return partial(RegistryMeta.get_classes("searchable_block", name), ctx=self)
        elif name in dict_cls_names:
            return partial(dict_cls_names[name], ctx=self)
        raise AttributeError('{}.{} is invalid.'.format(self.__class__.__name__, name))


class GermSuperNet(Component, nn.Module):
    REGISTRY = "germ_super_net"

    def __init__(self, schedule_cfg=None):
        super().__init__(schedule_cfg)
        nn.Module.__init__(self)

        self._begin_searchable_called = False
        self.ctx = SearchableContext()
        # call `with self.begin_searchable()`,
        # and initialize modules in the context

    def parse_searchable(self):
        supernet = self
        if not hasattr(supernet, "_searchable_decisions"):
            supernet._searchable_decisions = {}
            supernet._blockid_to_dimension2decision = defaultdict(dict)
        all_decisions = supernet._searchable_decisions
        blockid_to_dimension2decision = supernet._blockid_to_dimension2decision

        for name, mod in supernet.named_modules():
            if isinstance(mod, SearchableBlock):
                mod.block_id = name
                for d_name, decision in mod.named_decisions(avoid_repeat=False):
                    if decision not in all_decisions:
                        abs_name = name + "." + d_name
                        decision.decision_id = all_decisions[decision] = abs_name
                    else:
                        # use existing decision id (abs name)
                        abs_name = all_decisions[decision]
                    blockid_to_dimension2decision[name][d_name] = abs_name
        supernet._begin_searchable_called = True

    def named_decisions(self, prefix="", recurse=False):
        # By default, not recursive
        _get_named_decisions = lambda mod: mod._decisions.items()
        memo = set()
        mod_list = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mod_name, mod in mod_list:
            if isinstance(mod, SearchableBlock):
                for d_id, d_obj in _get_named_decisions(mod):
                    if d_obj is None or d_obj in memo:
                        continue
                    memo.add(d_obj)
                    d_name = mod_name + ("." if mod_name else "") + d_id
                    yield d_name, d_obj

    def forward_rollout(self, rollout, *args, **kwargs):
        self.ctx.rollout = rollout
        return self(*args, **kwargs)

    def begin_searchable(self):
        return self.ctx.begin_searchable(self)

    def generate_search_space_cfg_to_file(self, fname):
        cfg = self.generate_search_space_cfg()
        with open(fname, "w") as w_f:
            yaml.dump(cfg, w_f)

    def generate_search_space_cfg(self):
        expect(self._begin_searchable_called,
               "self.begin_searchable() is not called yet")
        return {
            "decisions": {decision_id: (decision.NAME, str(decision))
                          for decision, decision_id in self._searchable_decisions.items()},
            "blocks": self._blockid_to_dimension2decision
        }

    def finalize_rollout(self, rollout):
        """
        The default implementation: Make a copy of the germ supernet.
        Then, call finalize_rollout of all SearchableBlocks.
        """
        final_mod = copy.deepcopy(self)
        return finalize_rollout(final_mod, rollout)


class GermWeightsManager(BaseWeightsManager, nn.Module):
    NAME = "germ"

    def __init__(self, search_space, device, rollout_type="germ",
                 # initalize from registry
                 germ_supernet_type=None,
                 germ_supernet_cfg=None,
                 # support load a code snippet
                 germ_def_file=None,
                 max_grad_norm=None,
                 schedule_cfg=None):
        super().__init__(search_space, device, rollout_type, schedule_cfg)
        nn.Module.__init__(self)
        self.germ_def_file = germ_def_file

        if germ_def_file is not None:
            # python 3
            self.germ_def_module = {}
            with open(germ_def_file, "rb") as source_file:
                code = compile(source_file.read(), germ_def_file, "exec")
                exec(code, self.germ_def_module)

        self.max_grad_norm = max_grad_norm
        self.super_net = GermSuperNet.get_class_(germ_supernet_type)(**(germ_supernet_cfg or {}))

        self.to(self.device)
        # hook flops calculation
        self.set_hook()
        self._flops_calculated = False
        self.total_flops = 0

    # ---- APIs ----
    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward_rollout(self, rollout, *args, **kwargs):
        return self.super_net.forward_rollout(rollout, *args, **kwargs)

    def assemble_candidate(self, rollout):
        return GermCandidateNet(self, rollout)

    def step(self, gradients, optimizer):
        self.zero_grad() # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        if self.max_grad_norm is not None:
            # clip the gradients
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # apply the gradients
        optimizer.step()

    def step_current_gradients(self, optimizer):
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        optimizer.step()

    def save(self, path):
        torch.save({"epoch": self.epoch,
                    "state_dict": self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    @classmethod
    def supported_rollout_types(cls):
        return ["germ"]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    # ---- flops hook ----
    def reset_flops(self):
        self._flops_calculated = False
        self.total_flops = 0

    def set_hook(self):
        for _, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += (
                    inputs[0].size(1)
                    * outputs.size(1)
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * inputs[0].size(2)
                    * inputs[0].size(3)
                    / (module.stride[0] * module.stride[1] * module.groups)
                )
            elif isinstance(module, nn.Linear):
                self.total_flops += inputs[0].size(1) * outputs.size(1)
        else:
            pass


class GermCandidateNet(CandidateNet):
    def __init__(self, weights_manager, rollout, eval_no_grad=True):
        super(GermCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.weights_manager = weights_manager
        self._device = self.weights_manager.device
        self.rollout = rollout

    def forward(self, inputs): #pylint: disable=arguments-differ
        return self.weights_manager.forward_rollout(self.rollout, inputs)

    def _forward_with_params(self, *args, **kwargs): #pylint: disable=arguments-differ
        raise NotImplementedError()

    def get_device(self):
        return self._device
