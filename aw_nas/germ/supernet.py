import copy
import contextlib
from functools import partial
from collections import defaultdict, OrderedDict

import yaml
import torch
from torch import nn

from aw_nas.weights_manager.base import CandidateNet
from aw_nas.weights_manager.wrapper import BaseBackboneWeightsManager

from aw_nas.base import Component
from aw_nas.utils.registry import RegistryMeta
from aw_nas.utils import expect
from aw_nas.germ.germ import SearchableBlock, finalize_rollout
from aw_nas.germ.decisions import BaseDecision, NonleafDecision


class SearchableContext(object):
    def __init__(self):
        self.rollout = None

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
        raise AttributeError("{}.{} is invalid.".format(self.__class__.__name__, name))


class GermSuperNet(Component, nn.Module):
    REGISTRY = "germ_super_net"

    def __init__(self, search_space, schedule_cfg=None):
        super().__init__(schedule_cfg)
        nn.Module.__init__(self)

        self._begin_searchable_called = False
        self.ctx = SearchableContext()
        # call `with self.begin_searchable()`,
        # and initialize modules in the context

        self.search_space = search_space

    def parse_searchable(self):
        supernet = self
        if not hasattr(supernet, "_searchable_decisions"):
            supernet._searchable_decisions = OrderedDict()
            supernet._nonleaf_decisions = OrderedDict()
            supernet._all_decisions = OrderedDict()
            supernet._blockid_to_dimension2decision = defaultdict(OrderedDict)
        leaf_decisions = supernet._searchable_decisions
        nonleaf_decisions = supernet._nonleaf_decisions
        all_decisions = supernet._all_decisions
        blockid_to_dimension2decision = supernet._blockid_to_dimension2decision

        for name, mod in supernet.named_modules():
            if isinstance(mod, SearchableBlock):
                mod.block_id = name
                for d_name, decision in mod.named_decisions(avoid_repeat=False, recurse=False):
                    if decision not in all_decisions:
                        if isinstance(decision, NonleafDecision):
                            sub_dict = nonleaf_decisions
                        else:
                            sub_dict = leaf_decisions
                        if (
                            hasattr(decision, "decision_id")
                            and decision.decision_id is not None
                        ):
                            abs_name = decision.decision_id
                            if decision not in all_decisions:
                                all_decisions[decision] = sub_dict[decision] = abs_name
                        else:
                            abs_name = ((name + ".") if name else "") + d_name
                            decision.decision_id = sub_dict[decision] \
                                                = all_decisions[decision] = abs_name
                        if isinstance(decision, NonleafDecision):
                            decs = decision.set_children_id()
                            for dec in decs:
                                if isinstance(dec, NonleafDecision):
                                    sub_dict = nonleaf_decisions
                                else:
                                    sub_dict = leaf_decisions
                                sub_dict[dec] = all_decisions[dec] = dec.decision_id
                    else:
                        # use existing decision id (abs name)
                        abs_name = all_decisions[decision]
                    blockid_to_dimension2decision[name][d_name] = abs_name
        supernet._begin_searchable_called = True

        ss_cfg = self.generate_search_space_ref()
        self.search_space.set_cfg(ss_cfg)

    def named_decisions(self, prefix="", recurse=True, avoid_repeat=True):
        # By default, not recursive
        def _get_named_decisions(mod):
            return mod._decisions.items()

        memo = set()
        mod_list = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mod_name, mod in mod_list:
            if isinstance(mod, SearchableBlock):
                for d_id, d_obj in _get_named_decisions(mod):
                    if d_obj is None or (avoid_repeat and d_obj in memo):
                        continue
                    memo.add(d_obj)
                    d_name = mod_name + ("." if mod_name else "") + d_id
                    yield d_name, d_obj

    def forward_rollout(self, rollout, *args, **kwargs):
        self.ctx.rollout = rollout
        return self(*args, **kwargs)

    """
    @abc.abstractmethod
    def extract_features(self, *args, **kwargs):
        pass

    def extract_features_rollout(self, rollout, *args, **kwargs):
        self.ctx.rollout = rollout
        return self.extract_features(*args, **kwargs)
    """

    def begin_searchable(self):
        return self.ctx.begin_searchable(self)

    def generate_search_space_cfg_to_file(self, fname):
        cfg = self.generate_search_space_cfg()
        with open(fname, "w") as w_f:
            yaml.dump(cfg, w_f)

    def generate_search_space_cfg(self):
        expect(
            self._begin_searchable_called, "self.begin_searchable() is not called yet"
        )
        return {
            "decisions": {
                decision_id: (decision.NAME, str(decision))
                for decision, decision_id in self._searchable_decisions.items()
            },
            "nonleaf_decisions": {
                decision_id: (decision.NAME, str(decision))
                for decision, decision_id in self._nonleaf_decisions.items()
            },
            "blocks": self._blockid_to_dimension2decision,
        }

    def generate_search_space_ref(self):
        expect(
            self._begin_searchable_called, "self.begin_searchable() is not called yet"
        )
        return {
            "decisions": {
                decision_id: (decision.NAME, decision)
                for decision, decision_id in self._searchable_decisions.items()
            },
            "nonleaf_decisions": {
                decision_id: (decision.NAME, decision)
                for decision, decision_id in self._nonleaf_decisions.items()
            },
            "blocks": self._blockid_to_dimension2decision,
        }

    def finalize_rollout(self, rollout):
        """
        The default implementation: Make a copy of the germ supernet.
        Then, call finalize_rollout of all SearchableBlocks.
        """
        self.ctx.rollout = rollout
        final_mod = copy.deepcopy(self)
        final_mod = finalize_rollout(final_mod, rollout)
        final_mod.ctx.rollout = None
        return final_mod

    def on_epoch_start(self, epoch):
        for _, decs in self.named_decisions(recurse=True):
            decs.on_epoch_start(epoch)

class GermWeightsManager(BaseBackboneWeightsManager, nn.Module):
    NAME = "germ"

    def __init__(
        self,
        search_space,
        device,
        rollout_type="germ",
        # initialize from registry
        germ_supernet_type=None,
        germ_supernet_cfg=None,
        # support load a code snippet
        germ_def_file=None,
        candidate_eval_no_grad=True,
        max_grad_norm=None,
        schedule_cfg=None,
    ):
        super().__init__(search_space, device, rollout_type, schedule_cfg)
        nn.Module.__init__(self)
        self.germ_def_file = germ_def_file

        if germ_def_file is not None:
            # python 3
            germ_def_module = {}
            with open(germ_def_file, "rb") as source_file:
                code = compile(source_file.read(), germ_def_file, "exec")
                exec(code, germ_def_module)

        self.candidate_eval_no_grad = candidate_eval_no_grad
        self.max_grad_norm = max_grad_norm
        self.super_net = GermSuperNet.get_class_(germ_supernet_type)(
            search_space, **(germ_supernet_cfg or {})
        )

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

    def forward(self, *args, **kwargs):
        return self.super_net(*args, **kwargs)

    def extract_features(self, inputs, rollout=None):
        return self.super_net.extract_features_rollout(rollout, inputs)

    def get_feature_channel_num(self, feature_levels=None):
        return self.super_net.get_feature_channel_num(feature_levels)

    def finalize(self, rollout):
        self.super_net = self.super_net.finalize_rollout(rollout)
        return self

    def assemble_candidate(self, rollout, **kwargs):
        return GermCandidateNet(self, rollout, self.candidate_eval_no_grad)

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
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
        torch.save({"epoch": self.epoch, "state_dict": self.state_dict()}, path)

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

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
        self.super_net.on_epoch_start(epoch)


class GermCandidateNet(CandidateNet):
    def __init__(self, weights_manager, rollout, eval_no_grad=True):
        super(GermCandidateNet, self).__init__(eval_no_grad=eval_no_grad)
        self.super_net = weights_manager
        self._device = self.super_net.device
        self.rollout = rollout

    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.super_net.forward_rollout(self.rollout, inputs)

    def _forward_with_params(self, *args, **kwargs):  # pylint: disable=arguments-differ
        raise NotImplementedError()

    def get_device(self):
        return self._device
