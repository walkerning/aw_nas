"""
Wrapper search space and rollout, corresponding to WrapperWeightsManager.
A WrapperRollout wraps the backbone and neck rollouts.
"""

import os
import re
import copy
import collections

import numpy as np

from aw_nas.common import SearchSpace, genotype_from_str
from aw_nas.rollout.base import BaseRollout


class WrapperSearchSpace(SearchSpace):
    NAME = "wrapper"

    def __init__(
        self,
        backbone_search_space_type="cnn",
        backbone_search_space_cfg={},
        backbone_rollout_type="discrete",
        neck_search_space_type=None,
        neck_search_space_cfg={},
        neck_rollout_type=None,
        schedule_cfg=None,
    ):
        super().__init__(schedule_cfg)

        self.backbone = SearchSpace.get_class_(backbone_search_space_type)(
            **backbone_search_space_cfg
        )
        if neck_search_space_type is not None:
            self.neck = SearchSpace.get_class_(neck_search_space_type)(
                **neck_search_space_cfg
            )
        else:
            self.neck = None

        self.backbone_rollout_type = backbone_rollout_type
        self.neck_rollout_type = neck_rollout_type

    def random_sample(self):
        backbone_r = self.backbone.random_sample()
        if self.neck is not None:
            neck_r = self.neck.random_sample()
        else:
            neck_r = None
        return WrapperRollout(backbone_r, neck_r, self)

    def genotype(self, arch):
        """Convert arch (controller representation) to genotype (semantic representation)"""
        backbone_r, neck_r = arch
        return (backbone_r.genotype, neck_r.genotype if neck_r is not None else None)

    def rollout_from_genotype(self, genotype):
        """Convert genotype (semantic representation) to arch (controller representation)"""
        backbone_g, neck_g = genotype
        backbone_g = self.backbone.rollout_from_genotype(backbone_g)
        if neck_g is not None:
            neck_g = self.neck.rollout_from_genotype(neck_g)
        return WrapperRollout(backbone_g, neck_g, self)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        backbone_g, neck_g = genotypes
        fnames = []
        fnames += self.backbone.plot_arch(
            backbone_g, os.path.join(filename, "backbone"), label, **kwargs
        )
        if self.neck is not None:
            fnames += self.neck.plot_arch(
                neck_g, os.path.join(filename, "neck"), label, **kwargs
            )
        return fnames

    def distance(self, arch1, arch2):
        raise NotImplementedError()

    @classmethod
    def supported_rollout_types(cls):
        return ["wrapper"]

    def mutate(self, rollout, **mutate_kwargs):
        if self.neck is not None:
            mutate_backbone_prob = mutate_kwargs.get("mutate_backbone_prob", 0.5)
        else:
            # can only mutate backbone rollout
            mutate_backbone_prob = 1.0
        new_rollout = copy.deepcopy(rollout)
        if np.random.random() < mutate_backbone_prob:
            # mutate in backbone search space
            new_rollout.backbone = self.backbone.mutate(new_rollout.backbone)
        else:
            # mutate in neck search space
            new_rollout.neck = self.neck.mutate(new_rollout.neck)
        return new_rollout

    def genotype_from_str(self, genotype_str):
        match = re.search(
            r"\((.+Genotype\(.+\)), (.+Genotype\(.+\)|None)\)", genotype_str
        )
        b_genotype_str = match.group(1)
        n_genotype_str = match.group(2)
        b_genotype = genotype_from_str(b_genotype_str, self.backbone)
        if n_genotype_str == "None":
            n_genotype = None
        else:
            n_genotype = genotype_from_str(n_genotype_str, self.neck)
        return (b_genotype, n_genotype)


class WrapperRollout(BaseRollout):
    NAME = "wrapper"
    supported_components = [
        ("trainer", "simple"),
        ("evaluator", "mepa"),
        ("evaluator", "discrete_shared_weights"),
        ("evaluator", "differentiable_shared_weights"),
    ]

    def __init__(
        self, backbone_rollout, neck_rollout, search_space, candidate_net=None
    ):
        super().__init__()

        self.backbone = backbone_rollout
        self.neck = neck_rollout
        self.search_space = search_space
        self.candidate_net = candidate_net
        self._perf = collections.OrderedDict()
        self._genotype = None

    def set_candidate_net(self, c_net):
        self.candidate_net = c_net

    @property
    def genotype(self):
        if self._genotype is None:
            self._genotype = self.search_space.genotype((self.backbone, self.neck))
        return self._genotype

    def plot_arch(self, filename, label="", edge_labels=None, plot_format="pdf"):
        return self.search_space.plot_arch(
            self.genotype,
            filename=filename,
            label=label,
            edge_labels=edge_labels,
            plot_format=plot_format,
        )

    def plot_template(self, filename, label="", edge_labels=None, plot_format="pdf"):
        return self.search_space.plot_template(
            self.genotype,
            filename=filename,
            label=label,
            edge_labels=edge_labels,
            plot_format=plot_format,
        )

    def __eq__(self, other):
        return self.backbone == other.backbone and self.neck == other.neck


class GermWrapperSearchSpace(WrapperSearchSpace):
    NAME = "germ_wrapper"

    def random_sample(self):
        rollout = super().random_sample()
        br, nr = rollout.backbone, rollout.neck
        if nr is not None:
            duplicate_r = {k: v for k, v in br.arch.items() if k in nr.arch}
            nr.arch.update(duplicate_r)
        return rollout
