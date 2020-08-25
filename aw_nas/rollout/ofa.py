"""
Definitions of mnasnet OFA rollout and search space.
"""
import copy

import numpy as np

from aw_nas import utils
from aw_nas.rollout.base import Rollout
from aw_nas.common import SearchSpace, genotype_from_str


class MNasNetOFASearchSpace(SearchSpace):
    NAME = "ofa"
    SCHEDULABLE_ATTRS = ["width_choice", "depth_choice", "kernel_choice"]

    def __init__(
        self,
        width_choice=(4, 5, 6),
        depth_choice=(4, 5, 6),
        kernel_choice=(3, 5, 7),
        num_cell_groups=[1, 4, 4, 4, 4, 4],
        expansions=[1, 6, 6, 6, 6, 6],
        schedule_cfg=None,
    ):
        super(MNasNetOFASearchSpace, self).__init__(schedule_cfg)
        self.genotype_type_name = "MnasnetOFAGenotype"

        self.num_cell_groups = num_cell_groups
        self.expansions = expansions

        self.block_names = sum(
            [["cell_{}".format(i) for i in range(len(num_cell_groups))]]
            + [
                [
                    "cell_{}_block_{}".format(i, j)
                    for j in range(self.num_cell_groups[i])
                ]
                for i in range(len(num_cell_groups))
            ],
            [],
        )

        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, []
        )
        self.width_choice = width_choice
        self.depth_choice = depth_choice
        self.kernel_choice = kernel_choice

    def __getstate__(self):
        state = super(MNasNetOFASearchSpace, self).__getstate__().copy()
        del state["genotype_type"]
        return state

    def __setstate__(self, state):
        super(MNasNetOFASearchSpace, self).__setstate__(state)
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, []
        )

    def genotype(self, arch):
        geno_arch = arch["depth"] + sum(
            [
                list(zip(channels, kernels))
                for channels, kernels in zip(arch["width"], arch["kernel"])
            ],
            [],
        )
        return self.genotype_type(**dict(zip(self.block_names, geno_arch)))

    def rollout_from_genotype(self, genotype):
        if isinstance(genotype, str):
            genotype = genotype_from_str(genotype, self)

        depth = list(genotype[:len(self.num_cell_groups)])
        width = []
        kernel = []
        ind = len(self.num_cell_groups)
        for _, max_depth in zip(depth, self.num_cell_groups):
            width_list = []
            kernel_list = []
            for _ in range(max_depth):
                try:
                    width_list.append(genotype[ind][0])
                    kernel_list.append(genotype[ind][1])
                except Exception:
                    width_list.append(genotype[ind])
                    kernel_list.append(3)
                ind += 1
            width.append(width_list)
            kernel.append(kernel_list)
        arch = {"depth": depth, "width": width, "kernel": kernel}
        return MNasNetOFARollout(arch, {}, self)

    def supported_rollout_types(self):
        return ["ofa"]

    def _uniform_mutate(self, rollout, mutation_prob=1.0): #pylint: disable=arguments-differ
        arch = rollout.arch
        new_arch = {
            "depth": [1, ],
            "width": [[1], ],
            "kernel": [[3], ]
        }
        layers = sum(arch["depth"][1:], 0)
        layer_mutation_prob = mutation_prob / layers
        depth_mutation_prob = mutation_prob / len(arch["depth"])
        for i, depth in enumerate(arch["depth"][1:], 1):
            width = arch["width"][i]
            kernel = arch["kernel"][i]
            new_depth = depth
            if np.random.random() < depth_mutation_prob:
                new_depth = np.random.choice(self.depth_choice)
            new_arch["depth"] += [new_depth]
            new_arch["width"] += [[]]
            new_arch["kernel"] += [[]]
            for w, k in zip(width, kernel):
                new_w = w
                new_k = k
                if np.random.random() < layer_mutation_prob:
                    new_w = np.random.choice(self.width_choice)
                if np.random.random() < layer_mutation_prob:
                    new_k = np.random.choice(self.kernel_choice)
                new_arch["width"][-1] += [new_w]
                new_arch["kernel"][-1] += [new_k]
        import ipdb; ipdb.set_trace()
        return MNasNetOFARollout(new_arch, "", self)

    def _single_mutate(self, rollout, depth_mutate_prob=0.5):
        arch = rollout.arch
        new_arch = copy.deepcopy(arch)
        mutate_depth_idx = np.random.randint(1, len(arch["depth"]))
        if np.random.random() < depth_mutate_prob:
            new_depth = np.random.choice(self.depth_choice)
            while new_depth == new_arch["depth"][mutate_depth_idx]:
                new_depth = np.random.choice(self.depth_choice)
            new_arch["depth"][mutate_depth_idx] = new_depth
        else:
            mutate_block_idx = np.random.randint(0, len(arch["width"][mutate_depth_idx]))
            new_width = np.random.choice(self.width_choice)
            new_kernel = np.random.choice(self.kernel_choice)
            while new_width == new_arch["width"][mutate_depth_idx][mutate_block_idx] and \
                new_kernel == new_arch["kernel"][mutate_depth_idx][mutate_block_idx]:
                new_width = np.random.choice(self.width_choice)
                new_kernel = np.random.choice(self.kernel_choice)
            new_arch["width"][mutate_depth_idx][mutate_block_idx] = new_width
            new_arch["kernel"][mutate_depth_idx][mutate_block_idx] = new_kernel
        return MNasNetOFARollout(new_arch, "", self)

    def mutate(self, rollout, mutation="single", **kwargs): #pylint: disable=arguments-differ
        assert mutation in ("uniform", "single")
        if mutation == "uniform":
            return self._uniform_mutate(rollout, **kwargs)
        elif mutation == "single":
            return self._single_mutate(rollout, **kwargs)

    def plot_arch(self, genotypes, filename, label, **kwargs):
        pass

    def random_sample(self):
        return MNasNetOFARollout(
            MNasNetOFARollout.random_sample_arch(
                self.expansions,
                self.num_cell_groups,
                self.width_choice,
                self.depth_choice,
                self.kernel_choice,
            ),
            info={},
            search_space=self,
        )

    def distance(self, arch1, arch2):
        pass


class MNasNetOFARollout(Rollout):
    NAME = "ofa"
    channel = None

    @property
    def depth(self):
        return self.arch["depth"]

    @property
    def width(self):
        self.channel = self.arch["width"]
        return self.channel

    @property
    def kernel(self):
        return self.arch["kernel"]

    @classmethod
    def random_sample_arch(
        cls, num_channels, num_cell_groups, width_choice, depth_choice, kernel_choice
    ):
        arch = {}
        arch["depth"] = np.min(
            [
                np.random.choice(depth_choice, size=len(num_cell_groups)),
                num_cell_groups,
            ],
            axis=0,
        ).tolist()
        arch["width"] = [
            np.min(
                [np.random.choice(width_choice, size=c), [num_channels[i]] * c], axis=0
            ).tolist()
            for i, c in enumerate(num_cell_groups)
        ]
        arch["kernel"] = [[3]] + [
            np.random.choice(kernel_choice, size=c).tolist() for c in num_cell_groups[1:]
        ]
        return arch
