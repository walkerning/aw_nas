"""
Definitions of mnasnet OFA rollout and search space.
"""

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
        genotype_list = list(genotype._asdict().values())

        depth = genotype[:len(self.num_cell_groups)]
        width = []
        kernel = []
        ind = len(self.num_cell_groups)
        for i, max_depth in zip(depth, self.num_cell_groups):
            width_list = []
            kernel_list = []
            for j in range(max_depth):
                if j < i:
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
