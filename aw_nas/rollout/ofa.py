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
    SCHEDULABLE_ATTRS = ["width_choice", "depth_choice", "kernel_choice",
                         "image_size"]

    def __init__(
        self,
        width_choice=(4, 5, 6),
        depth_choice=(2, 3, 4),
        kernel_choice=(3, 5, 7),
        image_size_choice=[224, 192, 160, 128],
        num_cell_groups=[1, 4, 4, 4, 4, 4],
        expansions=[1, 6, 6, 6, 6, 6],
        schedule_cfg=None,
    ):
        super(MNasNetOFASearchSpace, self).__init__(schedule_cfg)
        self.genotype_type_name = "MnasnetOFAGenotype"

        self.num_cell_groups = num_cell_groups
        self.expansions = expansions

        self.block_names = sum(
            [["image_size"]] +
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
        self.image_size_choice = image_size_choice
        self.max_image_size = max(self.image_size_choice)

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
        geno_arch = [arch["image_size"]] + arch["depth"] + sum(
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

        image_size, depth, width, kernel = self.parse(genotype)
        arch = {"image_size": image_size, "depth": depth,
                "width": width, "kernel": kernel}
        return MNasNetOFARollout(arch, {}, self)

    def parse(self, genotype):
        depth = list(genotype[1: 1 + len(self.num_cell_groups)])
        width = []
        kernel = []
        ind = 1 + len(self.num_cell_groups)
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
        image_size = genotype[0]
        return image_size, depth, width, kernel

    @classmethod
    def supported_rollout_types(self):
        return ["ofa"]

    def _uniform_mutate(self, rollout, mutation_prob=1.0):  # pylint: disable=arguments-differ
        arch = rollout.arch
        new_arch = {
            "depth": [1, ],
            "width": [[1], ],
            "kernel": [[3], ]
        }
        layers = sum(arch["depth"][1:], 0)
        layer_mutation_prob = mutation_prob / layers
        depth_mutation_prob = mutation_prob / len(arch["depth"])
        shape_mutation_prob = mutation_prob / len(self.image_size_choice)
        for i, depth in enumerate(arch["depth"][1:], 1):
            if np.random.random() < shape_mutation_prob:
                new_s = np.random.choice(self.image_size_choice)
            new_arch["image_size"] = new_s

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
        return MNasNetOFARollout(new_arch, "", self)

    @staticmethod
    def _get_another_choice(current, choices):
        cur_ind = choices.index(current)
        num_choice = len(choices)
        offset = np.random.randint(1, num_choice)
        return choices[(cur_ind + offset) % num_choice]

    def _single_mutate(self, rollout, mutation_prob=None):
        arch = rollout.arch
        if mutation_prob is None:
            num_each_arch = [1, len(arch["depth"]) - 1, sum(arch["depth"][1:]),
                             sum(arch["depth"][1:])]
            mutation_prob = [n / sum(num_each_arch) for n in num_each_arch]

        mutation_part_choices = []
        num_imgsize_choice, num_depth_choice, num_width_choice, num_kernel_choice = \
            len(self.image_size_choice), len(self.depth_choice), len(self.width_choice), \
            len(self.kernel_choice)
        if num_imgsize_choice > 1:
            mutation_part_choices.append("image_size")
        if num_depth_choice > 1:
            mutation_part_choices.append("depth")
        if num_width_choice > 1:
            mutation_part_choices.append("width")
        if num_kernel_choice > 1:
            mutation_part_choices.append("kernel")
        mutation_part = np.random.choice(mutation_part_choices, p=mutation_prob)

        new_arch = copy.deepcopy(arch)
        if mutation_part == "image_size":
            # cur_ind = self.image_size_choice.index(new_arch["image_size"])
            # offset = np.random.randint(1, num_imgsize_choice)
            # new_s = self.image_size_choice[(cur_ind + offset) % num_imgsize_choice]
            new_arch["image_size"] = self._get_another_choice(
                new_arch["image_size"], self.image_size_choice)
        elif mutation_part == "depth":
            mutate_depth_idx = np.random.randint(1, len(arch["depth"]))
            new_arch["depth"][mutate_depth_idx] = self._get_another_choice(
                new_arch["depth"][mutate_depth_idx], self.depth_choice)
        else:
            mutate_depth_idx = np.random.randint(1, len(arch["depth"]))
            mutate_block_idx = np.random.randint(
                0, len(arch[mutation_part][mutate_depth_idx]))
            new_val = self._get_another_choice(
                new_arch[mutation_part][mutate_depth_idx][mutate_block_idx],
                getattr(self, "{}_choice".format(mutation_part)))
            new_arch[mutation_part][mutate_depth_idx][mutate_block_idx] = new_val
        return MNasNetOFARollout(new_arch, "", self)

    def mutate(self, rollout, mutation="single", **kwargs):  # pylint: disable=arguments-differ
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
                self.image_size_choice,
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
        cls, num_channels, num_cell_groups, width_choice, depth_choice,
        kernel_choice, image_size_choice
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
        arch["image_size"] = np.random.choice(image_size_choice)
        return arch


class SSDOFASearchSpace(MNasNetOFASearchSpace):
    NAME = "ssd_ofa"
    SCHEDULABLE_ATTRS = ["width_choice", "depth_choice", "kernel_choice",
                         "head_width_choice"]

    def __init__(
        self,
        width_choice=(4, 5, 6),
        depth_choice=(2, 3, 4),
        kernel_choice=(3, 5, 7),
        image_size_choice=[512, 384, 320, 256, 192],
        head_width_choice=(0.25, 0.5, 0.75),
        head_kernel_choice=[3],
        num_cell_groups=[1, 4, 4, 4, 4, 4],
        expansions=[1, 6, 6, 6, 6, 6],
        num_head=4,
        schedule_cfg=None,
    ):
        super().__init__(
            width_choice,
            depth_choice,
            kernel_choice,
            image_size_choice,
            num_cell_groups,
            expansions,
            schedule_cfg
        )
        self.head_width_choice = head_width_choice
        self.head_kernel_choice = head_kernel_choice
        self.num_head = num_head
        self.num_cell_groups = num_cell_groups
        self.block_names += ["head_{}".format(i)
                             for i in range(num_head)]
        self.genotype_type = utils.namedtuple_with_defaults(
            self.genotype_type_name, self.block_names, []
        )

    def genotype(self, arch):
        geno_arch = [arch["image_size"]] + arch["depth"] + sum(
            [
                list(zip(channels, kernels))
                for channels, kernels in zip(
                    arch["width"] + [arch["head_width"]], arch["kernel"] +
                    [arch["head_kernel"]]
                )
            ],
            [],
        )
        return self.genotype_type(**dict(zip(self.block_names, geno_arch)))

    def rollout_from_genotype(self, genotype):
        if isinstance(genotype, str):
            genotype = genotype_from_str(genotype, self)
        image_size, depth, width, kernel = self.parse(
            genotype[:-self.num_head])
        head_width, head_kernel = list(zip(*genotype[-self.num_head:]))
        arch = {"image_size": image_size, "depth": depth, "width": width, "kernel": kernel,
                "head_width": head_width, "head_kernel": head_kernel}
        return SSDOFARollout(arch, {}, self)

    def mutate(self, rollout, mutation="single", **kwargs):
        ofa_rollout = super().mutate(rollout, mutation, **kwargs)
        return SSDOFARollout(rollout.arch, {}, self)

    def random_sample(self):
        return SSDOFARollout(
            SSDOFARollout.random_sample_arch(
                self.expansions,
                self.num_cell_groups,
                self.num_head,
                self.width_choice,
                self.depth_choice,
                self.kernel_choice,
                self.image_size_choice,
                self.head_width_choice,
                self.head_kernel_choice
            ),
            info={},
            search_space=self,
        )


class SSDOFARollout(MNasNetOFARollout):
    NAME = "ssd_ofa"
    supported_components = [("evaluator", "mepa"), ("trainer", "simple")]

    @property
    def head_width(self):
        return self.arch["head_width"]

    @property
    def head_kernel(self):
        return self.arch["head_kernel"]

    @property
    def image_size(self):
        return self.arch["image_size"]

    @classmethod
    def random_sample_arch(
        cls, num_channels, num_cell_groups, num_head, width_choice,
        depth_choice, kernel_choice, image_size_choice, head_width_choice,
        head_kernel_choice
    ):
        arch = super(SSDOFARollout, cls).random_sample_arch(
            num_channels, num_cell_groups, width_choice, depth_choice,
            kernel_choice, image_size_choice)
        arch["head_width"] = np.random.choice(
            head_width_choice, size=num_head).tolist()
        arch["head_kernel"] = np.random.choice(
            head_kernel_choice, size=num_head).tolist()
        return arch
