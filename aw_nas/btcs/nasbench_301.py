"""
NASBench-301 search space (a `cnn` search space), evaluator (API query)
"""

import copy
from collections import namedtuple

from ConfigSpace.read_and_write import json as cs_json
import numpy as np
import nasbench301 as nb

from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.common import CNNSearchSpace
from aw_nas.rollout.base import Rollout


class NB301SearchSpace(CNNSearchSpace):
    NAME = "nb301"

    def __init__(self, shared_primitives=(
            "sep_conv_3x3",
            "sep_conv_5x5",
            "dil_conv_3x3",
            "dil_conv_5x5",
            "max_pool_3x3",
            "avg_pool_3x3",
            "skip_connect")):
        super(NB301SearchSpace, self).__init__(
            num_cell_groups=2,
            num_init_nodes=2,
            num_layers=8, cell_layout=None,
            reduce_cell_groups=(1,),
            num_steps=4, num_node_inputs=2,
            concat_op="concat",
            concat_nodes=None,
            loose_end=False,
            shared_primitives=shared_primitives,
            derive_without_none_op=False)

    def mutate(self, rollout, node_mutate_prob=0.5):
        # NOTE: NB301 search space does not permit double connections between the same pair of nodes
        new_arch = copy.deepcopy(rollout.arch)
        # randomly select a cell gorup to modify
        mutate_i_cg = np.random.randint(0, self.num_cell_groups)
        num_prims = self._num_primitives \
                    if not self.cellwise_primitives else self._num_primitives_list[mutate_i_cg]
        _num_step = self.get_num_steps(mutate_i_cg)
        if np.random.random() < node_mutate_prob:
            # mutate connection
            # no need to mutate the connection to node 2, since no double connection is permited
            start = self.num_node_inputs
            node_mutate_idx = np.random.randint(start, len(new_arch[mutate_i_cg][0]))
            i_out = node_mutate_idx // self.num_node_inputs
            # self.num_node_inputs == 2
            # if node_mutate_idx is odd/even,
            # node_mutate_idx-1/+1 corresponds the other input node choice, respectively
            the_other_idx = node_mutate_idx - (2 * (node_mutate_idx % self.num_node_inputs) - 1)
            should_not_repeat = new_arch[mutate_i_cg][0][the_other_idx]
            out_node = i_out + self.num_init_nodes
            offset = np.random.randint(1, out_node - 1)
            old_choice = new_arch[mutate_i_cg][0][node_mutate_idx]
            new_choice = old_choice + offset # before mod
            tmp_should_not_repeat = should_not_repeat + out_node if should_not_repeat < old_choice \
                                    else should_not_repeat
            if tmp_should_not_repeat <= new_choice:
                # Should add 1 to the new choice
                new_choice += 1
            new_arch[mutate_i_cg][0][node_mutate_idx] = new_choice % out_node
        else:
            # mutate op
            op_mutate_idx = np.random.randint(0, len(new_arch[mutate_i_cg][1]))
            offset = np.random.randint(1, num_prims)
            new_arch[mutate_i_cg][1][op_mutate_idx] = \
                    (new_arch[mutate_i_cg][1][op_mutate_idx] + offset) % num_prims
        return Rollout(new_arch, info={}, search_space=self)

    def random_sample(self):
        """
        Random sample a discrete architecture.
        """
        return NB301Rollout(NB301Rollout.random_sample_arch(
            self.num_cell_groups, # =2
            self.num_steps, #=4
            self.num_init_nodes, # =2
            self.num_node_inputs, # =2
            self._num_primitives),
                            info={}, search_space=self)

    def canonicalize(self, rollout):
        # TODO
        archs = rollout.arch
        ss = rollout.search_space
        num_groups = ss.num_cell_groups
        num_vertices = ss.num_steps
        num_node_inputs = ss.num_node_inputs
        Res = ""

        for i_cg in range(num_groups):
            prims = ss.shared_primitives
            S = []
            outS = []
            S.append("0")
            S.append("1")
            arch = archs[i_cg]
            res = ""
            index = 0
            nodes = arch[0]
            ops = arch[1]
            for i_out in range(num_vertices):
                preS = []
                s = ""
                for i in range(num_node_inputs):
                    if (ops[index] == 6):
                        s = S[nodes[index]]
                    else:
                        s = "(" + S[nodes[index]] + ")" + "@" + prims[ops[index]]
                    preS.append(s)
                    index = index + 1
                preS.sort()
                s = ""
                for i in range(num_node_inputs):
                    s = s + preS[i]
                S.append(s)
                outS.append(s)
            outS.sort()
            for i_out in range(num_vertices):
                res = res + outS[i_out]
            if (i_cg < num_groups - 1):
                Res = Res + res + "&"
            else:
                Res = Res + res

        return Res

    @classmethod
    def supported_rollout_types(cls):
        return ["nb301"]


class NB301Rollout(Rollout):
    """
    For one target node, do not permit choose the same from node.
    """
    NAME = "nb301"
    supported_components = [
        ("trainer", "simple"),
        ("evaluator", "mepa"),
        ("weights_manager", "supernet")
    ]

    @classmethod
    def random_sample_arch(cls, num_cell_groups, num_steps,
                           num_init_nodes, num_node_inputs, num_primitives):
        """
        random sample
        """
        arch = []
        for i_cg in range(num_cell_groups):
            num_prims = num_primitives if isinstance(num_primitives, int) else num_primitives[i_cg]
            nodes = []
            ops = []
            _num_step = num_steps if isinstance(num_steps, int) else num_steps[i_cg]
            for i_out in range(_num_step):
                nodes += list(np.random.choice(
                    range(i_out + num_init_nodes), num_node_inputs, replace=False))
                ops += list(np.random.randint(
                    0, high=num_prims, size=num_node_inputs))
            arch.append((nodes, ops))
        return arch


class NB301Evaluator(BaseEvaluator):
    NAME = "nb301"

    def __init__(
            self,
            dataset,
            weights_manager,
            objective,
            rollout_type="nb301",
            with_noise=True,
            path=None,
            schedule_cfg=None,
    ):
        super(NB301Evaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type
        )

        assert path is not None, "must specify benchmark path"
        self.path = path
        self.with_noise = with_noise
        self.perf_model = nb.load_ensemble(path)
        self.genotype_type = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

    @classmethod
    def supported_data_types(cls):
        # cifar10
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["nb301"]

    def suggested_controller_steps_per_epoch(self):
        return None

    def suggested_evaluator_steps_per_epoch(self):
        return None

    def evaluate_rollouts(
        self,
        rollouts,
        is_training=False,
        portion=None,
        eval_batches=None,
        return_candidate_net=False,
        callback=None,
    ):
        prims = rollouts[0].search_space.shared_primitives
        for rollout in rollouts:
            # 2 cell groups: normal, reduce
            normal_arch, reduce_arch = [[
                (prims[op_idx], from_idx)
                for from_idx, op_idx in zip(*cell_arch)]
                                        for cell_arch in rollout.arch]
            genotype_config = self.genotype_type(
                normal=normal_arch, normal_concat=[2, 3, 4, 5],
                reduce=reduce_arch, reduce_concat=[2, 3, 4, 5])
            reward = self.perf_model.predict(
                config=genotype_config, representation="genotype",
                with_noise=self.with_noise)
            rollout.set_perf(reward, name="reward")
        return rollouts

    # ---- APIs that is not necessary ----
    def update_evaluator(self, controller):
        pass

    def update_rollouts(self, rollouts):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
