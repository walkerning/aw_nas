#pylint: disable=invalid-name
from aw_nas.germ.decisions import SelectNonleafChoices
import os
import copy
import pickle
from pprint import pprint

import pytest
import numpy as np
import torch


def _cnn_data(device="cuda", batch_size=2, shape=28, input_c=3):
    return (
        torch.rand(batch_size, input_c, shape, shape, dtype=torch.float, device=device),
        torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device),
    )

code_snippet = """
from aw_nas import germ
from torch import nn
import torch.nn.functional as F
from aw_nas.germ import GermSuperNet
class _tmp_supernet(GermSuperNet):
    NAME = "tmp_code_snippet"

    def __init__(self, search_space):
        super().__init__(search_space)
        channel_choice_1 = germ.Choices([16, 32, 64])
        channel_choice_2 = germ.Choices([16, 32, 64])
        channel_choice_3 = germ.Choices([32, 64, 128])
        channel_choice_4 = germ.Choices([32, 64, 128])
        with self.begin_searchable() as ctx:
            self.block_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
            self.bn_0 = nn.BatchNorm2d(32)
            self.block_1 = ctx.SearchableConvBNBlock(in_channels=32, out_channels=channel_choice_1, kernel_size=germ.Choices([3, 5, 7]))
            self.block_2 = ctx.SearchableConvBNBlock(in_channels=channel_choice_1, out_channels=channel_choice_2, kernel_size=germ.Choices([3, 5]))

            self.block_3 = germ.SearchableConvBNBlock(ctx, in_channels=channel_choice_2, out_channels=channel_choice_3, kernel_size=3)
            self.block_4 = ctx.SearchableConvBNBlock(in_channels=channel_choice_3, out_channels=channel_choice_4, kernel_size=3)
            self.block_5 = ctx.SearchableConvBNBlock(in_channels=channel_choice_4, out_channels=128, kernel_size=3)
            self.fc_6 = nn.Linear(128, 10)

    def forward(self, inputs):
        # stage 1
        x = F.relu(self.bn_0(self.block_0(inputs)))
        x = F.relu(self.block_1(x))
        x = F.relu(self.block_2(x))

        # stage 2
        x = F.relu(self.block_5(
            F.relu(self.block_4(
                F.relu(self.block_3(x))))))
        x = self.fc_6(F.avg_pool2d(x, x.shape[-1]).view((x.shape[0], -1)))
        return x
"""
def test_germ_supernet(tmp_path):
    from aw_nas import germ
    from aw_nas.weights_manager.base import BaseWeightsManager
    from aw_nas.germ import GermWeightsManager
    from aw_nas.common import get_search_space, rollout_from_genotype_str, genotype_from_str

    from aw_nas import germ
    from torch import nn
    import torch.nn.functional as F
    from aw_nas.germ import GermSuperNet
    class _tmp_supernet(GermSuperNet):
        NAME = "tmp"

        def __init__(self, search_space):
            super().__init__(search_space)
            channel_choice_1 = germ.Choices([16, 32, 64])
            channel_choice_2 = germ.Choices([16, 32, 64])
            channel_choice_3 = germ.Choices([32, 64, 128])
            channel_choice_4 = germ.Choices([32, 64, 128])
            with self.begin_searchable() as ctx:
                self.block_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
                self.bn_0 = nn.BatchNorm2d(32)
                self.block_1 = ctx.SearchableConvBNBlock(in_channels=32, out_channels=channel_choice_1, kernel_size=germ.Choices([3, 5, 7]))
                self.block_2 = ctx.SearchableConvBNBlock(in_channels=channel_choice_1, out_channels=channel_choice_2, kernel_size=germ.Choices([3, 5]))

                self.block_3 = germ.SearchableConvBNBlock(ctx, in_channels=channel_choice_2, out_channels=channel_choice_3, kernel_size=3)
                self.block_4 = ctx.SearchableConvBNBlock(in_channels=channel_choice_3, out_channels=channel_choice_4, kernel_size=3)
                self.block_5 = ctx.SearchableConvBNBlock(in_channels=channel_choice_4,
                        out_channels=128, kernel_size=3)
                self.fc_6 = nn.Linear(128, 10)

        def forward(self, inputs):
            # stage 1
            x = F.relu(self.bn_0(self.block_0(inputs)))
            x = F.relu(self.block_1(x))
            x = F.relu(self.block_2(x))

    	    # stage 2
            x = F.relu(self.block_5(
                F.relu(self.block_4(
                    F.relu(self.block_3(x))))))
            x = self.fc_6(F.avg_pool2d(x, x.shape[-1]).view((x.shape[0], -1)))
            return x

    # test generate search space cfg from supernet definition
    ss = get_search_space("germ")
    super_net = _tmp_supernet(ss)
    ss_cfg = super_net.generate_search_space_cfg()
    assert len(ss_cfg["decisions"]) == 6
    assert len(ss_cfg["blocks"]) == 9

    rollout = ss.random_sample() # test random_sample
    rollout_2 = ss.mutate(rollout, mutate_num=2) # test mutate
    print(rollout.genotype)
    print(rollout_2.genotype)
    rollout_rec = ss.rollout_from_genotype(genotype_from_str(str(rollout.genotype), ss))
    # do not support eq checking
    print(rollout_rec)

    # test weights manager assemble/forward
    wm = BaseWeightsManager.get_class_("germ")(ss, "cuda", rollout_type="germ",
                                               germ_supernet_type="tmp")
    cand_net = wm.assemble_candidate(rollout)
    data = _cnn_data(device="cuda", batch_size=2)
    outputs = cand_net(data[0])
    assert outputs.shape == (2, 10)

    # test initialize weights manager from code snippet python file
    code_snippet_path = os.path.join(str(tmp_path), "tmp.py")
    with open(code_snippet_path, "w") as w_f:
        w_f.write(code_snippet)

    wm = BaseWeightsManager.get_class_("germ")(
        ss, "cuda", rollout_type="germ",
        germ_supernet_type="tmp_code_snippet",
        germ_def_file=code_snippet_path
    )
    cand_net = wm.assemble_candidate(rollout)
    data = _cnn_data(device="cuda", batch_size=2)
    outputs = cand_net(data[0])
    assert outputs.shape == (2, 10)

def test_germ_nb201_and_finalize(tmp_path):
    from aw_nas.germ.nb201 import GermNB201Net
    from aw_nas.germ import GermSearchSpace
    # ---- test ss generate ----
    ss = GermSearchSpace()
    net = GermNB201Net(ss, num_layers=5, dropout_rate=0.).cuda()
    rollout = ss.random_sample()
    assert ss.get_size() == 15625
    print("search space size: ", ss.get_size())

    # ---- cannot forward a supernet without rollout context ----
    data = _cnn_data(device="cuda", batch_size=2)
    with pytest.raises(Exception):
        outputs = net(data[0])

    # ---- test weights manager forward ----
    from aw_nas.weights_manager.base import BaseWeightsManager
    wm = BaseWeightsManager.get_class_("germ")(ss, "cuda", rollout_type="germ",
                                               germ_supernet_type="nb201",
                                               germ_supernet_cfg={
                                                   "num_layers": 5,
                                                   "dropout_rate": 0.
                                               })

    with pytest.raises(Exception):
        outputs_2 = wm.super_net(data[0])
    cand_net = wm.assemble_candidate(rollout)
    outputs_2 = cand_net(data[0])
    assert outputs_2.shape == (2, 10)

    # ---- test finalize ----
    final_net = wm.super_net.finalize_rollout(rollout)
    print(final_net)
    outputs = final_net(data[0])
    assert outputs.shape == (2, 10)
    assert (outputs == outputs_2).all().item()


def test_op_on_edge_finalize(tmp_path):
    from aw_nas import germ

    class _tmp_supernet(germ.GermSuperNet):
        NAME = "tmp"

        def __init__(self, search_space):
            super().__init__(search_space)
            with self.begin_searchable() as ctx:
                self.node_1 = germ.GermOpOnEdge(
                    ctx,
                    from_nodes=[0],
                    num_input_nodes=1,
                    op_list=["sep_conv_3x3", "max_pool_3x3", "skip_connect"],
                    aggregation="sum",
                    allow_repeat_choice=False,
                    # op_kwargs
                    C=32,
                    C_out=32,
                    stride=1,
                    affine=False
                )
                self.node_2 = germ.GermOpOnEdge(
                    ctx,
                    from_nodes=[0, 1],
                    num_input_nodes=1,
                    op_list=["sep_conv_3x3", "max_pool_3x3"],
                    aggregation="sum",
                    allow_repeat_choice=False,
                    # op_kwargs
                    C=32,
                    C_out=32,
                    stride=1,
                    affine=False
                )
                self.node_3 = germ.GermOpOnEdge(
                    ctx,
                    from_nodes=[0, 1],
                    from_nodes_choices=self.node_2.from_nodes_choices,
                    num_input_nodes=1,
                    op_list=["sep_conv_3x3", "max_pool_3x3"],
                    edgeops_choices_dict=self.node_2.edgeops_choices_dict,
                    aggregation="sum",
                    allow_repeat_choice=False,
                    # op_kwargs
                    C=32,
                    C_out=32,
                    stride=1,
                    affine=False
                )

        def forward(self, inputs):
            nodes = [inputs]
            for node in [self.node_1, self.node_2, self.node_3]:
                nodes.append(node(nodes))
            return nodes[-1]
    ss = germ.GermSearchSpace()
    net = _tmp_supernet(ss).cuda()
    rollout = ss.random_sample()
    for i_mutate in range(5):
        rollout = ss.mutate(rollout, mutate_num=1)
        finalized_net = copy.deepcopy(net).finalize_rollout(rollout)
        print("Mutate {}".format(i_mutate), rollout.arch, finalized_net)
        data = _cnn_data(device="cuda", batch_size=2, input_c=32)
        #finalized_net(data[0])

def test_op_on_node_finalize(tmp_path):
    from aw_nas import germ

    class _tmp_supernet(germ.GermSuperNet):
        NAME = "tmp"

        def __init__(self, search_space):
            super().__init__(search_space)
            with self.begin_searchable() as ctx:
                self.node_1 = germ.GermOpOnNode(
                    ctx,
                    num_input_nodes=1,
                    from_nodes=[0],
                    op_list=["sep_conv_3x3", "max_pool_3x3", "skip_connect"],
                    aggregation=["mul", "sum"],

                    allow_repeat_choice=False,
                    # op_kwargs
                    C=32,
                    C_out=32,
                    stride=1,
                    affine=False
                )
                self.node_2 = germ.GermOpOnNode(
                    ctx,
                    num_input_nodes=1,
                    from_nodes=[0, 1],
                    op_list=["sep_conv_3x3", "max_pool_3x3", "skip_connect"],
                    aggregation="sum",

                    # would choose the same op as node_1
                    op_choice=self.node_1.op_choice,

                    # would have no effect, since aggregation is not a list/tuple
                    aggregation_op_choice=self.node_1.aggregation_op_choice,

                    allow_repeat_choice=False,
                    # op_kwargs
                    C=32,
                    C_out=32,
                    stride=1,
                    affine=False
                )
                self.node_3 = germ.GermOpOnNode(
                    ctx,
                    num_input_nodes=2,
                    from_nodes=[0, 1, 2],
                    op_list=["sep_conv_3x3", "max_pool_3x3", "skip_connect"],
                    aggregation="concat",

                    op_choice=self.node_1.op_choice,

                    allow_repeat_choice=False,
                    # op_kwargs
                    C=64,
                    C_out=32,
                    stride=1,
                    affine=False
                )

        def forward(self, inputs):
            nodes = [inputs]
            for node in [self.node_1, self.node_2, self.node_3]:
                nodes.append(node(nodes))
            return nodes[-1]
    ss = germ.GermSearchSpace()
    net = _tmp_supernet(ss).cuda()
    rollout = ss.random_sample()
    for i_mutate in range(5):
        rollout = ss.mutate(rollout, mutate_num=1)
        finalized_net = net.finalize_rollout(rollout)
        print("Mutate {}".format(i_mutate), rollout.arch, finalized_net)
        data = _cnn_data(device="cuda", batch_size=2, input_c=32)
        #finalized_net(data[0])

@pytest.mark.parametrize("choices", [
    {
        "a": {
            "choices": [1, 2, 3, 4],
            "p": [0.25, 0.35, 0.15, 0.25]
        },
        "b": {"choices": [16, 32, 48, 92]},
        "mul_len": 12
    },
    {
        "a": {
            "choices": [1, 2, 3, 4],
            "p": [0.25, 0.35, 0.15, 0.25]
        },
        "b": 12,
        "mul_len": 4
    }
])
def test_choices_mul(choices):
    from aw_nas.germ.decisions import Choices
    def callback(choices, epoch):
        if epoch <= 10:
            choices.choices = [1]
        if epoch > 10:
            choices.choices = [1, 2]
        if epoch > 20:
            choices.choices = [1, 2, 3]
        if epoch > 30:
            choices.choices = [1, 2, 3, 4]

    a = Choices(choices["a"]["choices"], p=choices["a"].get("p"), epoch_callback=callback)
    b = choices["b"]
    if isinstance(b, dict):
        b = Choices(choices["b"]["choices"], p=choices["b"].get("p"))
    c = a * b
    assert len(c.choices) == choices["mul_len"]

    for epoch, nb_c in zip([5, 11, 21, 31], [1, 2, 3, 4]):
        a.on_epoch_start(epoch)
        if isinstance(b, Choices):
            b.on_epoch_start(epoch)
        print(c.choices, a.choices)
        assert a.num_choices == nb_c
        if isinstance(b, Choices):
            assert c.num_choices == len(np.unique(np.array(a.choices)[:, None] * b.choices))
        else:
            assert c.num_choices == a.num_choices

def test_decision_container():
    from aw_nas import germ
    decision_list = germ.DecisionList()
    decision_list.append(germ.Choices([1, 2, 3]))
    decision_list.append(2)
    decision_list.append(germ.Choices([2, 3, 4]))
    print(decision_list)
    assert len(dict(decision_list.named_decisions())) == 2
    assert len(decision_list) == 3
    decision_list.insert(0, germ.Choices([3, 4, 5]))
    print(decision_list)
    assert len(dict(decision_list.named_decisions())) == 3
    assert len(decision_list) == 4

def test_nonleaf_decisions():
    from aw_nas import germ
    from aw_nas.germ import GermSearchSpace

    class _tmp_supernet(germ.GermSuperNet, germ.SearchableBlock):
        NAME = "tmp_code_snippet_nonleaf_decisions"
        def __init__(self, search_space):
            super().__init__(search_space)
            with self.begin_searchable() as ctx:
                self.c1 = germ.Choices([4, 8, 16])\
                              .apply(lambda value: value + 1)\
                              .apply(lambda value: value - 1)
                self.c2 = germ.Choices([0.25, 0.5, 1.0])
                self.c3 = self.c1 * self.c2
                self.c4 = (self.c1 * self.c2).apply(lambda value: value - 2)
                self.c5 = self.c4 + self.c1
                self.c6 = self.c1 + self.c2
                self.c7 = self.c1 / 2
                self.c8 = self.c1 - 3.1
                self.c9 = germ.ChoiceMax(self.c3, self.c7)
                assert self.c3.search_space_size == 1
                assert self.c3.num_choices == 5
                print(self.c3) # use Choice original repr
                print(self.c5) # use Choice original repr
                assert self.c5.search_space_size == 1
            print("c5 choices:", self.c5.choices)
            print("c6 choices:", self.c6.choices)

    ss = GermSearchSpace()
    supernet = _tmp_supernet(ss)
    print(supernet.c5.to_string())
    print(supernet.c3) # use decision id
    pprint(supernet.generate_search_space_ref())
    pprint(supernet.generate_search_space_cfg())
    rollout = ss.random_sample()
    print(rollout.arch)
    print(rollout[supernet.c1], rollout[supernet.c2],
          rollout[supernet.c3], rollout[supernet.c4],
          rollout[supernet.c5], rollout[supernet.c6],
          rollout[supernet.c7], rollout[supernet.c8],
          rollout[supernet.c9])
    print("--- after mutate ---")
    rollout_m = ss.mutate(rollout, mutate_num=2)
    print(rollout_m[supernet.c1], rollout_m[supernet.c2],
          rollout_m[supernet.c3], rollout_m[supernet.c4],
          rollout_m[supernet.c5], rollout_m[supernet.c6],
          rollout_m[supernet.c7], rollout_m[supernet.c8],
          rollout_m[supernet.c9])

    # serialize decisions and reload
    ss2 = GermSearchSpace()
    ss2.set_cfg(supernet.generate_search_space_cfg())
    rollout2 = ss2.random_sample()
    print(rollout2[supernet.c1], rollout2[supernet.c2],
          rollout2[supernet.c3], rollout2[supernet.c4],
          rollout2[supernet.c5], rollout2[supernet.c6],
          rollout2[supernet.c7], rollout2[supernet.c8],
          rollout2[supernet.c9])

def test_nonleaf_derive_decisions():
    from aw_nas import germ
    from aw_nas.germ import GermSearchSpace

    class _tmp_supernet(germ.GermSuperNet, germ.SearchableBlock):
        NAME = "tmp_code_snippet_nonleaf_derive_decisions"
        def __init__(self, search_space):
            super().__init__(search_space)
            with self.begin_searchable() as ctx:
                self.c1 = germ.Choices([1, 2, 3])
                self.c2 = self.c1 + 8
                self.c3 = germ.Choices([1, 2, 3])
                self.c4 = self.c3 + 4
                self.c5 = self.c1 + self.c3
                self.c6 = self.c2 + self.c4
                self.c7 = self.c6 - self.c5
                self.c8 = germ.SelectNonleafChoices(
                    [
                        self.c1,
                        self.c2,
                        5,
                        self.c4,
                        self.c5,
                        self.c6,
                        self.c7,
                    ],
                    self.c5,
                    optional_choices = [1, 4, 5, 6, 12]
                )

    ss = GermSearchSpace()
    supernet = _tmp_supernet(ss)
    print(supernet.c8.choices)
    print(supernet.c7.choices)
    print(supernet.c6.choices)
    print(supernet.c5.choices)
    print(supernet.c4.choices)
    print(supernet.c3.choices)
    print(supernet.c2.choices)
    print(supernet.c1.choices)
    pprint(supernet.generate_search_space_cfg())
    rollout = ss.random_sample()
    print(rollout.arch)
    print(rollout[supernet.c8])
    print("--- after mutate ---")
    rollout_m = ss.mutate(rollout, mutate_num=2)
    print(rollout_m.arch)
    print(rollout_m[supernet.c8])

    # serialize decisions and reload
    ss2 = GermSearchSpace()
    ss2.set_cfg(supernet.generate_search_space_cfg())
    rollout2 = ss2.random_sample()
    print(rollout2.arch)
    print(rollout2[supernet.c8])


def test_nonleaf_pickle():
    from aw_nas import germ
    c1 = germ.Choices([1, 2, 3])
    c2 = germ.Choices([1, 2, 3])
    c3 = c1 * c2
    c4 = SelectNonleafChoices([c1, c2, c3], c1 - 1)
    # pickling
    dump_res = pickle.dumps(c4)
    reloaded_c4 = pickle.loads(dump_res)
    print(reloaded_c4)
