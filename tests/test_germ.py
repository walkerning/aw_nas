#pylint: disable=invalid-name
import copy
import os

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
    with pytest.raises(AttributeError):
        outputs = net(data[0])

    # ---- test weights manager forward ----
    from aw_nas.weights_manager.base import BaseWeightsManager
    wm = BaseWeightsManager.get_class_("germ")(ss, "cuda", rollout_type="germ",
                                               germ_supernet_type="nb201",
                                               germ_supernet_cfg={
                                                   "num_layers": 5,
                                                   "dropout_rate": 0.
                                               })

    # ---- test finalize ----
    final_net = wm.super_net.finalize_rollout(rollout)
    print(final_net)
    outputs = final_net(data[0])
    assert outputs.shape == (2, 10)

    return
    with pytest.raises(AttributeError):
        outputs_2 = wm.super_net(data[0])
    cand_net = wm.assemble_candidate(rollout)
    outputs_2 = cand_net(data[0])
    assert outputs_2.shape == (2, 10)
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
        "mul_len": 16 
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
    assert a.is_leaf()
    c = a * b
    assert not c.is_leaf()
    assert len(c.choices) == choices["mul_len"]
    assert len(c.p) == len(c.choices)

    for epoch, nb_c in zip([5, 11, 21, 31], [1, 2, 3, 4]):
        c.on_epoch_start(epoch)
        print(c.choices, a.choices)
        assert a.num_choices == nb_c
        if isinstance(b, Choices):
            assert c.num_choices == a.num_choices * b.num_choices
        else:
            assert c.num_choices == a.num_choices

