import os
import numpy as np
import torch

def _cnn_data(device="cuda", batch_size=2, shape=28):
    return (
        torch.rand(batch_size, 3, shape, shape, dtype=torch.float, device=device),
        torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device),
    )

code_snippet = """
from aw_nas import germ    
from torch import nn
import torch.nn.functional as F
from aw_nas.germ import GermSuperNet
class _tmp_supernet(GermSuperNet):
    NAME = "tmp_code_snippet"

    def __init__(self):
        super().__init__()
        channel_choice_1 = germ.Choices([16, 32, 64])
        channel_choice_2 = germ.Choices([16, 32, 64])
        channel_choice_3 = germ.Choices([32, 64, 128])
        channel_choice_4 = germ.Choices([32, 64, 128])
        with self.begin_searchable() as ctx:
            self.block_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
            self.bn_0 = nn.BatchNorm2d(32)
            self.block_1 = ctx.SearchableConvBNBlock(in_channels=32, out_channels=channel_choice_1, kernel_size=germ.Choices([3, 5, 7]))
            self.block_2 = ctx.SearchableConvBNBlock(in_channels=64, out_channels=channel_choice_2, kernel_size=germ.Choices([3, 5]))
            self.s_block_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

            self.block_3 = germ.SearchableConvBNBlock(ctx, in_channels=128, out_channels=channel_choice_3, kernel_size=3)
            self.block_4 = ctx.SearchableConvBNBlock(in_channels=128, out_channels=channel_choice_4, kernel_size=3)
            self.block_5 = ctx.SearchableConvBNBlock(in_channels=128, out_channels=channel_choice_3, kernel_size=3)
            self.fc_6 = nn.Linear(128, 10)

    def forward(self, inputs):
        # stage 1
        x = F.relu(self.bn_0(self.block_0(inputs)))
        x = F.relu(self.block_1(x))
        x = F.relu(self.block_2(x))
        x = self.s_block_1(x)

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

        def __init__(self):
            super().__init__()
            channel_choice_1 = germ.Choices([16, 32, 64])
            channel_choice_2 = germ.Choices([16, 32, 64])
            channel_choice_3 = germ.Choices([32, 64, 128])
            channel_choice_4 = germ.Choices([32, 64, 128])
            with self.begin_searchable() as ctx:
                self.block_0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
                self.bn_0 = nn.BatchNorm2d(32)
                self.block_1 = ctx.SearchableConvBNBlock(in_channels=32, out_channels=channel_choice_1, kernel_size=germ.Choices([3, 5, 7]))
                self.block_2 = ctx.SearchableConvBNBlock(in_channels=64, out_channels=channel_choice_2, kernel_size=germ.Choices([3, 5]))
                self.s_block_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

                self.block_3 = germ.SearchableConvBNBlock(ctx, in_channels=128, out_channels=channel_choice_3, kernel_size=3)
                self.block_4 = ctx.SearchableConvBNBlock(in_channels=128, out_channels=channel_choice_4, kernel_size=3)
                self.block_5 = ctx.SearchableConvBNBlock(in_channels=128, out_channels=channel_choice_3, kernel_size=3)
                self.fc_6 = nn.Linear(128, 10)

        def forward(self, inputs):
            # stage 1
            x = F.relu(self.bn_0(self.block_0(inputs)))
            x = F.relu(self.block_1(x))
            x = F.relu(self.block_2(x))
            x = self.s_block_1(x)

    	    # stage 2
            x = F.relu(self.block_5(
                F.relu(self.block_4(
                    F.relu(self.block_3(x))))))
            x = self.fc_6(F.avg_pool2d(x, x.shape[-1]).view((x.shape[0], -1)))
            return x

    # test generate search space cfg from supernet definition
    super_net = _tmp_supernet()
    ss_cfg = super_net.generate_search_space_cfg()
    assert len(ss_cfg["decisions"]) == 6
    assert len(ss_cfg["blocks"]) == 5
    ss_cfg_path = os.path.join(str(tmp_path), "ss_cfg.yaml")
    super_net.generate_search_space_cfg_to_file(ss_cfg_path)

    ss = get_search_space("germ", search_space_cfg_file=ss_cfg_path)
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
