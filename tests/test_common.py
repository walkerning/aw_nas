import os
import numpy as np

import pytest

@pytest.mark.parametrize("case", [
    {"cls": "cnn"},
    {"cls": "rnn"},
    {"cls": "rnn", "loose_end": True},
])
def test_search_space(case, tmp_path):
    from aw_nas.common import get_search_space

    ss = get_search_space(**case)
    rollout = ss.random_sample()
    mock_edge_label = np.random.rand(ss.num_cell_groups,
                                     ss.num_steps*ss.num_node_inputs)
    mock_edge_label = np.vectorize("{:.3f}".format)(mock_edge_label)
    prefix = os.path.join(str(tmp_path), "cell")
    print("genotype: ", rollout.genotype, "save to: ", prefix)
    fnames = rollout.plot_arch(prefix, label="test plot",
                               edge_labels=mock_edge_label.tolist())
    assert fnames == [(cn, prefix + "-{}.png".format(cn)) for cn in ss.cell_group_names]

def test_diff_rollout(tmp_path):
    import torch
    from aw_nas.common import get_search_space, DifferentiableRollout
    from aw_nas.utils import softmax

    ss = get_search_space(cls="cnn")
    k = sum(ss.num_init_nodes+i for i in range(ss.num_steps))
    logits = [np.random.randn(k, len(ss.shared_primitives)) for _ in range(ss.num_cell_groups)]
    eps = 1e-20
    sampled = arch = [torch.Tensor(softmax(
        cg_logits +
        -np.log(-np.log(np.random.rand(*cg_logits.shape)+eps)+eps)))
                      for cg_logits in logits]
    rollout = DifferentiableRollout(arch, sampled, logits, search_space=ss)
    print("genotype: ", rollout.genotype)
    prefix = os.path.join(str(tmp_path), "cell")
    fnames = rollout.plot_arch(prefix, label="test plot")
    assert fnames == [(cn, prefix + "-{}.png".format(cn)) for cn in ss.cell_group_names]

@pytest.mark.parametrize("genotype,cls", [
    ("""normal_0=[('dil_conv_5x5', 0, 2), ('sep_conv_5x5', 0, 2), ('avg_pool_3x3',
    2, 3), ('dil_conv_5x5', 0, 3), ('skip_connect', 1, 4), ('avg_pool_3x3', 0, 4),
    ('skip_connect', 0, 5), ('sep_conv_5x5', 0, 5)], reduce_1=[('max_pool_3x3', 0,
    2), ('sep_conv_3x3', 0, 2), ('skip_connect', 1, 3), ('dil_conv_5x5', 1, 3), ('dil_conv_5x5',
    2, 4), ('max_pool_3x3', 1, 4), ('sep_conv_5x5', 4, 5), ('skip_connect', 1, 5)]""",
     "cnn"),
    ("""cell=[('sigmoid', 0, 1), ('relu', 1, 2), ('relu', 1, 3), ('identity', 1,4), ('tanh', 2,5),
    ('sigmoid', 5,6), ('tanh', 3,7), ('relu', 5,8)], cell_concat=range(1, 9)""",
     "rnn")])
def test_plot_genotype_util(genotype, cls, tmp_path):
    from aw_nas.common import plot_genotype
    dest = os.path.join(str(tmp_path), cls)
    plot_genotype(genotype, dest, cls)
    print(dest)

@pytest.mark.parametrize("case", [
    {"cls": "cnn"},
    {"cls": "rnn"},
    {"cls": "rnn", "loose_end": True},
])
def test_rollout_from_genotype_str(case):
    from aw_nas.common import get_search_space, Rollout, rollout_from_genotype_str

    ss = get_search_space(**case)
    rollout = ss.random_sample()
    rec_rollout = rollout_from_genotype_str(str(rollout.genotype), ss)
    assert (np.array(rec_rollout.arch) == np.array(rollout.arch)).all()
