import os
import numpy as np

import pytest

@pytest.mark.parametrize("case", [
    {"cls": "cnn"},
    {"cls": "rnn"},
    {"cls": "rnn", "loose_end": True},
])
def test_search_space(case, tmp_path):
    from aw_nas.common import get_search_space, Rollout

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
    from aw_nas.common import get_search_space, DifferentiableRollout
    from aw_nas.utils import softmax

    ss = get_search_space(cls="cnn")
    k = sum(ss.num_init_nodes+i for i in range(ss.num_steps))
    logits = [np.random.randn(k, len(ss.shared_primitives)) for _ in range(ss.num_cell_groups)]
    eps = 1e-20
    sampled = arch = [softmax(cg_logits + -np.log(-np.log(np.random.rand(*cg_logits.shape)+eps)+eps)) for cg_logits in logits]
    rollout = DifferentiableRollout(arch, sampled, logits, search_space=ss)
    print("genotype: ", rollout.genotype)
    prefix = os.path.join(str(tmp_path), "cell")
    fnames = rollout.plot_arch(prefix, label="test plot")
    assert fnames == [(cn, prefix + "-{}.png".format(cn)) for cn in ss.cell_group_names]
