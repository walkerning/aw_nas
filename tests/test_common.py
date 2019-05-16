import os
import numpy as np

def test_search_space(tmp_path):
    from aw_nas.common import get_search_space, Rollout

    ss = get_search_space(cls="cnn")
    arch = ss.random_sample()
    rollout = Rollout(arch, info={}, search_space=ss)
    mock_edge_label = np.random.rand(ss.num_cell_groups,
                                     ss.num_steps*ss.num_node_inputs)
    mock_edge_label = np.vectorize("{:.3f}".format)(mock_edge_label)
    print("genotype: ", rollout.genotype)
    prefix = os.path.join(str(tmp_path), "cell")
    fnames = rollout.plot_arch(prefix, label="test plot",
                               edge_labels=mock_edge_label.tolist())
    assert fnames == [(cn, prefix + "-{}.png".format(cn)) for cn in ss.cell_group_names]
