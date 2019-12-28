import pytest

@pytest.mark.parametrize("case", [
    {"type": "lstm"},
    {"type": "gcn"}
])
def test_arch_embedder(case):
    from aw_nas.common import get_search_space
    from aw_nas.evaluator.arch_network import ArchEmbedder
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    cls = ArchEmbedder.get_class_(case["type"])
    embedder = cls(search_space)
    embedder.to(device)

    batch_size = 4
    rollouts = [search_space.random_sample() for _ in range(batch_size)]
    archs = [r.arch for r in rollouts]

    arch_embedding = embedder(archs)
    assert arch_embedding.shape[0] == batch_size
    assert arch_embedding.shape[1] == embedder.out_dim

def test_gcn_arch_embedder_utils():
    from aw_nas.evaluator.arch_network import GCNArchEmbedder
    from aw_nas.common import get_search_space
    from aw_nas.evaluator.arch_network import ArchEmbedder
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    embedder = GCNArchEmbedder(search_space)
    embedder.to(device)

    batch_size = 4
    rollouts = [search_space.random_sample() for _ in range(batch_size)]
    archs = [r.arch for r in rollouts]
    adj_matrices, node_embed = embedder.embed_and_transform_arch(archs)
    _num_nodes = search_space.num_steps + search_space.num_init_nodes
    assert adj_matrices.shape == (batch_size, search_space.num_cell_groups,
                                  _num_nodes, _num_nodes)
    assert node_embed.shape == (batch_size, search_space.num_cell_groups,
                                _num_nodes, embedder.op_dim)

def test_arch_comparator():
    from aw_nas.common import get_search_space
    from aw_nas.evaluator.arch_network import PointwiseComparator
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    comparator = PointwiseComparator(search_space)
    comparator.to(device)

    batch_size = 4
    archs_1 = [search_space.random_sample().arch for _ in range(batch_size)]
    archs_2 = [search_space.random_sample().arch for _ in range(batch_size)]

    # forward
    scores = comparator.predict(archs_1)
    assert len(scores) == batch_size
    compare_res = comparator.compare(archs_1, archs_2)
    assert len(compare_res) == batch_size

    # update
    comparator.update_predict(list(zip(archs_1, [0.1, 0.3, 0.4, 0.9])))
    comparator.update_compare(list(zip(archs_1, archs_2, [0.1, 0.3, 0.4, 0.9])))
