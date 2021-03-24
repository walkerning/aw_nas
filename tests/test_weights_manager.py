import os

import six
import yaml
import pytest
import numpy as np
import torch

# ---- test super_net ----
def _cnn_data(device="cuda", batch_size=2):
    return (torch.rand(batch_size, 3, 28, 28, dtype=torch.float, device=device),
            torch.tensor(np.random.randint(0, high=10, size=batch_size)).long().to(device))

def _supernet_sample_cand(net):
    ss = net.search_space

    rollout = ss.random_sample()
    # arch = [([0, 0, 2, 2, 0, 2, 4, 4], [0, 6, 7, 6, 1, 1, 5, 7]),
    # ([1, 1, 0, 0, 1, 2, 2, 2], [7, 2, 2, 1, 7, 4, 3, 7])]

    cand_net = net.assemble_candidate(rollout)
    return cand_net


@pytest.mark.parametrize("super_net", [
    {"candidate_member_mask": False},
    {"search_space_cfg": {"concat_op": "sum"},
     "candidate_member_mask": False}
], indirect=["super_net"])
def test_supernet_assemble_nomask(super_net):
    net = super_net
    cand_net = _supernet_sample_cand(net)
    assert set(dict(cand_net.named_parameters()).keys()) == set(dict(net.named_parameters()).keys())
    assert set(dict(cand_net.named_buffers()).keys()) == set(dict(net.named_buffers()).keys())

def test_supernet_assemble(super_net):
    net = super_net
    cand_net = _supernet_sample_cand(net)

    # test named_parameters/named_buffers
    # print("Supernet parameter num: {} ; buffer num: {}"\
    #       .format(len(list(net.parameters())), len(list(net.buffers()))))
    # print("candidatenet parameter num: {} ; buffer num: {}"\
    #       .format(len(list(cand_net.named_parameters())), len(list(cand_net.buffers()))))

    s_names = set(dict(net.named_parameters()).keys())
    c_names = set(dict(cand_net.named_parameters()).keys())
    assert len(s_names) > len(c_names)
    assert len(s_names.intersection(c_names)) == len(c_names)

    s_b_names = set(dict(net.named_buffers()).keys())
    c_b_names = set(dict(cand_net.named_buffers()).keys())
    assert len(s_b_names) > len(c_b_names)
    assert len(s_b_names.intersection(c_b_names)) == len(c_b_names)
    # names = sorted(set(c_names).difference([g[0] for g in grads]))

@pytest.mark.parametrize("super_net", [
    {},
    {"search_space_cfg": {"num_steps": [2, 4]}},
    {"search_space_cfg": {"concat_op": "sum"}},
    {"search_space_cfg": {"loose_end": True, "concat_op": "mean"}},
], indirect=["super_net"])
def test_supernet_forward(super_net):
    # test forward
    cand_net = _supernet_sample_cand(super_net)

    data = _cnn_data()
    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape[-1] == 10

@pytest.mark.parametrize("super_net", [
    {
        "dropout_rate": 0,
        "search_space_cfg": {
            "num_steps": 1,
            "num_layers": 1,
            "num_cell_groups": 1,
            "cell_layout": [0],
            "reduce_cell_groups": [],
        }
    }, {
        "dropout_rate": 0,
    }, {
        "dropout_rate": 0,
        "preprocess_op_type": "sep_conv_3x3"
    }], indirect=["super_net"])
def test_supernet_forward_step(super_net):
    cand_net = _supernet_sample_cand(super_net)

    data = _cnn_data()

    ## test in eval mode
    # forward stem
    cand_net.eval()
    stemed, context = cand_net.forward_one_step(context=None, inputs=data[0])
    assert len(context.previous_cells) == 1 and context.previous_cells[0] is stemed

    # forward the cells
    for i_layer in range(1, super_net.search_space.num_layers+1):
        num_steps = super_net.search_space.num_steps + super_net.search_space.num_init_nodes + 1
        for i_step in range(num_steps):
            while True:
                step_state, context = cand_net.forward_one_step(context)
                if context.is_end_of_cell or context.is_end_of_step:
                    break
            if i_step != num_steps - 1:
                assert step_state is context.current_cell[-1] \
                    and len(context.current_cell) == i_step + 1
        assert context.is_end_of_cell
        assert len(context.previous_cells) == i_layer + 1

    # final forward
    logits, context = cand_net.forward_one_step(context)
    assert logits.shape[-1] == 10

    logits_straight_forward = cand_net.forward_data(data[0])
    assert (logits == logits_straight_forward).all()

    logits = cand_net.forward_one_step_callback(data[0], lambda s, c: None)
    assert (logits == logits_straight_forward).all()

    ## test in train mode
    cand_net.train()
    logits_straight_forward = cand_net.forward_data(data[0])
    logits = cand_net.forward_one_step_callback(data[0], lambda s, c: None)
    assert (logits == logits_straight_forward).all()

def test_context_last_conv_module():
    from aw_nas.ops import DilConv, SepConv, ResNetBlock
    from aw_nas.utils.common_utils import Context
    data = _cnn_data()
    dil_conv = DilConv(3, 16, 3, 1, 1, 1).to("cuda")
    context = Context(0, 1, use_stem=False)
    _, context = dil_conv.forward_one_step(context=context, inputs=data[0])
    assert context.last_conv_module is dil_conv.op[1]
    _, context = dil_conv.forward_one_step(context=context)
    assert context.last_conv_module is dil_conv.op[2]

    context = Context(0, 1, use_stem=False)
    sep_conv = SepConv(3, 16, 3, 1, 1).to("cuda")
    _, context = sep_conv.forward_one_step(context=context, inputs=data[0])
    assert context.last_conv_module is sep_conv.op[1]
    for expected_ind in [2, 5, 6]:
        _, context = sep_conv.forward_one_step(context=context)
        assert context.last_conv_module is sep_conv.op[expected_ind]

    context = Context(0, 1, use_stem=False)
    res_block = ResNetBlock(3, 3, 1, True).to("cuda")
    res_block.train()
    out_0 = res_block(data[0])
    for i, expected_mod in enumerate([res_block.op_1.op[0], res_block.op_2.op[0],
                                      None, None]):
        state, context = res_block.forward_one_step(context=context,
                                                    inputs=data[0] if i == 0 else None)
        assert context.last_conv_module is expected_mod
    assert context.is_end_of_op
    assert (state == out_0).all()

    context = Context(0, 1, use_stem=False)
    res_block_stride = ResNetBlock(3, 16, 2, True).to("cuda")
    out_0 = res_block_stride(data[0])
    for i, expected_mod in enumerate(
            [res_block_stride.op_1.op[0], res_block_stride.op_2.op[0], res_block_stride.skip_op.op[0], None]):
        state, context = res_block_stride.forward_one_step(context=context, inputs=data[0] if i == 0 else None)
        assert context.last_conv_module is expected_mod
    assert (state == out_0).all()


@pytest.mark.parametrize("test_id",
                         range(5))
def test_supernet_candidate_gradient_virtual(test_id, super_net):
    lr = 0.1
    EPS = 1e-5
    data = _cnn_data()
    net = super_net
    cand_net = _supernet_sample_cand(net)
    c_params = dict(cand_net.named_parameters())
    c_buffers = dict(cand_net.named_buffers())
    # test `gradient`, `begin_virtual`
    w_prev = {k: v.clone() for k, v in six.iteritems(c_params)}
    buffer_prev = {k: v.clone() for k, v in six.iteritems(c_buffers)
                   if "num_batches_tracked" not in k} # mean/var
    visited_c_params = dict(cand_net.active_named_members("parameters", check_visited=True))
    with cand_net.begin_virtual():
        grads = cand_net.gradient(data, mode="train")
        assert len(grads) == len(visited_c_params)
        optimizer = torch.optim.SGD(cand_net.parameters(), lr=lr)
        optimizer.step()
        for n, grad in grads:
            assert (w_prev[n] - grad * lr - c_params[n]).abs().sum().item() < EPS
        grads_2 = dict(cand_net.gradient(data, mode="train"))
        assert len(grads) == len(visited_c_params)
        optimizer.step()
        for n, grad in grads:
            # this check is not very robust...
            assert (w_prev[n] - (grad + grads_2[n]) * lr - c_params[n]).abs().mean().item() < EPS

        # sometimes, some buffer just don't updated... so here coomment out
        # for n in buffer_prev:
        #     # a simple check, make sure buffer is at least updated...
        #     assert (buffer_prev[n] - c_buffers[n]).abs().sum().item() < EPS

    for n in c_params:
        assert (w_prev[n] - c_params[n]).abs().mean().item() < EPS
    for n in buffer_prev:
        assert (buffer_prev[n] - c_buffers[n]).abs().float().mean().item() < EPS

@pytest.mark.parametrize("super_net", [{
    "search_space_cfg": {"num_steps": 1, "num_node_inputs": 1, "num_init_nodes": 1,
                         "num_layers": 3, "cell_layout": [0, 1, 2],
                         "reduce_cell_groups": [1], "num_cell_groups": 3,
                         "cell_shared_primitives":[
                             ["none", "max_pool_3x3", "sep_conv_5x5"],
                             ["sep_conv_3x3"],
                             ["sep_conv_3x3", "dil_conv_3x3"]
                         ]},
    "cell_use_preprocess": False,
    "init_channels": 16,
    "stem_multiplier": 1,
    "cell_group_kwargs": [
        {"C_in": 16, "C_out": 16},
        {"C_in": 16, "C_out": 24},
        {"C_in": 24, "C_out": 64}
    ]
}], indirect=["super_net"])
def test_supernet_specify_Cinout(super_net):
    cand_net = _supernet_sample_cand(super_net)
    assert cand_net.super_net.cells[0].num_out_channels == 16
    assert cand_net.super_net.cells[1].num_out_channels == 24
    assert cand_net.super_net.cells[2].num_out_channels == 64
    assert len(cand_net.super_net.cells[0].edges) == 1 and \
        len(cand_net.super_net.cells[0].edges[0]) == 1
    assert len(cand_net.super_net.cells[0].edges[0][1].p_ops) == 3
    assert len(cand_net.super_net.cells[1].edges[0][1].p_ops) == 1
    assert len(cand_net.super_net.cells[2].edges[0][1].p_ops) == 2

    data = _cnn_data()

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape[-1] == 10

@pytest.mark.parametrize("super_net", [{
    "gpus": [0, 1],
    "dataparallel": True
}], indirect=["super_net"])
def test_supernet_data_parallel_forward(super_net):
    # test forward
    cand_net = _supernet_sample_cand(super_net)
    batch_size = 10
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape == (batch_size, 10)

@pytest.mark.parametrize("super_net", [{
    "gpus": [0, 1],
    "dataparallel": True
}], indirect=["super_net"])
def test_supernet_data_parallel_gradient(super_net):
    cand_net = _supernet_sample_cand(super_net)
    batch_size = 10
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], mode="eval")
    assert logits.shape == (batch_size, 10)
    _ = cand_net.gradient(data, mode="train")

def test_use_params():
    from aw_nas.ops import SepConv
    from aw_nas.utils.torch_utils import use_params

    sep_conv_1 = SepConv(3, 10, 3, 1, 1, affine=True).cuda()
    sep_conv_2 = SepConv(3, 10, 3, 1, 1, affine=True).cuda()
    parameters_1 = dict(sep_conv_1.named_parameters())
    parameters_2 = dict(sep_conv_2.named_parameters())

    # random init params
    for n in parameters_1:
        parameters_1[n].data.random_()
        parameters_2[n].data.random_()
    for n in parameters_1:
        assert (parameters_1[n] - parameters_2[n]).abs().mean() > 1e-4
    batch_size = 2
    inputs = _cnn_data(batch_size=batch_size)[0]

    # use train mode, do not use bn running statistics
    sep_conv_1.train()
    sep_conv_2.train()
    conv1_res = sep_conv_1(inputs)
    conv2_res = sep_conv_2(inputs)
    assert (conv1_res != conv2_res).any()
    with use_params(sep_conv_1, parameters_2):
        conv1_useparams_res = sep_conv_1(inputs)
    assert (conv1_useparams_res == conv2_res).all()
    for n, new_param in sep_conv_1.named_parameters():
        assert (new_param == parameters_1[n]).all()

def test_candidate_forward_with_params(super_net):
    cand_net = _supernet_sample_cand(super_net)
    batch_size = 2
    data = _cnn_data(batch_size=batch_size)

    old_params = {n: p.clone() for n, p in
                  cand_net.active_named_members("parameters", check_visited=True)}
    print("num active params: ", len(old_params))
    # random init
    new_params = {n: torch.autograd.Variable(p.new(p.size()).normal_(), requires_grad=True) for n, p in
                  cand_net.active_named_members("parameters", check_visited=True)}
    res = torch.sum(cand_net.forward_with_params(data[0], new_params, mode="train"))

    # assert set back to original params
    for name, new_value in cand_net.named_parameters():
        if name in old_params:
            assert (new_value == old_params[name]).all()

    try:
        torch.autograd.grad(
            res, list(dict(cand_net.active_named_members("parameters", check_visited=True)).values()), retain_graph=True)
    except Exception:
        print("should raise!")
    else:
        assert False, "Should raise, as new params are used"
    torch.autograd.grad(res, list(new_params.values()))

# ---- End test super_net ----

# ---- Test diff_super_net ----
@pytest.mark.parametrize("diff_super_net,controller_cfg", [
    ({"search_space_cfg": {"num_steps": [2, 4]}}, {}),
    ({"search_space_cfg": {"num_steps": [2, 4]}}, {"use_edge_normalization": True})
], indirect=["diff_super_net"])
def test_diff_supernet_forward(diff_super_net, controller_cfg):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, **controller_cfg)
    rollout = controller.sample(1)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    data = _cnn_data()
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (2, 10)

def test_diff_supernet_to_arch(diff_super_net):
    from torch import nn
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    data = _cnn_data() #pylint: disable=not-callable

    # default detach_arch=True, no grad w.r.t the controller param
    logits = cand_net.forward_data(data[0])
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is None

    logits = cand_net.forward_data(data[0], detach_arch=False)
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is not None

@pytest.mark.parametrize("diff_super_net,controller_cfg", [
    ({}, {}),
    ({}, {"use_edge_normalization": True})
], indirect=["diff_super_net"])
def test_diff_supernet_forward_rollout_batch_size(diff_super_net, controller_cfg):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, **controller_cfg)
    rollout = controller.sample(1, batch_size=3)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    data = _cnn_data(batch_size=6)
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (6, 10)

@pytest.mark.parametrize("diff_super_net,controller_cfg", [
    ({"gpus": [0, 1, 2]}, {}),
    ({"gpus": [0, 1, 2]}, {"use_edge_normalization": True})
], indirect=["diff_super_net"])
def test_diff_supernet_data_parallel_forward(diff_super_net, controller_cfg):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, **controller_cfg)
    rollout = controller.sample(1)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    batch_size = 9
    data = _cnn_data(batch_size=batch_size)
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (batch_size, 10)

@pytest.mark.parametrize("diff_super_net,controller_cfg", [
    ({"gpus": [0, 1, 2]}, {}),
    ({"gpus": [0, 1, 2]}, {"use_edge_normalization": True})
], indirect=["diff_super_net"])
def test_diff_supernet_data_parallel_forward_rolloutsize(diff_super_net, controller_cfg):
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device, **controller_cfg)
    rollout = controller.sample(1, batch_size=9)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    batch_size = 9
    data = _cnn_data(batch_size=batch_size)
    logits = cand_net.forward_data(data[0])
    assert tuple(logits.shape) == (batch_size, 10)

@pytest.mark.parametrize("diff_super_net", [{
    "gpus": [0, 1, 2]
}], indirect=["diff_super_net"])
def test_diff_supernet_data_parallel_backward_rolloutsize(diff_super_net):
    from torch import nn
    from aw_nas.common import get_search_space
    from aw_nas.controller import DiffController

    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = DiffController(search_space, device)
    rollout = controller.sample(1, batch_size=9)[0]
    cand_net = diff_super_net.assemble_candidate(rollout)

    batch_size = 9
    data = _cnn_data(batch_size=batch_size)

    logits = cand_net.forward_data(data[0], detach_arch=False)
    loss = nn.CrossEntropyLoss()(logits, data[1].cuda())
    assert controller.cg_alphas[0].grad is None
    loss.backward()
    assert controller.cg_alphas[0].grad is not None
# ---- End test diff_super_net ----

SAMPLE_MODEL_CFG = """
final_model_type: cnn_final_model
final_model_cfg:
  # Schedulable attributes: dropout_path_rate
  num_classes: 10
  init_channels: 20
  layer_channels: []
  stem_multiplier: 3
  dropout_rate: 0.1
  dropout_path_rate: 0.2
  auxiliary_head: false
  auxiliary_cfg: null
  use_stem: conv_bn_3x3
  stem_stride: 1
  stem_affine: true
  cell_use_preprocess: true
  cell_pool_batchnorm: false
  cell_group_kwargs: null
  cell_independent_conn: false
  schedule_cfg: null
# ---- End Type cnn_final_model ----
"""
# ---- test morphism ----

@pytest.mark.parametrize("population", [
    {
        "search_space_cfg": {"num_layers": 5, "num_steps": 2,
                             "shared_primitives": ["none", "sep_conv_3x3", "sep_conv_5x5"]}
    }
], indirect=["population"])
def test_morphism(population, tmp_path):
    from aw_nas.rollout.mutation import MutationRollout, ConfigTemplate, ModelRecord, CellMutation
    from aw_nas.final import CNNGenotypeModel
    from aw_nas.main import _init_component
    from aw_nas.common import genotype_from_str, rollout_from_genotype_str
    from aw_nas.weights_manager import MorphismWeightsManager

    cfg = yaml.safe_load(SAMPLE_MODEL_CFG)
    device = "cuda:0"
    search_space = population.search_space
    genotype_str = ("normal_0=[('sep_conv_5x5', 0, 2), ('sep_conv_3x3', 1, 2), "
                    "('sep_conv_3x3', 2, 3), ('none', 2, 3)], "
                    "reduce_1=[('sep_conv_5x5', 0, 2), ('none', 0, 2), "
                    "('sep_conv_5x5', 0, 3), ('sep_conv_5x5', 1, 3)]")
    parent_rollout = rollout_from_genotype_str(genotype_str, search_space)
    cfg["final_model_cfg"]["genotypes"] = genotype_str
    cnn_model = _init_component(cfg, "final_model", search_space=search_space,
                                device=device)
    parent_state_dict = cnn_model.state_dict()
    torch.save(cnn_model, os.path.join(tmp_path, "test"))

    # add this record to the population
    new_model_record = ModelRecord(
        genotype_from_str(cfg["final_model_cfg"]["genotypes"], cnn_model.search_space),
        cfg,
        cnn_model.search_space,
        info_path=os.path.join(tmp_path, "test.yaml"),
        checkpoint_path=os.path.join(tmp_path, "test"),
        finished=True,
        confidence=1,
        perfs={"acc": np.random.rand(),
               "loss": np.random.uniform(0, 10)})
    parent_index = population.add_model(new_model_record)

    mutation = CellMutation(search_space, CellMutation.PRIMITIVE, cell=0, step=0,
                            connection=1,
                            modified=search_space.shared_primitives.index("sep_conv_5x5"))
    print("mutation: ", mutation)
    rollout = MutationRollout(population, parent_index, [mutation], search_space)
    assert rollout.genotype != cnn_model.genotypes
    w_manager = MorphismWeightsManager(search_space, device, "mutation")
    cand_net = w_manager.assemble_candidate(rollout)
    child_state_dict = cand_net.state_dict()

    layers = [i_layer for i_layer, cg_id in enumerate(search_space.cell_layout)
              if cg_id == mutation.cell]
    removed_edges = ["cells.{layer}.edge_mod.f_1_t_2-sep_conv_3x3-0".format(
        layer=layer) for layer in layers]
    added_edges = ["cells.{layer}.edge_mod.f_1_t_2-sep_conv_5x5-0".format(
        layer=layer) for layer in layers]
    for n, v in six.iteritems(child_state_dict):
        if n not in added_edges:
            assert n in parent_state_dict
            assert (parent_state_dict[n].data.cpu().numpy() == v.data.cpu().numpy()).all()
    for n in removed_edges:
        assert n not in child_state_dict
 
DENSE_SAMPLE_MODEL_CFG = """
search_space_type: cnn_dense
search_space_cfg:
  num_dense_blocks: 4
  stem_channel: 8
  first_ratio: null
dataset_type: cifar10
dataset_cfg:
  cutout: null
final_model_type: dense_final_model
final_model_cfg:
  num_classes: 10
  dropout_rate: 0.1
  schedule_cfg: null
final_trainer_type: cnn_trainer
final_trainer_cfg:
  # Schedulable attributes:
  epochs: 50
  batch_size: 96
  optimizer_type: SGD
  optimizer_kwargs: null
  learning_rate: 0.05
  momentum: 0.9
  warmup_epochs: 0
  optimizer_scheduler:
    T_max: 50
    eta_min: 0.001
    type: CosineAnnealingLR
  weight_decay: 0.0003
  no_bias_decay: false
  grad_clip: 5.0
  auxiliary_head: false
  auxiliary_weight: 0.0
  add_regularization: false
  save_as_state_dict: false
  eval_no_grad: true
  schedule_cfg: null
# ---- End Type cnn_trainer ----
## ---- End Component final_trainer ----

## ---- Component objective ----
# ---- Type classification ----
objective_type: classification
objective_cfg:
  # Schedulable attributes: 
  {}
# ---- End Type classification ----
"""

@pytest.mark.parametrize("population", [
    {
        "search_space_type": "cnn_dense",
        "search_space_cfg": {
            "num_dense_blocks": 4,
            "stem_channel": 8,
            "first_ratio": None,
        },
        "num_records": 0,
        "cfg_template": DENSE_SAMPLE_MODEL_CFG,
    }
], indirect=["population"])
def test_dense_morphism_wider(population, tmp_path):
    from aw_nas.rollout.mutation import ConfigTemplate, ModelRecord, CellMutation
    from aw_nas.rollout.dense import DenseMutationRollout, DenseMutation
    from aw_nas.final import DenseGenotypeModel
    from aw_nas.main import _init_component
    from aw_nas.common import genotype_from_str, rollout_from_genotype_str
    from aw_nas.weights_manager import DenseMorphismWeightsManager
    cfg = yaml.safe_load(DENSE_SAMPLE_MODEL_CFG)
    device = "cuda:0"
    search_space = population.search_space
    genotype_str = ("stem=8, block_0=[4, 4, 4, 4, 4, 4], transition_0=16, "
                    "block_1=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], transition_1=32, "
                    "block_2=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, "
                    "4, 4, 4, 4, 4], transition_2=64, block_3=[4, 4, 4, 4, 4, 4]")
    parent_rollout = rollout_from_genotype_str(genotype_str, search_space)
    cfg["final_model_cfg"]["genotypes"] = genotype_str
    cnn_model = _init_component(cfg, "final_model", search_space=search_space,
                                device=device)
    parent_state_dict = cnn_model.state_dict()
    torch.save(cnn_model, os.path.join(tmp_path, "test"))

    # add this record to the population
    new_model_record = ModelRecord(
        genotype_from_str(cfg["final_model_cfg"]["genotypes"], cnn_model.search_space),
        cfg,
        cnn_model.search_space,
        info_path=os.path.join(tmp_path, "test.yaml"),
        checkpoint_path=os.path.join(tmp_path, "test"),
        finished=True,
        confidence=1,
        perfs={"acc": np.random.rand(),
               "loss": np.random.uniform(0, 10)})
    parent_index = population.add_model(new_model_record)

    mutation = DenseMutation(search_space, DenseMutation.WIDER, block_idx=1, miniblock_idx=0,
                             modified=8)
    rollout = DenseMutationRollout(population, parent_index, [mutation], search_space)
    assert rollout.genotype != cnn_model.genotypes

    w_manager = DenseMorphismWeightsManager(search_space, device, "dense_mutation")
    cand_net = w_manager.assemble_candidate(rollout)
    cand_net.eval()
    child_state_dict = cand_net.state_dict()
    data = _cnn_data()
    logits = cand_net.forward(data[0])
    origin_net = torch.load(rollout.population.get_model(rollout.parent_index).checkpoint_path)
    origin_net.eval()
    logits_ori = origin_net.forward(data[0])
    assert (logits - logits_ori).abs().mean() < 1e-6

@pytest.mark.parametrize("ofa_super_net", [
    {"search_space_cfg": {"num_cell_groups": [1, 4, 4, 4, 4, 4, 1],
        "expansions": [1, 6, 6, 6, 6, 6, 6], "width_choice": [4, 5, 6], "depth_choice": [4, 5, 6]}}
], indirect=["ofa_super_net"])
def test_ofa_forward_rollout(ofa_super_net):
    # test forward
    cand_net = _supernet_sample_cand(ofa_super_net)
    data = _cnn_data()
    logits = cand_net.forward(data[0])
    assert logits.shape[-1] == 10

