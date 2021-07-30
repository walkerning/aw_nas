from aw_nas.utils.torch_utils import random_cnn_data

def test_searchableconv_group_choices():
    from aw_nas import germ

    class Context(object):
        pass
    context = Context()
    group_choices = germ.Choices([4, 2, 1])
    group_choices.decision_id = "g_choices"
    ss = germ.GermSearchSpace()
    ss.set_cfg({
        "decisions": {"g_choices": ("choices", group_choices)},
        "blocks": {"": {"g_choices": group_choices}}
    })
    conv = germ.SearchableConv(context, 8, 16, 1, groups=group_choices).cuda()
    data = random_cnn_data(batch_size=1, shape=28, input_c=8)
    assert conv.weight.grad is None
    for _choice in group_choices.choices:
        rollout = ss.random_sample()
        rollout.arch['g_choices'] = _choice
        context.rollout = rollout
        outputs = conv(data[0])
        outputs.sum().backward()
        assert conv.weight.grad is not None
