import os
import pytest

# we use environments variable to mark slow instead of register new pytest marks here.
AWNAS_TEST_SLOW = os.environ.get("AWNAS_TEST_SLOW", None)

@pytest.mark.skipif(not AWNAS_TEST_SLOW, reason="parse corpus might be slow, by default not test")
def test_ptb_batchify():
    from aw_nas.dataset import BaseDataset
    from aw_nas.utils import batchify_sentences
    dataset = BaseDataset.get_class_("ptb")()
    assert len(dataset.splits()["train"]) == 45438
    assert len(dataset.splits()["test"]) == 3761
    assert dataset.vocab_size == 10000

    inputs, targets = batchify_sentences(dataset.splits()["test"], 32)
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[1] == targets.shape[1] == 32

def test_infinite_get_callback():
    from aw_nas.utils.torch_utils import get_inf_iterator, SimpleDataset
    import torch
    import torch.utils.data
    dataset = torch.utils.data.DataLoader(SimpleDataset(([0,1,2,3], [1,2,3,4])),
                                          batch_size=2, num_workers=1)
    hiddens = [torch.rand(2) for _ in range(2)]
    ids = [id(hid) for hid in hiddens]
    callback = lambda: [hid.zero_() for hid in hiddens]

    queue = get_inf_iterator(dataset, callback)
    _ = next(queue)
    _ = next(queue)
    _ = next(queue) # should trigger callback
    assert all((hid == 0).all() for hid in hiddens), "hiddens should be reset"
    assert all(id_ == id(hid) for id_, hid in zip(ids, hiddens)), "hiddens should be reset in-place"

@pytest.mark.skipif(not AWNAS_TEST_SLOW, reason="parse dataset might be slow, by default not test")
def test_imagenet_sample_class():
    from aw_nas.dataset import BaseDataset
    dataset = BaseDataset.get_class_("imagenet")(load_train_only=True, num_sample_classes=20, random_choose=True)
    assert len(dataset.choosen_classes) == 20
