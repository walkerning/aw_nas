import six
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.target = data[1]

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        return data, target

    def __len__(self):
        return len(self.data)

def batchify_sentences(data, bsz):
    data = torch.cat(data, -1)
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data[:-1, :], data[1:, :]

class InfIterator(six.Iterator):
    def __init__(self, iterable, callback):
        self.iterable = iterable
        self.iter_ = iter(self.iterable)
        self.callback = callback

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        try:
            data = next(self.iter_)
        except StopIteration:
            if self.callback is not None:
                self.callback()
            self.iter_ = iter(self.iterable)
            data = next(self.iter_)
        return data

    next = __next__

def get_inf_iterator(iterable, callback):
    return InfIterator(iterable, callback)

def accuracy(outputs, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Cutout(object):
    """
    Cutout randomized rectangle of size (length, length)
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        h, w = image.shape[1:]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(h)
        y_1 = np.clip(y - self.length // 2, 0, h)
        y_2 = np.clip(y + self.length // 2, 0, h)
        x_1 = np.clip(x - self.length // 2, 0, w)
        x_2 = np.clip(x + self.length // 2, 0, w)
        mask[y_1:y_2, x_1:x_2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image *= mask
        return image

def get_variable(inputs, device, **kwargs):
    if isinstance(inputs, (list, np.ndarray)):
        inputs = torch.Tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out

def get_numpy(arr):
    if isinstance(arr, (torch.Tensor, Variable)):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.array(arr)
    return arr

def count_parameters_in_MB(model): #pylint: disable=invalid-name
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() \
                  if "auxiliary" not in name)/1e6

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

# gumbel softmax
def sample_gumbel(shape, eps=1e-20):
    uniform_rand = torch.rand(shape).cuda()
    return Variable(-torch.log(-torch.log(uniform_rand + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=None):
    if eps is None:
        eps = sample_gumbel(logits.size())
    y = logits + eps
    return F.softmax(y / temperature, dim=-1), eps

def gumbel_softmax(logits, temperature, eps=None, hard=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y, eps = gumbel_softmax_sample(logits, temperature, eps)
    if hard:
        y = straight_through(y)
    return y, eps

def straight_through(y):
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def mask2d(batch_size, dim, keep_prob, device):
    mask = torch.floor(torch.rand(batch_size, dim) + keep_prob) / keep_prob
    return mask.to(device)

def submodule_named_members(module, member, prefix, not_include=tuple()):
    for mod_name, mod in six.iteritems(module._modules): #pylint: disable=protected-access
        if mod_name in not_include:
            continue
        _func = getattr(mod, "named_" + member)
        for n, v in _func(prefix=prefix+mod_name):
            yield n, v
