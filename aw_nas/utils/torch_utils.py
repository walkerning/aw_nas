import six
import numpy as np
import torch
from torch.autograd import Variable

class InfIterator(six.Iterator):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iter_ = iter(self.iterable)

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        try:
            data = next(self.iter_)
        except StopIteration:
            self.iter_ = iter(self.iterable)
            data = next(self.iter_)
        return data

    next = __next__

def get_inf_iterator(iterable):
    return InfIterator(iterable)

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
        arr = arr.cpu().numpy()
    else:
        arr = np.array(arr)
    return arr
