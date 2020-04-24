import math
from contextlib import contextmanager

import six
import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler

from aw_nas.utils.exception import expect
from aw_nas.utils.lr_scheduler import get_scheduler_cls


## --- model/training utils ---
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

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = x.new(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
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
    mask = torch.floor(torch.rand(batch_size, dim, device=device) + keep_prob) / keep_prob
    return mask

def submodule_named_members(module, member, prefix, not_include=tuple()):
    for mod_name, mod in six.iteritems(module._modules): #pylint: disable=protected-access
        if mod_name in not_include:
            continue
        _func = getattr(mod, "named_" + member)
        for n, v in _func(prefix=prefix+mod_name):
            yield n, v

def substitute_params(module, params, prefix=""):
    prefix = (prefix + ".") if prefix else ""
    for n in module._parameters:
        if prefix + n in params:
            module._parameters[n] = params[prefix + n]

@contextmanager
def use_params(module, params):
    backup_params = dict(module.named_parameters())
    for mod_prefix, mod in module.named_modules():
        substitute_params(mod, params, prefix=mod_prefix)
    yield
    for mod_prefix, mod in module.named_modules():
        substitute_params(mod, backup_params, prefix=mod_prefix)

class DenseGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Reference:
    https://github.com/tkipf/pygcn/blob/88c6676b2ab98b04bf3bef96b46ea037ebb07b12/pygcn/layers.py
    Dense matrix multiply for batching
    """

    def __init__(self, in_features, out_features, plus_I=False, normalize=False, bias=True):
        super(DenseGraphConvolution, self).__init__()

        self.plus_I = plus_I
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        if self.plus_I:
            adj_aug = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
            if self.normalize:
                degree_invsqrt = 1. / adj_aug.sum(dim=-1).float().sqrt()
                degree_norm = degree_invsqrt.unsqueeze(2) * degree_invsqrt.unsqueeze(1)
                adj_aug = degree_norm * adj_aug
        else:
            adj_aug = adj
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj_aug, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DenseGraphFlow(nn.Module):

    def __init__(self, in_features, out_features, op_emb_dim,
                 has_attention=True, plus_I=False, normalize=False, bias=True,
                 residual_only=None):
        super(DenseGraphFlow, self).__init__()

        self.plus_I = plus_I
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.residual_only = residual_only
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if has_attention:
            self.op_attention = nn.Linear(op_emb_dim, out_features)
        else:
            assert self.op_emb_dim == self.out_features
            self.op_attention = nn.Identity()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, op_emb):
        if self.plus_I:
            adj_aug = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
            if self.normalize:
                degree_invsqrt = 1. / adj_aug.sum(dim=-1).float().sqrt()
                degree_norm = degree_invsqrt.unsqueeze(2) * degree_invsqrt.unsqueeze(1)
                adj_aug = degree_norm * adj_aug
        else:
            adj_aug = adj
        support = torch.matmul(inputs, self.weight)
        if self.residual_only is None:
            # use residual
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) + support
        else:
            # residual only the first `self.residual_only` nodes
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) + torch.cat(
                (support[:, :self.residual_only, :],
                 torch.zeros([support.shape[0], support.shape[1] - self.residual_only, support.shape[2]], device=support.device)),
                dim=1)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DenseGraphOpEdgeFlow(nn.Module):
    """
    For search space that has operation on the edge.
    """
    def __init__(self, in_features, out_features, op_emb_dim,
                 has_attention=True, plus_I=False, share_self_op_emb=False,
                 normalize=False, bias=False,
                 residual_only=None, use_sum=False,
                 concat=None, has_aggregate_op=False):
        super(DenseGraphOpEdgeFlow, self).__init__()

        self.plus_I = plus_I
        self.share_self_op_emb = share_self_op_emb
        self.residual_only = residual_only
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.use_sum = use_sum
        # self.concat = concat
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.has_aggregate_op = has_aggregate_op
        if self.has_aggregate_op:
            self.aggregate_op = nn.Linear(out_features, out_features)
        if has_attention:
            self.op_attention = nn.Linear(op_emb_dim, out_features)
        else:
            assert self.op_emb_dim == self.out_features
            self.op_attention = nn.Identity()
        if self.plus_I and not self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(op_emb_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        # if self.concat is not None:
        #     assert isinstance(self.concat, int)
        #     self.concats
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, adj_op_inds_lst, op_emb, zero_index, self_op_emb=None):
        # if self.plus_I:
        #     adj_aug = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
        # else:
        #     adj_aug = adj
        # if self.normalize:
        #     # degree_invsqrt = 1. / (adj_aug.sum(dim=-1).float() + 1e-8).sqrt()
        #     # degree_norm = degree_invsqrt.unsqueeze(-1) * degree_invsqrt.unsqueeze(-2)
        #     # adj_aug = degree_norm * adj_aug
        #     num_input = adj_aug.sum(dim=-1, keepdim=True)
        #     adj_aug = torch.where(num_input > 0, adj_aug / (num_input + 1e-8), adj_aug)

        # support: (b, n_cg, V, h_i)
        support = torch.matmul(inputs, self.weight)
        op_emb_adj_lst = [F.embedding(adj_op_inds, op_emb) for adj_op_inds in adj_op_inds_lst]
        attn_mask_inds_lst = [(adj_op_inds == zero_index).unsqueeze(-1)
                              for adj_op_inds in adj_op_inds_lst]
        if self.plus_I:
            eye_mask = support.new(np.eye(adj.shape[-1])).unsqueeze(-1).bool()
            # for i in range(len(adj_op_inds_lst)):
            #     op_emb_adj_lst[i] = torch.where(eye_mask, self.self_op_emb, op_emb_adj_lst[i])
            #     attn_mask_inds_lst[i] = attn_mask_inds_lst[i] & (~eye_mask.bool())
            self_op_emb = self_op_emb if self.share_self_op_emb else self.self_op_emb
            op_emb_adj_lst[0] = torch.where(eye_mask, self_op_emb, op_emb_adj_lst[0])
            attn_mask_inds_lst[0] = attn_mask_inds_lst[0] & (~eye_mask)

        # attn_mask_inds_stack: (n_d, b, n_cg, V, V, 1)
        attn_mask_inds_stack = torch.stack(attn_mask_inds_lst)
        # ob_emb_adj_stack: (n_d, b, n_cg, V, V, h_o)
        op_emb_adj_stack = torch.stack(op_emb_adj_lst)

        attn = torch.sigmoid(self.op_attention(op_emb_adj_stack))
        attn = torch.where(
            attn_mask_inds_stack,
            attn.new(1, 1, 1, 1, 1, attn.shape[-1]).zero_(),
            attn)
        # attn: (n_d, b, n_cg, V, V, h_o)

        # output = (adj_aug.unsqueeze(-1) * attn \
        #           * support.unsqueeze(2)).sum(-2) + support
        if self.residual_only is None:
            res_output = support
        else:
            res_output = torch.cat(
                (support[:, :, :self.residual_only, :],
                 torch.zeros([support.shape[0], support.shape[1],
                              support.shape[2] - self.residual_only, support.shape[3]],
                             device=support.device)),
                dim=2)
        processed_info = (attn * support.unsqueeze(2)).sum(-2)
        processed_info = processed_info.sum(0) if self.use_sum else processed_info.mean(0)
        if self.has_aggregate_op:
            output = self.aggregate_op(processed_info) + res_output
        else:
            output = processed_info + res_output
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # def forward(self, inputs, adj, adj_op_inds, op_emb, zero_index):
    #     if self.plus_I:
    #         adj_aug = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
    #     else:
    #         adj_aug = adj
    #     if self.normalize:
    #         # degree_invsqrt = 1. / (adj_aug.sum(dim=-1).float() + 1e-8).sqrt()
    #         # degree_norm = degree_invsqrt.unsqueeze(-1) * degree_invsqrt.unsqueeze(-2)
    #         # adj_aug = degree_norm * adj_aug
    #         num_input = adj_aug.sum(dim=-1, keepdim=True)
    #         adj_aug = torch.where(num_input > 0, adj_aug / (num_input + 1e-8), adj_aug)
    #     support = torch.matmul(inputs, self.weight)
    #     op_emb_adj = F.embedding(adj_op_inds, op_emb)
    #     attn_mask_inds = (adj_op_inds == zero_index).unsqueeze(-1)
    #     if self.plus_I:
    #         eye_mask = attn_mask_inds.new(np.eye(adj.shape[-1])).unsqueeze(-1)

    #         op_emb_adj = torch.where(eye_mask, self.self_op_emb, op_emb_adj)
    #         attn_mask_inds = attn_mask_inds & (~eye_mask.bool())

    #     attn = torch.sigmoid(self.op_attention(op_emb_adj))
    #     attn = torch.where(
    #         attn_mask_inds,
    #         attn.new(1, 1, 1, 1, attn.shape[-1]).zero_(),
    #         attn)
    #     output = (adj_aug.unsqueeze(-1) * attn \
    #               * support.unsqueeze(2)).sum(-2) + support
    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DenseGraphSimpleOpEdgeFlow(nn.Module):
    """
    For search space that has operation on the edge.
    SimpleOpEdgeFlow is applicable to the search space in which no double connection
    between nodes exists.
    """
    def __init__(self, in_features, out_features, op_emb_dim,
                 has_attention=True, plus_I=False, share_self_op_emb=False,
                 normalize=False, bias=False,
                 residual_only=None,
                 concat=None, has_aggregate_op=False):
        super(DenseGraphSimpleOpEdgeFlow, self).__init__()
        
        self.plus_I = plus_I
        self.share_self_op_emb = share_self_op_emb
        self.residual_only = residual_only
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.has_aggregate_op = has_aggregate_op
        if self.has_aggregate_op:
            # TODO: a non linear aggregate op can brought more representation ability?
            # but no nonlinear aggreate op that have good interpreatability
            # occurs to me for now..
            # how the aggregation of different information dimensions
            # differ? maybe can only use things like... sum, powered sum
            # differ in the degree of emphasizing ``stronger'' info..
            self.aggregate_op = nn.Linear(out_features, out_features)
        if has_attention:
            self.op_attention = nn.Linear(op_emb_dim, out_features)
        else:
            assert self.op_emb_dim == self.out_features
            self.op_attention = nn.Identity()
        if self.plus_I and not self.share_self_op_emb:
            self.self_op_emb = nn.Parameter(torch.FloatTensor(op_emb_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, op_emb, self_op_emb=None):
        # support: (b, V, h_i)
        support = torch.matmul(inputs, self.weight)

        if self.plus_I:
            eye_mask = support.new(np.eye(adj.shape[-1]))
            # for i in range(len(adj_op_inds_lst)):
            #     op_emb_adj_lst[i] = torch.where(eye_mask, self.self_op_emb, op_emb_adj_lst[i])
            #     attn_mask_inds_lst[i] = attn_mask_inds_lst[i] & (~eye_mask.bool())
            self_op_emb = self_op_emb if self.share_self_op_emb else self.self_op_emb
            op_emb = torch.where(eye_mask.unsqueeze(-1).to(torch.bool), self_op_emb, op_emb)
            adj = adj + eye_mask.to(torch.long)

        # attn: (b, V, V, h_i)
        attn = torch.sigmoid(self.op_attention(op_emb))
        attn = ((adj != 0).unsqueeze(-1).to(torch.float32)).detach() * attn

        if self.residual_only is None:
            res_output = support
        else:
            res_output = torch.cat(
                (support[:, :self.residual_only, :],
                 torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                              support.shape[2]],
                             device=support.device)), dim=1)

        processed_info = (attn * support.unsqueeze(-3)).sum(-2)
        if self.has_aggregate_op:
            output = self.aggregate_op(processed_info) + res_output
        else:
            output = processed_info + res_output
        if self.bias is not None:
            return output + self.bias
        else:
            return output

## --- dataset ---
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.target = data[1]

    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        return data, target

    def __len__(self):
        return len(self.data)

def batchify_sentences(data, bsz, device="cuda"):
    data = torch.cat(data, -1)
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if "cuda" in device:
        data = data.cuda()
    return data[:-1, :], data[1:, :]

class InfIterator(six.Iterator):
    def __init__(self, iterable, callbacks=()):
        self.iterable = iterable
        self.iter_ = None
        self.callbacks = list(callbacks)

    def __getattr__(self, name):
        return getattr(self.iterable, name)

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        if self.iter_ is None:
            self.iter_ = iter(self.iterable)
        try:
            data = next(self.iter_)
        except StopIteration:
            if self.callbacks:
                [callback() for callback in self.callbacks if callback is not None]
            self.iter_ = iter(self.iterable)
            data = next(self.iter_)
        # except RuntimeError as e:
        #     self.logger.error(e)
        #     raise
        return data

    def add_callback(self, callback):
        assert callable(callback)
        self.callbacks.append(callback)

    next = __next__

def get_inf_iterator(iterable, callback):
    return InfIterator(iterable, [callback])

def prepare_data_queues(splits, queue_cfg_lst, data_type="image", drop_last=False,
                        shuffle=True, num_workers=2):
    """
    Further partition the dataset splits, prepare different data queues.

    Example::
    @TODO: doc
    """
    expect(data_type in {"image", "sequence"})

    dset_splits = splits
    dset_sizes = {n: len(d) for n, d in six.iteritems(dset_splits)}
    dset_indices = {n: list(range(size)) for n, size in dset_sizes.items()}
    if shuffle:
        [np.random.shuffle(indices) for indices in dset_indices.values()]
    used_portions = {n: 0. for n in splits}
    queues = []
    for cfg in queue_cfg_lst: # all the queues interleave sub-dataset
        batch_size = cfg["batch_size"]
        split = cfg["split"]
        portion = cfg["portion"]
        callback = cfg.get("callback", None)

        if portion == 0:
            queues.append(None)
            continue

        used_portion = used_portions[split]
        indices = dset_indices[split]
        size = dset_sizes[split]
        if data_type == "image":
            kwargs = {
                "batch_size": batch_size,
                "pin_memory": True,
                "num_workers": num_workers,
                "sampler": torch.utils.data.SubsetRandomSampler(
                    indices[int(size*used_portion):int(size*(used_portion+portion))]),
                "drop_last": drop_last,
                "timeout": 10
            }
            queue = get_inf_iterator(torch.utils.data.DataLoader(
                dset_splits[split], **kwargs), callback)
        else: # data_type == "sequence"
            expect("bptt_steps" in cfg)
            bptt_steps = cfg["bptt_steps"]
            dataset = SimpleDataset(
                batchify_sentences(
                    dset_splits[split][int(size*used_portion):int(size*(used_portion+portion))],
                    batch_size)
            )
            kwargs = {
                "batch_size": bptt_steps,
                "pin_memory": False,
                "num_workers": 0,
                "shuffle": False
            }
            queue = get_inf_iterator(torch.utils.data.DataLoader(
                dataset, **kwargs), callback)

        used_portions[split] += portion
        queues.append(queue)

    return queues

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


## --- init from configs ---
def init_optimizer(params, cfg):
    if cfg:
        cfg = {k:v for k, v in six.iteritems(cfg)}
        opt_cls = getattr(optim, cfg.pop("type"))
        return opt_cls(params, **cfg)
    return None

def init_scheduler(optimizer, cfg):
    if cfg and optimizer is not None:
        cfg = {k:v for k, v in six.iteritems(cfg)}
        sch_cls = get_scheduler_cls(cfg.pop("type"))
        return sch_cls(optimizer, **cfg)
    return None

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# def _new_step_tensor(scheduler, epoch=None):
#     scheduler.ori_step(epoch)
#     scheduler.scheduled_tensor[:] = scheduler.get_lr()[0]

# def _patch_step_tensor(scheduler, tensor):
#     scheduler.ori_step = scheduler.step
#     scheduler.scheduled_tensor = tensor
#     scheduler.step = _new_step_tensor.__get__(scheduler)

class TensorScheduler(_LRScheduler):
    def __init__(self, tensor, inner_scheduler_cfg, last_epoch=-1):
        mock_optimizer = torch.optim.SGD([torch.zeros(1)], lr=tensor.item())
        self.tensor = tensor
        scheduler_cls = get_scheduler_cls(inner_scheduler_cfg.pop("type"))
        self.inner_scheduler = scheduler_cls(mock_optimizer, **inner_scheduler_cfg)
        super(TensorScheduler, self).__init__(mock_optimizer, last_epoch)

    def get_lr(self):
        return self.inner_scheduler.get_lr()

    def step(self, epoch=None):
        super(TensorScheduler, self).step(epoch)
        self.inner_scheduler.step(epoch)
        self.tensor[:] = self.get_lr()[0]

def init_tensor_scheduler(tensor, cfg):
    if cfg is not None:
        return TensorScheduler(tensor, cfg)
    return None

## --- misc helpers ---
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

def count_parameters(model):
    return sum(p.nelement() for name, p in model.named_parameters() if "auxiliary" not in name)

def to_device(datas, device):
    return [data.to(device) for data in datas]
