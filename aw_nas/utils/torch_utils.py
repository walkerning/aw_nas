import os
import copy
import math
import pickle
from contextlib import contextmanager

import six
import yaml

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from torch import distributed as dist

from aw_nas.utils.common_utils import AverageMeter
from aw_nas.utils.log import getLogger as _getLogger
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
            correct_k = correct[:k].reshape(-1).float().sum(0)
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
def sample_gumbel(shape, device, eps=1e-20):
    uniform_rand = torch.rand(shape).to(device)
    return Variable(-torch.log(-torch.log(uniform_rand + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=None):
    if eps is None:
        eps = sample_gumbel(logits.size(), logits.device)
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

# bernoulli sample
def relaxed_bernoulli_sample(logits, temperature):
    relaxed_bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature, logits=logits)
    y = relaxed_bernoulli.rsample()
    return y

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
                 residual_only=None, reverse=False):
        super(DenseGraphFlow, self).__init__()

        self.plus_I = plus_I
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.residual_only = residual_only
        self.reverse = reverse

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
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) \
                     + support
        else:
            # residual only the first `self.residual_only` nodes
            if self.residual_only == 0:
                residual = 0
            else:
                if self.reverse:
                    residual = torch.cat(
                        (torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                                      support.shape[2]], device=support.device),
                         support[:, -self.residual_only:, :]),
                        dim=1)
                else:
                    residual = torch.cat(
                        (support[:, :self.residual_only, :],
                         torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                                      support.shape[2]], device=support.device)),
                        dim=1)
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support)\
                     + residual

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
                 concat=None, has_aggregate_op=False, reverse=False):
        super(DenseGraphOpEdgeFlow, self).__init__()

        self.plus_I = plus_I
        self.share_self_op_emb = share_self_op_emb
        self.residual_only = residual_only
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.use_sum = use_sum
        self.reverse = reverse
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
            if self.reverse:
                res_output = torch.cat(
                    (torch.zeros([support.shape[0], support.shape[1],
                                  support.shape[2] - self.residual_only, support.shape[3]],
                                 device=support.device),
                     support[:, :, -self.residual_only:, :]
                    ),
                    dim=2)
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
                 residual_only=None, reverse=False,
                 concat=None, has_aggregate_op=False,
                 nonlinear="sigmoid", return_message=False,
                 skip_connection_index=None, attn_scale_factor=1.0):
        super(DenseGraphSimpleOpEdgeFlow, self).__init__()

        self.plus_I = plus_I
        self.share_self_op_emb = share_self_op_emb
        self.residual_only = residual_only
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.reverse = reverse
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.has_aggregate_op = has_aggregate_op
        self.return_message = return_message
        self.skip_connection_index = skip_connection_index # handle skip connection directlyy
        self.attn_scale_factor = attn_scale_factor

        if nonlinear == "sigmoid":
            self.nonlinear = torch.sigmoid
        elif nonlinear == "relu":
            self.nonlinear = torch.relu
        elif nonlinear == "softplus":
            self.nonlinear = nn.Softplus()
        else:
            assert nonlinear is None, "only support three types of nonlinear: sigmoid, relu, null"
            self.nonlinear = None

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
        if self.nonlinear is not None:
            attn = self.nonlinear(self.op_attention(op_emb))
        else:
            attn = self.op_attention(op_emb)
        attn = ((adj != 0).unsqueeze(-1).to(torch.float32)).detach() * attn * self.attn_scale_factor

        if self.skip_connection_index is not None:
            is_skip_mask = (adj == self.skip_connection_index)\
                           .unsqueeze(-1).to(torch.float32).detach()
            attn = is_skip_mask * attn.new(np.ones((adj.shape[-1], adj.shape[-1], attn.shape[-1])))\
                   + (1 - is_skip_mask) * attn

        if self.residual_only is None:
            res_output = support
        else:
            # residual only the first `self.residual_only` nodes
            # since there would be not inputs for the input nodes,
            # residual must be added to their output
            if self.reverse:
                res_output = torch.cat(
                    (torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                                  support.shape[2]],
                                 device=support.device),
                     support[:, -self.residual_only:, :]),
                    dim=1)
            else:
                res_output = torch.cat(
                    (support[:, :self.residual_only, :],
                     torch.zeros([support.shape[0], support.shape[1] - self.residual_only,
                                  support.shape[2]],
                                 device=support.device)), dim=1)

        processed_message = attn * support.unsqueeze(-3)
        processed_info = processed_message.sum(-2)
        if self.has_aggregate_op:
            output = self.aggregate_op(processed_info) + res_output
        else:
            output = processed_info + res_output

        if self.bias is not None:
            return output + self.bias
        else:
            if self.return_message:
                return output, processed_message
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

    def reset(self):
        self.iter_ = iter(self.iterable)
        # the old iterator should be garbage collected
        # and the old worker should be shut down

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

def prepare_data_queues(dataset, queue_cfg_lst, data_type="image", drop_last=False,
                        shuffle=False, shuffle_seed=None, num_workers=2, multiprocess=False,
                        shuffle_indice_file=None, pin_memory=True):
    """
    Further partition the dataset splits, prepare different data queues.

    Example::
    @TODO: doc
    """
    expect(data_type in {"image", "sequence"})

    dset_splits = dataset.splits()
    same_dset_mapping = dataset.same_data_split_mapping()
    dset_sizes = {n: len(d) for n, d in six.iteritems(dset_splits)}
    dset_indices = {n: list(range(size)) for n, size in dset_sizes.items()}
    # save cur np random seed state and apply data seed only for shuffling data
    # then restore seed for the rest of the process
    if shuffle:
        if shuffle_indice_file and os.path.exists(shuffle_indice_file):
            # load from `shuffle_indice_file`
            with open(shuffle_indice_file, "r") as r_f:
                dset_indices = yaml.load(r_f)
            _getLogger("aw_nas.torch_utils").info(
                "Load dataset split indices from %s", shuffle_indice_file)
        else:
            # shuffle using `shuffle_seed` as the random seed
            assert shuffle_seed is not None
            np_random_state = np.random.get_state()
            np.random.seed(shuffle_seed)
            [np.random.shuffle(indices) for indices in dset_indices.values()]
            np.random.set_state(np_random_state)
            if shuffle_indice_file:
                # dump the shuffled indices to `shuffle_indice_file`
                with open(shuffle_indice_file, "w") as w_f:
                    yaml.dump(dset_indices, w_f)
                _getLogger("aw_nas.torch_utils").info(
                    "Dump dataset split indices to %s", shuffle_indice_file)

    used_portions = {n: 0. for n in dset_splits}
    queues = []
    if isinstance(num_workers, int):
        num_workers = [num_workers] * len(queue_cfg_lst)
    if isinstance(pin_memory, bool):
        pin_memory = [pin_memory] * len(queue_cfg_lst)
    for cfg, worker_num, pin_memory_per_queue in zip(queue_cfg_lst, num_workers, pin_memory):
        # all the queues interleave sub-dataset
        batch_size = cfg["batch_size"]
        split = cfg["split"]
        portion = cfg["portion"]
        callback = cfg.get("callback", None)
        other_kwargs = cfg.get("kwargs", {})

        #if portion == 0:
        #    queues.append([])
        #    continue

        used_portion = used_portions[split]
        indices = dset_indices[same_dset_mapping.get(split, split)]
        size = dset_sizes[split]
        d_kwargs = getattr(dset_splits[split], "kwargs", {})
        group_sample = d_kwargs.pop("group_sample", False)
        if isinstance(portion, (list, tuple)) and len(portion) == 2:
            ranges = int(size * portion[0]), int(size * portion[1])
            # do not accumulate `used_portions` with range portion specification
            # used_portions[split] = portion[1]
        elif isinstance(portion, float) and 0. <= portion <= 1.:
            ranges = int(size * used_portion), int(size *(used_portion + portion))
            used_portions[split] += portion
        else:
            raise ValueError("Except portion to be a float between 0~1 or a list like ["
                             "left, right], got {} instead.".format(portion))
        subset_indices = indices[ranges[0]: ranges[1]]
        if data_type == "image":
            shuffle_queue = other_kwargs.get("shuffle", True)
            no_distributed_sampler = other_kwargs.pop("no_distributed_sampler", False)
            # by default, use shuffle=True for each queue in the search process
            # can be overrided use kwargs
            kwargs = {
                "batch_size": batch_size,
                "pin_memory": pin_memory_per_queue,
                "num_workers": worker_num,
                "drop_last": drop_last,
                "timeout": 0,
            }
            if not shuffle_queue:
                # choose a subset of the dataset, and do not shuffle
                dataset_split = torch.utils.data.Subset(dset_splits[split], subset_indices)
                if multiprocess and not no_distributed_sampler:
                    # for multiprocess (distributed) and no-shuffle data queue
                    kwargs["sampler"] = DistributedSampler(dataset_split, shuffle=False)
                elif multiprocess and group_sample:
                    raise ValueError("shuffle_queue must be True when using group sampler.")
            else:
                # use subset random samplers
                dataset_split = dset_splits[split]
                if not group_sample:
                    kwargs["sampler"] = torch.utils.data.SubsetRandomSampler(subset_indices) \
                                    if not multiprocess or no_distributed_sampler else \
                                       CustomDistributedSampler(dataset_split, subset_indices)
                else:
                    kwargs["sampler"] = GroupSampler(dataset_split, subset_indices, batch_size) \
                                        if not multiprocess or no_distributed_sampler \
                                        else DistributedGroupSampler(dataset_split, subset_indices,
                                                batch_size)
            kwargs.update(d_kwargs) # first update dataset-specific kwargs
            kwargs.update(other_kwargs) # then update queue-specific kwargs
            queue = get_inf_iterator(torch.utils.data.DataLoader(dataset_split, **kwargs), callback)
        else: # data_type == "sequence"
            expect("bptt_steps" in cfg)
            bptt_steps = cfg["bptt_steps"]
            dataset = SimpleDataset(
                batchify_sentences(
                    dset_splits[split][ranges[0]: ranges[1]],
                    batch_size)
            )
            kwargs = {
                "batch_size": bptt_steps,
                "num_workers": 0,
                "shuffle": False
            }
            # update the dataset/queue-specific kwargs
            kwargs.update(d_kwargs)
            kwargs.update(other_kwargs)
            queue = get_inf_iterator(torch.utils.data.DataLoader(
                dataset, **kwargs), callback)

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

def count_parameters(model, count_binary=False):
    if not count_binary:
        return sum(p.nelement() for name, p in model.named_parameters() if "auxiliary" not in name)

    # For binary search
    params = 0
    bi_params = 0
    from aw_nas import ops
    for _, module in model.named_modules():
        if isinstance(module, ops.BinaryConv2d):
            for _, p in module.named_parameters():
                bi_params += p.nelement()
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            for _, p in module.named_parameters():
                params += p.nelement()
    return np.array([params, bi_params])

def _to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (tuple, list)):
        return [_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return torch.tensor(data).to(device)
    else:
        return data

def to_device(datas, device):
    return [_to_device(data, device) for data in datas]


class CustomDistributedSampler(DistributedSampler):
    """
    This sampler is the mix of SubsetSampler and DistributedSampler.
    Because DistributedSampler does not support sample a subset of dataset,
    which is required by function `prepare_data_queues` during the search process.
    """
    def __init__(self, dataset, indices, *args, **kwargvs):
        super(CustomDistributedSampler, self).__init__(dataset, *args, **kwargvs)
        self.indices = indices
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = [self.indices[i] for i in torch.randperm(len(self.indices), generator=g)]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, indices, samples_per_gpu=1):
        if hasattr(dataset, 'group_index'):
            self.flag = dataset.group_index.astype(np.int64)
        else:
            self.flag = np.zeros(len(dataset), np.int64)

        self.indices = indices or list(range(len(dataset)))
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, _ in enumerate(self.group_sizes):
            indice = np.where(self.flag[self.indices] == i)[0]
            size = len(indice)
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag[self.indices] == i)[0]
            size = len(indice)
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 indices=None,
                 samples_per_gpu=1):
        super(DistributedGroupSampler, self).__init__(dataset)
        self.dataset = dataset
        self.indices = indices or list(range(len(dataset)))
        self.samples_per_gpu = samples_per_gpu
        self.epoch = 0

        if hasattr(self.dataset, "group_index"):
            self.flag = self.dataset.group_index
        else:
            self.flag = np.zeros(len(self.dataset), dtype=np.uint8)
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            indice = np.where(self.flag[self.indices] == i)[0]
            size = len(indice)
            self.num_samples += int(
                math.ceil(size * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag[self.indices] == i)[0]
                size = len(indice)
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1

def collect_results_gpu(result_part):
    """
    Adopted from MMDetection: https://github.com/open-mmlab/mmdetection
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        #with open("/tmp/xxx.pkl", "wb") as fw:
        #    pickle.dump(part_list, fw)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        # ordered_results = ordered_results
        return ordered_results


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


# ------ for bn calibration ------
def accumulate_bn(inputs, running_means, running_vars):
    batch_size = inputs.shape[0]
    batch_mean = inputs.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
    batch_var = (inputs - batch_mean) * (inputs - batch_mean)
    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

    batch_mean = torch.squeeze(batch_mean)
    batch_var = torch.squeeze(batch_var)

    running_means.update(batch_mean.data, batch_size)
    running_vars.update(batch_var.data, batch_size)
    return batch_mean, batch_var

def calib_bn(model, data_queue, max_inputs=2000):
    from aw_nas.ops import FlexibleBatchNorm2d
    """
    Adopted from once-for-all.
    """
    num_inputs = 0
    bn_running_mean = {}
    bn_running_var = {}
    data_queue = iter(data_queue)

    new_model = copy.deepcopy(model)
    forward_model = copy.deepcopy(model)

    def forward_factory(bn, running_means, running_vars):
        def forward(inputs):
            batch_mean, batch_var = accumulate_bn(inputs, running_means, running_vars)
            if isinstance(bn, nn.BatchNorm2d):
                # forward method of an instance of nn.BatchNorm2d in FlexibleBatchNorm will not be called
                return F.batch_norm(
                        inputs, batch_mean, batch_var, bn.weight,
                        bn.bias, False,
                        0.0, bn.eps,
                    )
            elif isinstance(bn, FlexibleBatchNorm2d):
                return bn.forward_mask(inputs, bn.mask)
            else:
                raise ValueError
        return forward

    for name, m in forward_model.named_modules():
        if not isinstance(m, (nn.BatchNorm2d, FlexibleBatchNorm2d)):
            continue
        bn_running_mean[name] = AverageMeter()
        bn_running_var[name] = AverageMeter()
        m.forward = forward_factory(m, bn_running_mean[name], bn_running_var[name])

    with torch.no_grad():
        num_inputs = 0
        for inputs, _ in data_queue:
            inputs = inputs.to(forward_model.get_device())
            forward_model(inputs)
            num_inputs += inputs.shape[0]
            if num_inputs > max_inputs:
                break

    for name, m in new_model.named_modules():
        if name in bn_running_mean and not bn_running_mean[name].is_empty():
            feature_dim = bn_running_mean[name].avg.shape[0]
            assert isinstance(m, (nn.BatchNorm2d, FlexibleBatchNorm2d))
            if hasattr(m, "flex_bn"):
                m = m.flex_bn
            m.running_mean.data[:feature_dim].copy_(bn_running_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_running_var[name].avg)
    return new_model


# higher order grad utils
# https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0

    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x, flatten=True):
    if flatten:
        n = x.nelement()
        return jacobian(jacobian(y, x, create_graph=True), x).view(n, n)

    return jacobian(jacobian(y, x, create_graph=True), x)


def max_eig_of_hessian(y, x):
    max_eig = torch.eig(hessian(y, x))[0].norm(dim=1).max()

    return max_eig.item()


def random_cnn_data(device="cuda", batch_size=2, shape=28, input_c=3, output_c=10):
    return (
        torch.rand(batch_size, input_c, shape, shape, dtype=torch.float, device=device),
        torch.tensor(np.random.randint(0, high=output_c, size=batch_size)).long().to(device),
    )
