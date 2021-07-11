# -*- coding: utf-8 -*-
import copy
import inspect
from inspect import signature
import os
import pickle
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import optim

import numpy as np
import yaml

try:
    from sklearn import linear_model
    from sklearn.neural_network import MLPRegressor
except ImportError as e:
    from aw_nas.utils import getLogger
    getLogger("hardware").warn(
        ("Cannot import module hardware.utils: {}\n"
         "Should install scikit-learn to make some hardware-related"
         " functionalities work").format(e))


from aw_nas.hardware.base import (
    BaseHardwarePerformanceModel,
    MixinProfilingSearchSpace,
    Preprocessor
)
from aw_nas.ops import get_op

Prim_ = namedtuple(
    "Prim",
    ["prim_type", "spatial_size", "C", "C_out", "stride", "affine", "kwargs"],
)


class Prim(Prim_):
    def __new__(cls, prim_type, spatial_size, C, C_out, stride, affine, **kwargs):
        position_params = ["C", "C_out", "stride", "affine"]
        prim_constructor = get_op(prim_type)
        prim_sig = signature(prim_constructor)
        params = prim_sig.parameters
        for name, param in params.items():
            if param.default != inspect._empty:
                if name in position_params:
                    continue
                if kwargs.get(name) is None:
                    kwargs[name] = param.default
            else:
                assert name in position_params or name in kwargs, \
                    "{} is a non-default parameter which should be provided explicitly.".format(
                    name)

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        assert set(params.keys()) == set(
            position_params + list(kwargs.keys())),\
            ("The passed parameters are different from the formal parameter list of primitive "
             "type `{}`, expected {}, got {}").format(
                 prim_type,
                 str(params.keys()),
                 str(position_params + list(kwargs.keys()))
             )

        kwargs = tuple(
            sorted([(k, v) for k, v in kwargs.items()]))
        return super(Prim, cls).__new__(
            cls,
            prim_type,
            int(spatial_size),
            int(C),
            int(C_out),
            int(stride),
            affine,
            kwargs,
        )

    def _asdict(self):
        origin_dict = dict(super(Prim, self)._asdict())
        kwargs = origin_dict.pop("kwargs")
        origin_dict.update(dict(kwargs))
        return origin_dict

    def __getnewargs_ex__(self):
        return tuple(), self._asdict()

def assemble_profiling_nets_from_file(fname,
                                      base_cfg_fname,
                                      image_size=224,
                                      sample=None,
                                      max_layers=20):
    with open(fname, "r") as f:
        prof_prims = yaml.load(f)
    with open(base_cfg_fname, "r") as f:
        base_cfg = yaml.load(f)
    return assemble_profiling_nets(prof_prims, base_cfg, image_size, sample,
                                   max_layers)


def sample_networks(mixin_search_space,
                    base_cfg_template,
                    num_sample,
                    **kwargs):
    for _ in range(num_sample):
        rollout = mixin_search_space.random_sample()
        primitives = mixin_search_space.rollout_to_primitives(rollout,
                                                **kwargs)
        base_cfg_template["final_model_cfg"]["genotypes"] = [
            p._asdict() for p in primitives
        ]
        yield copy.deepcopy(base_cfg_template)

def assemble_profiling_nets(profiling_primitives,
                            base_cfg_template,
                            num_sample=None,
                            image_size=224,
                            max_layers=20,
                            fix_stem_layer=True):
    """
    Args:
        profiling_primitives: (list of dict)
            possible keys: prim_type, spatial_size, C, C_out, stride, primitive_kwargs
            (Don't use dict and list that is unhashable. Use tuple instead: (key, value) )
        base_cfg_template: (dict) final configuration template
        image_size: (int) the inputs size
        sample: (int) the number of nets
        max_layers: (int) the number of max layers of each net (glue layers do not count)

    Returns:
        a generator of yaml configs

    This function assembles all profiling primitives into multiple networks, which takes a several steps:
    1. Each network has a stride=2 layer as the first conv layer (like many convolution network, in order to reduce the size of feature map.)
    2. Find a available primitive for current spatial_size and current channel number:
        a). If there is a primitive has exactly same channel number and spatial size with previous primitive, append it to genotype;
        b). else we select a primitive which has the same or smaller spatial size, and insert a glue layer between them to make the number of channels consistant.
    3. Iterate profiling primitives until there is not available primitive or the number of genotype's layers exceeds the max layer.
    """

    if num_sample is None:
        num_sample = np.inf

    # genotypes: "[prim_type, *params], [] ..."
    profiling_primitives = [prim for prim in profiling_primitives if prim["C"] != 3]
    profiling_primitives = sorted(profiling_primitives,
                                  key=lambda x: (x["C"], x["stride"]))
    ith_arch = 0
    glue_layer = lambda spatial_size, C, C_out, stride=1: {
        "prim_type": "conv_1x1",
        "spatial_size": spatial_size,
        "C": C,
        "C_out": C_out,
        "stride": stride,
        "affine": True,
    }

    # use C as the index of df to accelerate query
    available_idx = list(range(len(profiling_primitives)))
    channel_to_idx = {}
    for i, prim in enumerate(profiling_primitives):
        channel_to_idx[prim["C"]] = channel_to_idx.get(prim["C"], []) + [i]
    channel_to_idx = {k: set(v) for k, v in channel_to_idx.items()}

    while len(available_idx) > 0 and ith_arch < num_sample:
        ith_arch += 1
        geno = []

        # the first conv layer reduing the size of feature map.
        sampled_prim = profiling_primitives[available_idx[0]]
        cur_channel = int(sampled_prim["C"])
        if fix_stem_layer:
            first_cov_op = {
                "prim_type": "conv_3x3",
                "spatial_size": image_size,
                "C": 3,
                "C_out": cur_channel,
                "stride": 2,
                "affine": True,
            }
            geno.append(first_cov_op)
        
        cur_size = round(image_size / 2)

        for _ in range(max_layers):
            if len(available_idx) == 0:
                break
            try:
                # find layer which has exactly same channel number and spatial size with the previous one
                idx = channel_to_idx[cur_channel]
                if len(idx) == 0:
                    raise ValueError
                for i in idx:
                    sampled_prim = profiling_primitives[i]
                    if sampled_prim["spatial_size"] == cur_size:
                        break
                else:
                    raise ValueError
            except:
                # or find a layer which has arbitrary channel number but has smaller spatial size
                # we need to assure that spatial size decreases as the layer number (or upsample layer will be needed.)
                for i in available_idx:
                    if profiling_primitives[i]["spatial_size"] <= cur_size:
                        sampled_prim = profiling_primitives[i]
                        break
                else:
                    break

                out_channel = int(sampled_prim["C"])
                spatial_size = int(sampled_prim["spatial_size"])
                stride = int(round(cur_size / spatial_size))
                assert isinstance(
                    stride, int) and stride > 0, "stride: {stride}".format(
                        stride=stride)
                glue_conv = glue_layer(cur_size, cur_channel, out_channel, min(stride, 2))
                geno.append(glue_conv)

                if stride > 2:
                    _stride = stride // 2
                    _cur_size = int(round(cur_size / 2))
                    _channel = out_channel
                    while _stride > 1:
                        glue_conv = glue_layer(_cur_size, _channel, _channel,
                                2)
                        geno.append(glue_conv)
                        _stride //= 2
                        _cur_size //= 2

            cur_channel = int(sampled_prim["C_out"])
            cur_size = int(
                round(sampled_prim["spatial_size"] / sampled_prim["stride"]))

            available_idx.remove(i)
            channel_to_idx[sampled_prim["C"]].remove(i)

            geno.append(sampled_prim)

        base_cfg_template["final_model_cfg"]["genotypes"] = geno
        yield copy.deepcopy(base_cfg_template)


class BlockSumPreprocessor(Preprocessor):
    NAME = "block_sum"

    def __init__(self, preprocessors=None, schedule_cfg=None):
        super().__init__(preprocessors, schedule_cfg)

    def __call__(self, unpreprocessed, **kwargs):
        for prof_net in unpreprocessed:
            for ith_prof in prof_net:
                block_sum = {}
                for prim in ith_prof["primitives"]:
                    for k, perf in prim["performances"].items():
                        block_sum[k] = block_sum.get(k, 0.) + perf
                for k, perf in block_sum.items():
                    ith_prof["block_sum_{}".format(k)] = block_sum[k]
            yield prof_net


class FlattenPreprocessor(Preprocessor):
    NAME = "flatten"

    def __init__(self, preprocessors=None, schedule_cfg=None):
        super().__init__(preprocessors, schedule_cfg)

    def __call__(self, unpreprocessed, **kwargs):
        for prof_net in unpreprocessed:
            for ith_prof in prof_net:
                yield ith_prof


class RemoveAnomalyPreprocessor(Preprocessor):
    NAME = "remove_anomaly"

    def __init__(self, preprocessors=None, schedule_cfg=None):
        super().__init__(preprocessors, schedule_cfg)

    def __call__(self, unpreprocessed, **kwargs):
        is_training = kwargs.get("is_training", True)
        if not is_training:
            for net in unpreprocessed:
                yield net
        tolerance_std = kwargs.get("tolerance_std", 0.1)
        for prof_net in unpreprocessed:
            # FIXME: assert every primitive has the performance keys.
            perf_keys = prof_net[0]["primitives"][0]["performances"].keys()
            block_sum_avg = {
                k: np.mean([
                    ith_prof["block_sum_{}".format(k)] for ith_prof in prof_net
                ])
                for k in perf_keys
            }
            filtered_net = []
            for ith_prof in prof_net:
                for k in perf_keys:
                    if abs(ith_prof["block_sum_{}".format(k)] -
                           block_sum_avg[k]
                           ) > block_sum_avg[k] * tolerance_std:
                        break
                else:
                    filtered_net += [ith_prof]
            yield filtered_net


class ExtractSumFeaturesPreprocessor(Preprocessor):
    NAME = "extract_sum_features"

    def __init__(self, preprocessors=None, schedule_cfg=None):
        super().__init__(preprocessors, schedule_cfg)

    def __call__(self, unpreprocessed, **kwargs):
        is_training = kwargs.get("is_training", True)
        performance = kwargs.get("performance", "latency")
        unpreprocessed = list(unpreprocessed)
        train_x = []
        train_y = []
        for prof_net in unpreprocessed:
            train_x += [[prof_net["block_sum_{}".format(performance)]]]
            if is_training:
                train_y += [prof_net["overall_{}".format(performance)]]
        train_x = np.array(train_x).reshape(-1, 1)
        if is_training:
            train_y = np.array(train_y).reshape(-1)
            return unpreprocessed, train_x, train_y
        return unpreprocessed, train_x

class ExtractLSTMFeaturesPreProcessor(Preprocessor):
    NAME = "extract_lstm_features"

    def __init__(self, preprocessors=None, schedule_cfg=None):
        super().__init__(preprocessors, schedule_cfg)

    def __call__(self, unpreprocessed, **kwargs):
        is_training = kwargs.get('is_training', True)
        perf_name = kwargs.get("performance", 'latency')
        unpreprocessed = list(unpreprocessed)
        train_x = list()
        train_y = list()
        for prof_net in unpreprocessed:
            x_feature = list()
            for prim in prof_net['primitives']:
                cin    = prim['C']
                cout   = prim['C_out']
                exp    = prim['expansion'] if 'expansion' in prim else 1
                k      = prim['kernel_size'] if 'kernel_size' in prim else 3
                perf   = prim['performances'][perf_name]
                size   = prim['spatial_size']
                stride = prim['stride']
                x_feature.append([cin, cout, exp, k, perf, size, stride])
            train_x.append(x_feature)
            if is_training:
                train_y.append(prof_net['overall_{}'.format(perf_name)])
        if is_training:
            return unpreprocessed, train_x, train_y
        else:
            return unpreprocessed, train_x
            
class PaddingPreProcessor(Preprocessor):
    NAME = "padding"

    def __init__(self, preprocessors=None, schedule_cfg=None):
        super().__init__(preprocessors, schedule_cfg)

    def __call__(self, unpreprocessed, **kwargs):
        is_training = kwargs.get("is_training", True)
        perf_name = kwargs.get('performance', 'latency')
        unpreprocessed = list(unpreprocessed)
        train_x = list()
        train_y = list()
        for prof_net in unpreprocessed:
            x_feature = list()
            for prim in prof_net['primitives']:
                x_feature.append(prim['performances'][perf_name])
            train_x.append(x_feature)
            if is_training:
                train_y.append(prof_net['overall_{}'.format(perf_name)])
        # pad train_x
        max_len = max([len(_) for _ in train_x])
        for x in train_x:
            if len(x) == max_len: continue
            x.extend([0] * (max_len - len(x)) )

        if is_training:
            return unpreprocessed, train_x, train_y
        else:
            return unpreprocessed, train_x


class TableBasedModel(BaseHardwarePerformanceModel):
    NAME = "table"

    def __init__(
        self,
        mixin_search_space,
        *,
        perf_name="latency",
        preprocessors=("flatten", ),
        prof_prims_cfg={},
        schedule_cfg=None,
    ):
        super(TableBasedModel, self).__init__(
            mixin_search_space,
            perf_name=perf_name,
            preprocessors=preprocessors,
            schedule_cfg=schedule_cfg,
        )
        self.prof_prims_cfg = prof_prims_cfg

        self._table = {}

    def _train(self, args):
        prof_nets = args
        for net in prof_nets:
            for prim in net.get("primitives", []):
                perf = prim.pop("performances")[self.perf_name]
                prim = Prim(**prim)
                self._table.setdefault(prim, []).append(perf)

        self._table = {k: np.mean(v) for k, v in self._table.items()}

    def predict(self, rollout, assemble_fn=sum):
        # return random.random()
        primitives = self.mixin_search_space.rollout_to_primitives(
            rollout, **self.prof_prims_cfg)
        perfs = []
        for prim in primitives:
            perf = self._table.get(prim)
            if perf is None:
                self.logger.warn(
                    "primitive %s is not found in the table, return default value 0.",
                    prim)
                perf = 0.
            perfs += [perf]
        return assemble_fn(perfs)

    def save(self, path):
        pickled_table = [(k._asdict(), v) for k, v in self._table.items()]
        with open(path, "wb") as wf:
            pickle.dump(
                {
                    "table": pickled_table,
                }, wf)

    def load(self, path):
        with open(path, "rb") as fr:
            m = pickle.load(fr)
        self._table = {Prim(**k): v for k, v in m["table"]}


class RegressionModel(TableBasedModel):
    NAME = "regression"

    def __init__(
        self,
        mixin_search_space,
        *,
        perf_name="latency",
        preprocessors=("block_sum", "remove_anomaly", "flatten",
                       "extract_sum_features"),
        prof_prims_cfg={},
        schedule_cfg=None,
    ):
        super().__init__(mixin_search_space,
                         perf_name=perf_name,
                         preprocessors=preprocessors,
                         prof_prims_cfg=prof_prims_cfg,
                         schedule_cfg=schedule_cfg)
        self.regression_model = linear_model.LinearRegression()

        assert isinstance(mixin_search_space, MixinProfilingSearchSpace)

    def _train(self, args):
        prof_nets, train_x, train_y = args
        super()._train(prof_nets)
        return self.regression_model.fit(train_x, train_y)

    def predict(self, rollout):
        primitives = self.mixin_search_space.rollout_to_primitives(
            rollout, **self.prof_prims_cfg)
        perfs = super().predict(rollout, assemble_fn=lambda x: x)
        primitives = [p._asdict() for p in primitives]
        for prim, perf in zip(primitives, perfs):
            prim["performances"] = {self.perf_name: perf}
        prof_nets = [[{"primitives": primitives}]]
        prof_nets, test_x = self.preprocessor(
            prof_nets, is_training=False, performance=self.perf_name)
        return float(self.regression_model.predict(test_x)[0])

    def save(self, path):
        pickled_table = [(k._asdict(), v) for k, v in self._table.items()]
        with open(path, "wb") as fw:
            pickle.dump(
                {
                    "table": pickled_table,
                    "model": self.regression_model
                }, fw)

    def load(self, path):
        with open(path, "rb") as fr:
            m = pickle.load(fr)
        self._table = {Prim(**k): v for k, v in m["table"]}
        self.regression_model = m["model"]


class MLPModel(TableBasedModel):
    NAME = 'mlp'

    def __init__(
        self,
        mixin_search_space,
        *,
        perf_name='latency',
        preprocessors=('block_sum','remove_anomaly','flatten','extract_sum_features'),
        prof_prims_cfg={},
        schedule_cfg=None,
    ):

        super().__init__(
            mixin_search_space,
            perf_name=perf_name,
            preprocessors=preprocessors,
            prof_prims_cfg=prof_prims_cfg,
            schedule_cfg=schedule_cfg
        )       

        self.mlp_model = MLPRegressor(
            solver='adam',
            alpha=1e-4,
            hidden_layer_sizes=(100,100,100),
            random_state=1,
            max_iter=10000
        )
           
    def _train(self, args):
        prof_nets, train_x, train_y = args
        super()._train(prof_nets)
        return self.mlp_model.fit(train_x, train_y)

    def predict(self, rollout):
        primitives = self.mixin_search_space.rollout_to_primitives(
            rollout, **self.prof_prims_cfg)
        perfs = super().predict(rollout, assemble_fn=lambda x: x)
        primitives = [p._asdict() for p in primitives]
        for prim, perf in zip(primitives, perfs):
            prim["performances"] = {self.perf_name: perf}
        prof_nets = [[{"primitives": primitives}]]
        prof_nets, test_x = self.preprocessor(
            prof_nets, is_training=False, performance=self.perf_name)
        return float(self.mlp_model.predict(test_x)[0])

    def save(self, path):
        pickled_table = [(k._asdict(), v) for k, v in self._table.items()]
        with open(path, "wb") as fw:
            pickle.dump(
                {
                    "table": pickled_table,
                    "model": self.mlp_model
                }, fw)

    def load(self, path):
        with open(path, "rb") as fr:
            m = pickle.load(fr)
        self._table = {Prim(**k): v for k, v in m["table"]}
        self.mlp_model = m["model"]




class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, device="cpu"):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True).to(self.device)
        self.linear = nn.Linear(hidden_layer_size, output_size).to(self.device)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(self.device))
        

    def forward(self, seq, bs=1):
        self.hidden_cell = (torch.zeros(1, bs, self.hidden_layer_size).to(self.device),
                                    torch.zeros(1, bs, self.hidden_layer_size).to(self.device))
        seq = rnn_utils.pack_sequence([torch.tensor(s).reshape(-1, 1).to(self.device) for s in seq], enforce_sorted=False)
        lstm_out, self.hidden_cell = self.lstm(seq, self.hidden_cell)
        lstm_out, index = rnn_utils.pad_packed_sequence(lstm_out)
        lstm_out = lstm_out.permute([1, 0, 2])
        select = torch.zeros(lstm_out.shape[:2]).scatter_(1, index.reshape(-1, 1) - 1, 1).to(torch.bool).to(self.device)
        lstm_out = lstm_out[select]
        predictions = self.linear(lstm_out)
        return predictions[:, -1]

    def predict(self, input_seqs):
        return self.forward(input_seqs, bs=len(input_seqs)).cpu().detach().numpy()

    def fit(self, train_X, train_y, epochs=3000, bs=128):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for i in range(epochs):
            for j in range(len(train_X) // bs + 1):
                seq = train_X[j * bs: (j + 1) * bs]
                label = train_y[j * bs: (j + 1) * bs]
                optimizer.zero_grad()
                pred = self.forward(seq, bs=bs)
                loss = loss_function(pred, torch.tensor(label).to(pred.device))
                loss.backward()
                optimizer.step()

            if i % 5 == 0:
                print("epoch: {:3}, loss: {:10.5f}".format(
                    i, loss.item()))
        return self 


class LSTMModel(TableBasedModel):
    NAME = "lstm"

    def __init__(
        self,
        mixin_search_space,
        *,
        perf_name="latency",
        preprocessors=("block_sum", "remove_anomaly", "flatten",
                       "extract_lstm_features"),
        prof_prims_cfg={},
        schedule_cfg=None,
    ):
        super().__init__(mixin_search_space,
                         perf_name=perf_name,
                         preprocessors=preprocessors,
                         prof_prims_cfg=prof_prims_cfg,
                         schedule_cfg=schedule_cfg)
        gpu = torch.cuda.current_device()
        self.device = 'cuda:' + str(gpu)
        self.lstm_model = LSTM(1, 100, 1, device=self.device)
        assert isinstance(mixin_search_space, MixinProfilingSearchSpace) 

    def _train(self, args):
        prof_nets, train_x, train_y = args
        # build Prim -> performance look-up table
        super()._train(prof_nets)
        return self.lstm_model.fit(train_x, train_y)


    def predict(self, rollout):
        primitives = self.mixin_search_space.rollout_to_primitives(
            rollout, **self.prof_prims_cfg)
        perfs = super().predict(rollout, assemble_fn=lambda x: x)
        primitives = [p._asdict() for p in primitives]
        for prim, perf in zip(primitives, perfs):
            prim["performances"] = {self.perf_name: perf}
        prof_nets = [[{"primitives": primitives}]]
        prof_nets, test_x = self.preprocessor(
            prof_nets, is_training=False, performance=self.perf_name)
        return float(self.lstm_model.predict(test_x)[0])

    def save(self, path):
        pickled_table = [(k._asdict(), v) for k, v in self._table.items()]
        with open(path, "wb") as fw:
            pickle.dump(
                {
                    "table": pickled_table,
                    "model": self.lstm_model
                }, fw)

    def load(self, path):
        with open(path, "rb") as fr:
            m = pickle.load(fr)
        self._table = {Prim(**k): v for k, v in m["table"]}
        self.lstm_model = m["model"].to(self.device)



def iterate(prof_prim_dir):
    for _dir in os.listdir(prof_prim_dir):
        cur_dir = os.path.join(prof_prim_dir, _dir)
        if not os.path.isdir(cur_dir):
            continue
        prof_net = []
        for f in os.listdir(cur_dir):
            if not f.endswith("yaml"):
                continue
            with open(os.path.join(cur_dir, f), "r") as fr:
                prof_net += [yaml.load(fr)]
        yield prof_net
