# -*- coding: utf-8 -*-

import numpy as np
import torch

from aw_nas.final.general_model import GeneralGenotypeModel
from aw_nas.utils.common_utils import Ticker, tick

def analyze_elapses(model, inputs, use_cuda=True, forward_time=100):
    assert isinstance(model, GeneralGenotypeModel)

    model.eval()

    @tick("_forward_elapse")
    def forward(model, inputs, *args, **kwargs):
        return model(inputs, *args, **kwargs)

    for _ in range(2):
        model(inputs)
        torch.cuda.synchronize()

    def callback(model, inputs, out):
        if use_cuda:
            torch.cuda.synchronize()
        elapses.append(ticker.tick() * 1000)

    all_elapses = []
    async_elapse = 0.
    sync_elapse = 0.
    for _ in range(forward_time):
        ticker = Ticker("general_forward")
        elapses = []
        forward(model, inputs, callback=callback)
        forward(model, inputs)
        all_elapses.append(elapses)
        async_elapse += model._forward_elapse
        sync_elapse += ticker.total_time * 1000

    mean_elapse = np.array(all_elapses).mean(axis=0)
    async_elapse /= forward_time
    sync_elapse /= forward_time

    genotypes = [{"elapse": elapse, **geno} for elapse, geno in zip(mean_elapse, model.genotypes)]

    return {"primitives": genotypes, "async_elapse": async_elapse, "sync_elapse": sync_elapse}
