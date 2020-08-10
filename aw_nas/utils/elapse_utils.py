# -*- coding: utf-8 -*-

import numpy as np
import torch

from aw_nas.final.general_model import GeneralGenotypeModel
from aw_nas.utils.common_utils import Ticker, tick


def analyze_elapses(model, inputs, device, forward_time=100):
    assert isinstance(model, GeneralGenotypeModel)
    model = model.to(device)
    inputs = inputs.to(device)
    model.eval()

    @tick("_forward_elapse", device)
    def forward(model, inputs, *args, **kwargs):
        return model(inputs, *args, **kwargs)

    for _ in range(2):
        model(inputs)
        if device != "cpu":
            torch.cuda.synchronize(device=device)

    def callback(model, inputs, out):
        if device != "cpu":
            torch.cuda.synchronize(device=device)
        elapses.append(ticker.tick() * 1000)

    for _ in range(forward_time):
        ticker = Ticker("general_forward")
        elapses = []
        forward(model, inputs, callback=callback)
        forward(model, inputs)
        async_elapse = model._forward_elapse
        sync_elapse = ticker.total_time * 1000

        genotypes = [{
            "performances": {
                "latency": float(elapse)
            },
            **geno
        } for elapse, geno in zip(elapses, model.genotypes)]

        yield {
            "primitives": genotypes,
            "overall_latency": float(async_elapse),
            "block_sum_latency": float(sync_elapse)
        }
