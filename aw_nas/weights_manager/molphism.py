# -*- coding: utf-8 -*-

import torch
from torch import nn

from aw_nas.weights_manager.base import BaseWeightsManager
from aw_nas.utils.exception import expect, ConfigException

class MolphismWeightsManager(BaseWeightsManager):
    NAME = "molphism"
    def __init__(self, search_space, device, rollout_type):
        super(MolphismWeightsManager, self).__init__(search_space, device, rollout_type)

        self.search_space = search_space
        self.device = device
        expect(rollout_type in self.supported_rollout_types(),
               "Unsupported `rollout_type`: {}".format(rollout_type),
               ConfigException) # supported rollout types
        self.rollout_type = rollout_type

    def assemble_candidate(self, rollout):
        """Assemble a candidate net using rollout.
        """
        _model_record = rollout.population.get_model(rollout.parent_index)
        _parent_model = torch.load(_model_record.checkpoint_path)
        return _parent_model.state_dict()

    def step(self, gradients, optimizer):
        """Update the weights manager state using gradients."""
        pass

    def save(self, path):
        """Save the state of the weights_manager to `path` on disk."""
        pass

    def load(self, path):
        """Load the state of the weights_manager from `path` on disk."""
        pass

    @classmethod
    def supported_rollout_types(cls):
        """Return the accepted rollout-type."""
        return ["mutation"]

    @classmethod
    def supported_data_types(cls):
        """Return the supported data types"""
        return ["image"]

