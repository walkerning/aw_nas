# -*- coding: utf-8 -*-

import torch

from aw_nas.weights_manager.base import BaseWeightsManager
from aw_nas.final.base import FinalModel

__all__ = ["MorphismWeightsManager"]

class MorphismWeightsManager(BaseWeightsManager):
    NAME = "morphism"

    def __init__(self, search_space, device, rollout_type):
        super(MorphismWeightsManager, self).__init__(search_space, device, rollout_type)

        self.search_space = search_space
        self.device = device
        self.rollout_type = rollout_type

    def assemble_candidate(self, rollout):
        """Assemble a candidate net using rollout.
        """
        _model_record = rollout.population.get_model(rollout.parent_index)
        _parent_model = torch.load(_model_record.checkpoint_path)
        if not isinstance(_parent_model, dict):
            parent_state_dict = _parent_model.state_dict()
        else:
            parent_state_dict = _parent_model
        # construct a new CNNGenotypeModel using new configuration
        _child_model = FinalModel.get_class_(_model_record.config["final_model_type"])(
            self.search_space, self.device,
            **_model_record.config["final_model_cfg"]
        )
        for n, v in _child_model.named_parameters():
            if n in parent_state_dict:
                v.data.copy_(parent_state_dict[n])
        return _child_model

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

    def set_device(self, device):
        self.device = device
