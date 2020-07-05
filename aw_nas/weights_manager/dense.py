# -*- coding: utf-8 -*-

import numpy as np
import torch

from aw_nas.weights_manager.base import BaseWeightsManager
from aw_nas.final.base import FinalModel
from aw_nas.rollout import DenseMutation

__all__ = ["DenseMorphismWeightsManager"]

class DenseMorphismWeightsManager(BaseWeightsManager):
    NAME = "dense"

    def __init__(self, search_space, device, rollout_type, noise_type=None):
        super(DenseMorphismWeightsManager, self).__init__(search_space, device, rollout_type)

        self.search_space = search_space
        self.device = device
        self.rollout_type = rollout_type
        self.noise_type = noise_type

    def assemble_candidate(self, rollout):
        """Assemble a candidate net using rollout.
        """
        _model_record = rollout.population.get_model(rollout.parent_index)
        _parent_model = torch.load(_model_record.checkpoint_path)
        _mutation = rollout.mutations[-1]
        if not isinstance(_parent_model, dict):
            parent_state_dict = _parent_model.state_dict()
        else:
            parent_state_dict = _parent_model
        # construct a new CNNGenotypeModel using new configuration
        _child_model = FinalModel.get_class_(rollout.model_record.config["final_model_type"])(
            self.search_space, self.device,
            **rollout.model_record.config["final_model_cfg"]
        )
        for n, v in _child_model.named_parameters():
            if n.startswith("dense_blocks"):
                block_idx, miniblock_idx = [int(i) for i in n.split(".")[1:3]]
                if block_idx < _mutation.block_idx or block_idx == _mutation.block_idx and miniblock_idx < _mutation.miniblock_idx:
                    v.data.copy_(parent_state_dict[n])
            elif n.startswith("trans_blocks"):
                block_idx = int(n.split(".")[1])
                if block_idx < _mutation.block_idx:
                    v.data.copy_(parent_state_dict[n])
            elif n.startswith("stem"):
                v.data.copy_(parent_state_dict[n])
        if _mutation.mutation_type == DenseMutation.WIDER:
            self.widen(rollout, _child_model, parent_state_dict, _mutation.block_idx, _mutation.miniblock_idx,\
                       _mutation.modified)
        elif _mutation.mutation_type == DenseMutation.DEEPER:
            self.deepen(rollout, _child_model, parent_state_dict, _mutation.block_idx, _mutation.miniblock_idx)
        return _child_model

    def add_noise(self, tensor):
        if self.noise_type is None:
            return tensor
        if self.noise_type == 'normal':
            std = np.std(tensor)
            noise = np.random.normal(0, std*1e-3, size=tensor.shape)
        elif self.noise_type == 'uniform':
            mean, _max = np.mean(tensor), np.max(tensor)
            width = (_max - mean) * 1e-3
            noise = np.random.uniform(-width, width, size=tensor.shape)
        else:
            raise NotImplementedError
        return tensor + noise

    def widen(self, rollout, child_model, parent_state_dict, block_idx, miniblock_idx, modified):
        widen_list = ["conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]
        prefix = "dense_blocks.{}.{}.".format(block_idx, miniblock_idx)
        assert prefix + widen_list[0] in parent_state_dict.keys()
        output_channels = parent_state_dict[prefix + "conv.weight"].shape[0]
        input_channels = parent_state_dict[prefix + "conv.weight"].shape[1]
        widen_record_bc = [i for i in range(input_channels)]
        if rollout.search_space.bc_mode:
            origin_bc = parent_state_dict[prefix + "bc_" + widen_list[0]]
            input_channels = origin_bc.shape[1]
            modified_bc_output_channels = rollout.search_space.bc_ratio * modified
            assert modified_bc_output_channels >= origin_bc.shape[0]
            magnifier_bc = [1 for i in range(origin_bc.shape[0])]
            widen_record_bc = [i for i in range(origin_bc.shape[0])]
            for i in range(modified_bc_output_channels - origin_bc.shape[0]):
                sample_i = np.random.randint(0, origin_bc.shape[0])
                magnifier_bc[sample_i] += 1
                widen_record_bc.append(sample_i)
            for i in range(modified_bc_output_channels - origin_bc.shape[0]):
                magnifier_bc.append(magnifier_bc[widen_record_bc[i + origin_bc.shape[0]]])
        assert output_channels <= modified
        magnifier = [1 for i in range(output_channels)]
        widen_record = [i for i in range(output_channels)]
        for i in range(modified - output_channels):
            sample_i = np.random.randint(0, output_channels)
            magnifier[sample_i] += 1
            widen_record.append(sample_i)
        for i in range(modified - output_channels):
            magnifier.append(magnifier[widen_record[i + output_channels]])
        magnifier = torch.tensor(np.array(magnifier).reshape(1, len(magnifier), 1, 1), dtype=parent_state_dict[prefix+"conv.weight"].dtype,\
								device=parent_state_dict[prefix+"conv.weight"].get_device())
        if rollout.search_space.bc_mode:
            magnifier_bc = torch.tensor(np.array(magnifier_bc).reshape(1, len(magnifier_bc), 1, 1), dtype=parent_state_dict[prefix+"conv.weight"].dtype,\
                                device=parent_state_dict[prefix+"conv.weight"].get_device())
        if rollout.search_space.bc_mode:
            origin_layer = prefix + "bc_conv.weight"
            child_model.state_dict()[origin_layer][:,:,:,:] = parent_state_dict[origin_layer][widen_record_bc,:,:,:]
        for widen_layer in widen_list:
            origin_layer = prefix + widen_layer
            assert origin_layer in parent_state_dict.keys()
            if widen_layer == "conv.weight" and rollout.search_space.bc_mode:
                child_model.state_dict()[origin_layer][:output_channels,:,:,:] =\
                     self.add_noise(parent_state_dict[origin_layer][:,widen_record_bc,:,:] / magnifier_bc)
                child_model.state_dict()[origin_layer][output_channels:,:,:,:] =\
                    child_model.state_dict()[origin_layer][widen_record[output_channels:],:,:,:]
            else:
                child_model.state_dict()[origin_layer][:] \
                    = parent_state_dict[origin_layer][widen_record_bc]
        miniblock_iter = miniblock_idx + 1
        if rollout.search_space.bc_mode:
            layer_name = "bc_conv.weight"
        else:
            layer_name = "conv.weight"
        transition_wider = modified - output_channels
        for i in range(block_idx, rollout.search_space.num_dense_blocks):
            channel_bias = 0
            if i != block_idx:
                miniblock_iter = 0
            while True:
                prefix = "dense_blocks.{}.{}.".format(i, miniblock_iter)
                origin_layer = prefix + layer_name
                if origin_layer not in parent_state_dict.keys():
                    break
                child_model.state_dict()[origin_layer][:,channel_bias+len(widen_record):,:,:] = parent_state_dict[origin_layer][:,channel_bias+output_channels:,:,:]
                child_model.state_dict()[origin_layer][:,channel_bias:channel_bias+len(widen_record),:,:] = parent_state_dict[origin_layer][:,[x+channel_bias for x in widen_record],:,:] / magnifier
                child_model.state_dict()[origin_layer][:,:channel_bias,:,:] = parent_state_dict[origin_layer][:,:channel_bias,:,:]
                if rollout.search_space.bc_mode:
                    child_model.state_dict()[prefix + "conv.weight"][:] = parent_state_dict[prefix + "conv.weight"]
                    for bn in widen_list[1:]:
                        child_model.state_dict()[prefix + "bc_" + bn][channel_bias+len(widen_record):] = parent_state_dict[prefix + "bc_" + bn][channel_bias+output_channels:]
                        child_model.state_dict()[prefix + "bc_" + bn][channel_bias:channel_bias+len(widen_record)] = parent_state_dict[prefix + "bc_" + bn][[x+channel_bias for x in widen_record]]
                        child_model.state_dict()[prefix + "bc_" + bn][:channel_bias] = parent_state_dict[prefix + "bc_" + bn][:channel_bias]
                child_model.state_dict()[prefix + bn][:] = parent_state_dict[prefix + bn] 
                channel_bias += child_model.state_dict()[prefix + "conv.weight"].shape[0]
                miniblock_iter += 1
            origin_layer = "trans_blocks.{}.conv.weight".format(i)
            if origin_layer not in parent_state_dict.keys():
                break
            transition_wider //= 2
            child_model.state_dict()[origin_layer][:-transition_wider,channel_bias+len(widen_record):,:,:] = parent_state_dict[origin_layer][:,channel_bias+output_channels:,:,:]
            child_model.state_dict()[origin_layer][:-transition_wider,channel_bias:channel_bias+len(widen_record),:,:] = parent_state_dict[origin_layer][:,[x+channel_bias for x in widen_record],:,:] / magnifier
            child_model.state_dict()[origin_layer][:-transition_wider,:channel_bias,:,:] = parent_state_dict[origin_layer][:,:channel_bias,:,:]
            output_channels = parent_state_dict[origin_layer].shape[0]
            widen_record = [j for j in range(output_channels)]
            magnifier = [1 for j in range(output_channels)]
            for j in range(transition_wider):
                widen_record.append(np.random.randint(0, output_channels))
                magnifier[widen_record[-1]] += 1
            for j in range(transition_wider):
                magnifier.append(magnifier[widen_record[output_channels + j]])
            magnifier = torch.tensor(np.array(magnifier).reshape(1, len(magnifier), 1, 1), dtype=parent_state_dict[origin_layer].dtype,\
								device=parent_state_dict[origin_layer].get_device())
            child_model.state_dict()[origin_layer][-transition_wider:,:,:,:] = child_model.state_dict()[origin_layer][widen_record[output_channels:],:,:,:]
        final_bn = ['final_bn.bias', 'final_bn.weight', 'final_bn.running_mean', 'final_bn.running_var']
        for ele in final_bn:
            child_model.state_dict()[ele][channel_bias:channel_bias+len(widen_record)] = parent_state_dict[ele][[x+channel_bias for x in widen_record]]
            child_model.state_dict()[ele][channel_bias+len(widen_record):] = parent_state_dict[ele][channel_bias+output_channels:]
            child_model.state_dict()[ele][:channel_bias] = parent_state_dict[ele][:channel_bias]
        child_model.state_dict()['classifier.weight'][:,channel_bias:channel_bias+len(widen_record)] = parent_state_dict['classifier.weight'][:,[x+channel_bias for x in widen_record]] / magnifier.reshape([1,len(widen_record)])
        child_model.state_dict()['classifier.weight'][:,channel_bias+len(widen_record):] = parent_state_dict['classifier.weight'][:,channel_bias+output_channels:]
        child_model.state_dict()['classifier.weight'][:,:channel_bias] = parent_state_dict['classifier.weight'][:,:channel_bias]
        child_model.state_dict()['classifier.bias'][:] = parent_state_dict['classifier.bias']




    def deepen(self, rollout, child_model, parent_state_dict, block_idx, miniblock_idx):
        if rollout.search_space.bc_mode:
            input_name = "dense_blocks.{}.{}.bc_conv.weight".format(block_idx, miniblock_idx)
        else:
            input_name = "dense_blocks.{}.{}.conv.weight".format(block_idx, miniblock_idx)
        output_name = "dense_blocks.{}.{}.conv.weight".format(block_idx, miniblock_idx)
        input_channels = child_model.state_dict()[input_name].shape[1]
        output_channels = child_model.state_dict()[output_name].shape[0]
        growth = output_channels - input_channels
        bc_random_mapping = [i for i in range(input_channels)]
        if rollout.search_space.bc_mode:
            bc_random_mapping = np.random.randint(0, input_channels, size=rollout.search_space.bc_ratio*growth)
            random_mapping = np.random.randint(0, rollout.search_space.bc_ratio*growth, size=growth)
            child_model.state_dict()[input_name] = 0.
            for i in range(rollout.search_space.bc_ratio*growth):
                child_model.state_dict()[input_name][i,bc_random_mapping[i],1,1] = 1.
        else:
            random_mapping = np.random.randint(0, input_channels, size=output_channels)
        for i in range(output_channels):
            child_model.state_dict()[output_name][i,random_mapping[i],1,1] = 1.
        child_model.state_dict()[output_name] = self.add_noise(child_model.state_dict()[output_name]) #TODO: do we need this?
        miniblock_iter = miniblock_idx + 1
        if rollout.search_space.bc_mode:
            layer_name = "bc_conv.weight"
        else:
            layer_name = "conv.weight"
        widen_record = [i for i in range(input_channels)]
        magnifier = [1 for i in range(input_channels)]
        for i in random_mapping:
            widen_record.append(bc_random_mapping[i])
            magnifier[bc_random_mapping[i]] += 1
        for i in random_mapping:
            magnifier.append(magnifier[bc_random_mapping[i]])
        magnifier = torch.tensor(np.array(magnifier).reshape(1, len(magnifier), 1, 1), dtype=child_model.state_dict()[input_name].dtype,\
                                 device=child_model.state_dict()[input_name].get_device())
        while True:
            prefix = "dense_blocks.{}.{}.".format(block_idx, miniblock_iter)
            prefix_child = "dense_blocks.{}.{}.".format(block_idx, miniblock_iter + 1)
            origin_layer = prefix + layer_name
            new_layer = prefix_child + layer_name
            if origin_layer not in parent_state_dict.keys():
                break
            child_model.state_dict()[new_layer][:,len(widen_record):,:,:] = parent_state_dict[origin_layer][:,input_channels:,:,:]
            child_model.state_dict()[new_layer][:,:len(widen_record),:,:] = parent_state_dict[origin_layer][:,widen_record,:,:] / magnifier
            miniblock_iter += 1
        origin_layer = "trans_blocks.{}.conv.weight".format(block_idx)
        child_model.state_dict()[origin_layer][:,len(widen_record):,:,:] = parent_state_dict[origin_layer][:,input_channels:,:,:]
        child_model.state_dict()[origin_layer][:,:len(widen_record),:,:] = parent_state_dict[origin_layer][:,widen_record,:,:] / magnifier


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
        return ["dense_mutation"]

    @classmethod
    def supported_data_types(cls):
        """Return the supported data types"""
        return ["image"]

    def set_device(self, device):
        self.device = device
