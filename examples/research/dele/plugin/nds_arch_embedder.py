import torch
import torch.nn as nn
import torch.nn.functional as F

from aw_nas.evaluator.arch_network import ArchEmbedder


class ResNetArchEmbedder(ArchEmbedder):
    NAME = "resnet_lstm"

    def __init__(self, search_space, depth_choices, width_choices,
            depth_embedding_size: int,
            width_embedding_size: int,
            hidden_size: int,
            dropout_rate: float = 0.,
            num_layers: int = 1,
            schedule_cfg = None
    ):
        super(ResNetArchEmbedder, self).__init__(schedule_cfg)
        
        self.search_space = search_space
        self.depth_choices = depth_choices
        self.width_choices = width_choices
        self.depth_embedding_size = depth_embedding_size
        self.width_embedding_size = width_embedding_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # assume all the layers share the same depth choices and width choices
        self.depth_emb = nn.Embedding(len(self.depth_choices), self.depth_embedding_size)
        self.width_emb = nn.Embedding(len(self.width_choices), self.width_embedding_size)

        self.rnn = nn.LSTM(input_size = self.depth_embedding_size + self.width_embedding_size,
                           hidden_size = self.hidden_size, num_layers = self.num_layers,
                           batch_first = True, dropout = self.dropout_rate)

        self.depth_mapping = {choice: i for i, choice in enumerate(self.depth_choices)}
        self.width_mapping = {choice: i for i, choice in enumerate(self.width_choices)}
        
        self.out_dim = self.hidden_size

    def embed_and_transform_arch(self, archs):
        all_arch_depths = []
        all_arch_widths = []

        for arch in archs:
            decisions = list(arch.values())
            network_layer_num = len(decisions) // 2

            all_arch_depths.append([
                self.depth_mapping[choice] 
                for choice in decisions[:network_layer_num]
            ])
            all_arch_widths.append([
                self.width_mapping[choice] 
                for choice in decisions[network_layer_num:]
            ])

        all_arch_depths = self.depth_emb.weight.new(all_arch_depths).long()
        all_arch_widths = self.width_emb.weight.new(all_arch_widths).long()

        depth_embs = self.depth_emb(all_arch_depths)
        width_embs = self.width_emb(all_arch_widths)

        cat_emb = torch.cat([depth_embs, width_embs], dim = -1)
        return cat_emb

    def forward(self, archs):
        emb = self.embed_and_transform_arch(archs)
        out, _ = self.rnn(emb)
        out = F.normalize(out, 2, dim = -1)

        out = torch.mean(out, dim = 1)
        out = F.normalize(out, 2, dim = -1)
        return out


class ResNextAArchEmbedder(ArchEmbedder):
    NAME = "resnexta_lstm"

    def __init__(self, search_space, 
            depth_choices, width_choices,
            bot_mul_choices, num_group_choices,
            depth_embedding_size: int,
            width_embedding_size: int,
            bot_mul_embedding_size: int,
            num_group_embedding_size: int,
            hidden_size: int,
            dropout_rate: float = 0.,
            num_layers: int = 1,
            schedule_cfg = None
    ):
        super(ResNextAArchEmbedder, self).__init__(schedule_cfg)
        
        self.search_space = search_space
        
        self.depth_choices = depth_choices
        self.width_choices = width_choices
        self.bot_mul_choices = bot_mul_choices
        self.num_group_choices = num_group_choices

        self.depth_embedding_size = depth_embedding_size
        self.width_embedding_size = width_embedding_size
        self.bot_mul_embedding_size = bot_mul_embedding_size
        self.num_group_embedding_size = num_group_embedding_size

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # assume all the layers share the same depth choices and width choices
        self.depth_emb = nn.Embedding(len(self.depth_choices), self.depth_embedding_size)
        self.width_emb = nn.Embedding(len(self.width_choices), self.width_embedding_size)
        self.bot_mul_emb = nn.Embedding(len(self.bot_mul_choices), self.bot_mul_embedding_size)
        self.num_group_emb = nn.Embedding(len(self.num_group_choices), self.num_group_embedding_size)

        rnn_input_emb_size = self.depth_embedding_size + self.width_embedding_size +\
                self.bot_mul_embedding_size + self.num_group_embedding_size
        self.rnn = nn.LSTM(input_size = self.depth_embedding_size + self.width_embedding_size +
                                        self.bot_mul_embedding_size + self.num_group_embedding_size,
                           hidden_size = self.hidden_size, num_layers = self.num_layers,
                           batch_first = True, dropout = self.dropout_rate)

        self.depth_mapping = {choice: i for i, choice in enumerate(self.depth_choices)}
        self.width_mapping = {choice: i for i, choice in enumerate(self.width_choices)}
        self.bot_mul_mapping = {choice: i for i, choice in enumerate(self.bot_mul_choices)}
        self.num_group_mapping = {choice: i for i, choice in enumerate(self.num_group_choices)}
        
        self.out_dim = self.hidden_size

    def embed_and_transform_arch(self, archs):
        all_arch_depths = []
        all_arch_widths = []
        all_arch_bot_muls = []
        all_arch_num_groups = []

        for arch in archs:
            decisions = list(arch.values())
            each_layer_num = len(decisions) // 4

            all_arch_depths.append([
                self.depth_mapping[choice] 
                for choice in decisions[:each_layer_num]
            ])
            all_arch_widths.append([
                self.width_mapping[choice] 
                for choice in decisions[each_layer_num : 2 * each_layer_num]
            ])
            all_arch_bot_muls.append([
                self.bot_mul_mapping[choice]
                for choice in decisions[2 * each_layer_num : 3 * each_layer_num]
            ])
            all_arch_num_groups.append([
                self.num_group_mapping[choice]
                for choice in decisions[3 * each_layer_num : ]
            ])

        all_arch_depths = self.depth_emb.weight.new(all_arch_depths).long()
        all_arch_widths = self.width_emb.weight.new(all_arch_widths).long()
        all_arch_bot_muls = self.bot_mul_emb.weight.new(all_arch_bot_muls).long()
        all_arch_num_groups = self.num_group_emb.weight.new(all_arch_num_groups).long()

        depth_embs = self.depth_emb(all_arch_depths)
        width_embs = self.width_emb(all_arch_widths)
        bot_mul_embs = self.bot_mul_emb(all_arch_bot_muls)
        num_groups_embs = self.num_group_emb(all_arch_num_groups)

        cat_emb = torch.cat([depth_embs, width_embs, bot_mul_embs, num_groups_embs], dim = -1)
        return cat_emb

    def forward(self, archs):
        emb = self.embed_and_transform_arch(archs)
        out, _ = self.rnn(emb)
        out = F.normalize(out, 2, dim = -1)

        out = torch.mean(out, dim = 1)
        out = F.normalize(out, 2, dim = -1)
        return out
