import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregator import Mean_aggregator
import numpy as np
import random


class GraphSAGE(nn.Module):
    def __init__(self, add_self, concat, aggregator_method, node_features, input_dim, output_dim, hidden_dim, adj_list, sample_num):
        super(GraphSAGE, self).__init__()
        self.add_self = add_self
        self.concat = concat
        self.aggregator_method = aggregator_method
        self.node_features = torch.FloatTensor(node_features)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.adj_list = adj_list
        self.sample_num = sample_num

        self.embedding_layer = nn.Embedding(self.node_features.size(0), self.node_features.size(1)).from_pretrained(self.node_features, freeze=True)

        if aggregator_method == 'mean':
            self.aggregator_1 = Mean_aggregator(self.node_features)
            self.aggregator_2 = Mean_aggregator(self.node_features)

        self.fc_layer_1 = nn.Linear(self.input_dim * 2 if self.concat else self.input_dim, self.hidden_dim, bias=False)
        self.fc_layer_2 = nn.Linear(self.input_dim * 2 if self.concat else self.hidden_dim, self.hidden_dim, bias=False)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        nn.init.xavier_uniform_(self.fc_layer_1.weight.data)
        nn.init.xavier_uniform_(self.fc_layer_2.weight.data)
        nn.init.xavier_uniform_(self.output_layer.weight.data)

    def forward(self, nodes):
        sample_neighbors_1, neighs_1 = self.get_neighbor_nodes(nodes)
        neighs_nodes_1 = torch.LongTensor(list(neighs_1.keys()))
        sample_neighbors_2, neighs_2 = self.get_neighbor_nodes(neighs_nodes_1)
        neighs_nodes_2 = torch.LongTensor(list(neighs_2.keys()))

        features_1 = self.embedding_layer(neighs_nodes_1)
        features_origin = self.embedding_layer(nodes)
        agg_feat_1 = self.aggregator_1.forward(sample_neighbors_2, neighs_2)
        agg_feat_1 = torch.cat([features_1, agg_feat_1], dim=-1)
        agg_embed_1 = F.relu(self.fc_layer_1.forward(agg_feat_1))
        agg_feat_2 = self.aggregator_2.forward(sample_neighbors_1, neighs_1)
        agg_feat_2 = torch.cat([features_origin, agg_feat_2], dim=-1)
        agg_embed_2 = F.relu(self.fc_layer_2.forward(agg_feat_2))

        output = self.output_layer.forward(agg_embed_2)
        return F.softmax(output, dim=-1)


    def get_neighbor_nodes(self, nodes):
        to_neighs = [self.adj_list[int(node)] for node in nodes]
        if self.sample_num is not None:
            sample_neighbors = [set(random.sample(to_neigh, self.sample_num)) if len(to_neigh) >= self.sample_num else set(to_neigh) for to_neigh in to_neighs]
        else:
            sample_neighbors = to_neighs

        if self.add_self:
            sample_neighbors = [sample_neighbor | set([nodes[i]]) for i, sample_neighbor in enumerate(sample_neighbors)]

        unique_nodes_list = list(set.union(* sample_neighbors))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        return sample_neighbors ,unique_nodes