import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Mean_aggregator(nn.Module):
    def __init__(self, node_features):
        super(Mean_aggregator, self).__init__()
        self.node_features = node_features

    def forward(self, sample_neighbors, unique_nodes):
        unique_nodes_list = list(unique_nodes.keys())
        mask = torch.zeros(len(sample_neighbors), len(unique_nodes))
        col_idx = [unique_nodes[n] for sample_neighbor in sample_neighbors for n in sample_neighbor]
        row_idx = [i for i in range(len(sample_neighbors)) for _ in range(len(sample_neighbors[i]))]
        mask[row_idx, col_idx] = 1

        neighber_sum = mask.sum(-1, keepdim=True)
        mask = mask / neighber_sum
        output = torch.mm(mask, self.node_features[unique_nodes_list])
        return output
